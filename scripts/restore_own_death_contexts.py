"""Recover selected own-policy pre-action states using immutable saved actions."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from own_policy_capture import CaptureNativeEnv, sha
from restore_rank_contexts import save_context, load_context, observation_digest
from training.evaluation.evaluate_resumable import durable_json
from verify_feature_backend_benchmark import identical


def load_verified_chunk(directory, manifest_sha, identity):
    """The externally frozen manifest and both files are checked before pickle."""
    if sha(directory/'manifest.json') != manifest_sha:
        raise ValueError('Frozen capture manifest checksum differs')
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['identity'] != identity or set(manifest['files']) != {'state.pkl', 'rows.npz'} or
            any(sha(directory/name) != expected for name, expected in manifest['files'].items())):
        raise ValueError('Capture chunk identity or file checksum differs')
    saved = pickle.loads((directory/'state.pkl').read_bytes())
    if saved['round'] != manifest['round'] or saved['environment_steps'] != manifest['environment_steps']:
        raise ValueError('Capture snapshot counters differ')
    with np.load(directory/'rows.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if any(value.dtype.hasobject or len(value) != manifest['rows'] for value in data.values()):
        raise ValueError('Capture chunk array count or dtype differs')
    return saved, data, manifest


def restore_vector(env, saved):
    if len(saved['workers']) != env.num_envs or len(saved['views']) != env.num_envs:
        raise ValueError('Capture snapshot must contain every lane, including stopped lanes')
    views = [env.env_method('capture_restore', state, indices=[lane])[0]
             for lane, state in enumerate(saved['workers'])]
    if (not identical(saved['workers'], env.env_method('capture_export')) or
            [None if view is None else observation_digest(**view) for view in views] != saved['views']):
        raise ValueError('Full capture restoration changed worker state, observations or masks')
    return views


def replay_chunk(env, views, data, start_round, end_round, targets):
    if not 0 < end_round-start_round <= 256:
        raise ValueError('Replay is limited to one original bounded chunk')
    contexts = {}; cursor = steps = 0
    count = len(data['lane'])
    for decision_round in range(start_round, end_round):
        active = [lane for lane, view in enumerate(views) if view is not None]
        stop = cursor
        while stop < count and int(data['round'][stop]) == decision_round:
            stop += 1
        if not active or list(map(int, data['lane'][cursor:stop])) != active:
            raise ValueError('Replay saved rows do not match the original ordered active lanes')
        # Extract all target lanes BEFORE advancing any lane in this round.
        for target in targets:
            if target['round'] != decision_round:
                continue
            lane = target['lane']
            if lane not in active or observation_digest(**views[lane]) != target['observation_digest']:
                raise ValueError('Target pre-action view differs from original capture')
            row = cursor+active.index(lane)
            if (int(data['action'][row]) != target['candidate_action'] or
                    int(data['shadow_action'][row]) != target['shadow_action']):
                raise ValueError('Frozen selected target actions differ')
            contexts[target['key']] = dict(state=env.env_method('get_state', indices=[lane])[0],
                                           view=copy.deepcopy(views[lane]))
        for row, lane in zip(range(cursor, stop), active):
            view = views[lane]; action, shadow = int(data['action'][row]), int(data['shadow_action'][row])
            expected_digest = dict(observation=str(data['observation_sha256'][row]), mask=str(data['mask_sha256'][row]))
            if (observation_digest(**view) != expected_digest or not view['mask'][action] or not view['mask'][shadow] or
                    not np.array_equal(view['observation']['candidate'][action], data['candidate_features'][row]) or
                    not np.array_equal(view['observation']['candidate'][shadow], data['shadow_features'][row])):
                raise ValueError('Saved-action replay observation, mask or selected features differ')
            outcome = env.env_method('capture_step', action, indices=[lane])[0]
            if outcome is None or any(value != data[key][row] for key, value in outcome['row'].items()):
                raise ValueError('Saved-action replay reward, termination, episode seed or information differs')
            steps += 1; views[lane] = outcome['view']
        cursor = stop
    if cursor != count or set(contexts) != {target['key'] for target in targets}:
        raise ValueError('Replay did not consume the complete chunk or recover every target')
    return contexts, views, steps


def sampling_rng():
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def run(folder, through_chunks):
    import torch
    from training.train_transaction import restore_global_rng
    from training.train_vector_transaction import training_source_identity
    from training.vector_env import make_vector_env
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from resource_guard import now
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for path, expected in design['frozen_files'].items():
            if sha(path) != expected:
                raise ValueError('Frozen context recovery input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no context recovery host/CPU fallback')
    if (training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('Context recovery numerical source/runtime differs')
    torch.set_num_threads(1)
    selection = json.loads(Path(design['targets_path']).read_text())
    targets = selection['targets']; chunks = sorted({target['source_chunk'] for target in targets})
    if len(targets) != 2 or len(chunks) > 2 or not 1 <= through_chunks <= len(chunks):
        raise ValueError('Frozen all-death target recovery scope differs')
    capture = Path(selection['capture_dir']); identity = selection['capture_identity']
    run_identity = dict(design_sha256=sha(folder/'design.json'), targets_sha256=sha(design['targets_path']),
                        capture_identity=identity)
    lock = (folder/'recovery.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = None; computed = []; reused = []; total_steps = 0
    def context_identity(target):
        return dict(run_identity, target=target)
    try:
        for number in chunks[:through_chunks]:
            chosen = [target for target in targets if target['source_chunk'] == number]
            receipt_path = folder/f'replay_{number:04d}.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt['identity'] != run_identity or receipt['status'] != 'passed':
                    raise ValueError('Committed replay receipt identity differs')
                for target in chosen:
                    directory = folder/'contexts'/target['key']
                    if sha(directory/'manifest.json') != receipt['context_manifests'][target['key']]:
                        raise ValueError('Committed recovered context changed')
                    load_context(directory, context_identity(target))
                reused.append(number); continue
            def load(number):
                directory = capture/f'chunk_{number:04d}'
                return load_verified_chunk(directory, design['frozen_files'][str(directory/'manifest.json')], identity)
            initial, _, _ = load(number-1); expected, data, manifest = load(number)
            if len(initial['workers']) != 8 or len(expected['workers']) != 8:
                raise ValueError('Recovery must retain all eight worker slots')
            if env is None:
                env = make_vector_env(8, max_pieces=20000, env_class=CaptureNativeEnv,
                    env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
            restore_global_rng(initial['rng']); rng = sampling_rng()
            views = restore_vector(env, initial)
            contexts, views, steps = replay_chunk(env, views, data, initial['round'], expected['round'], chosen)
            if (not identical(env.env_method('capture_export'), expected['workers']) or
                    [None if view is None else observation_digest(**view) for view in views] != expected['views'] or
                    not identical(rng, sampling_rng())):
                raise ValueError('Replay full end state or parent sampling RNG differs')
            if steps != manifest['rows'] or steps > 2048:
                raise ValueError('Replay actual work differs or exceeds original chunk budget')
            for target in chosen:
                context = contexts[target['key']]
                if context['state']['pieces'] != target['step']-1:
                    raise ValueError('Recovered context is not immediately before the selected action')
                save_context(folder/'contexts'/target['key'], context_identity(target),
                             context['state'], **context['view'])
            unchanged()
            durable_json(receipt_path, dict(at=now(), status='passed', identity=run_identity,
                chunk=number, pid=os.getpid(), start_round=initial['round'], end_round=expected['round'],
                actual_replayed_environment_steps=steps, policy_forward_calls=0, optimizer_steps=0,
                full_eight_worker_end_state_and_views_exact=True, parent_sampling_rng_unchanged=True,
                context_manifests={target['key']:sha(folder/'contexts'/target['key']/'manifest.json') for target in chosen}))
            total_steps += steps; computed.append(number)
        process = dict(at=now(), pid=os.getpid(), computed_chunks=computed, reused_chunks=reused,
            actual_environment_steps_this_process=total_steps, policy_forward_calls=0, optimizer_steps=0)
        durable_json(folder/f'process_{os.getpid()}.json', process)
        if through_chunks == len(chunks):
            receipts = [json.loads((folder/f'replay_{number:04d}.json').read_text()) for number in chunks]
            steps = sum(receipt['actual_replayed_environment_steps'] for receipt in receipts)
            if steps > 4096:
                raise ValueError('All-target recovery exceeded predeclared replay step budget')
            durable_json(folder/'contexts_summary.json', dict(at=now(), status='passed', identity=run_identity,
                targets=targets, source_chunks=chunks, actual_replayed_environment_steps=steps,
                policy_forward_calls=0, optimizer_steps=0, process=process,
                interpretation='Exact saved-action reconstruction of all selected true-death pre-action states; '
                    'no new action sampling, intervention or training. CUDA present, GPU telemetry; no policy forward.'))
        print(json.dumps(process))
    finally:
        if env is not None:
            env.close()
        lock.close()


def verify_restart(folder):
    from resource_guard import now
    if (folder/'restart_receipt.json').exists():
        raise RuntimeError('Recovery restart verification already completed; inspect, do not repeat')
    if any(folder.glob('replay_*.json')) or (folder/'contexts').exists():
        raise RuntimeError('Recovery outputs already exist; use bounded direct continuation')
    def execute(count):
        command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(folder),
                   '--through-chunks', str(count)]
        child = subprocess.Popen(command)
        if child.wait():
            raise RuntimeError('Recovery child failed: '+str(child.pid))
        return child.pid
    first = execute(1)
    protected = [*folder.glob('replay_*.json'), *folder.glob('contexts/*/*')]
    before = {str(path):sha(path) for path in protected if path.is_file()}
    second = execute(2)
    process = json.loads((folder/f'process_{second}.json').read_text())
    if (first == second or len(process['reused_chunks']) != 1 or len(process['computed_chunks']) != 1 or
            any(sha(path) != expected for path, expected in before.items())):
        raise ValueError('Actual context recovery restart failed or rewrote its first committed context')
    durable_json(folder/'restart_receipt.json', dict(at=now(), status='passed', first_pid=first, second_pid=second,
        first_chunk_receipt_and_context_files_unchanged=True, protected_files=before, second_process=process))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--through-chunks', type=int, choices=[1,2], default=2)
    parser.add_argument('--verify-restart', action='store_true')
    args = parser.parse_args()
    if args.verify_restart:
        verify_restart(args.run_dir.resolve())
    else:
        run(args.run_dir.resolve(), args.through_chunks)
