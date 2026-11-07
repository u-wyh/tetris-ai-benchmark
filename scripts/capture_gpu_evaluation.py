"""Explicit resumable CUDA8 evaluation for any eight unique ordered uint32 seeds."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from own_policy_capture import advance_round, save_commit, read_chain, sha
from parallel_capture import ParallelNativeEnv, advance_parallel, validate_seeds
from restore_rank_contexts import observation_digest
from training.evaluation.evaluate_resumable import durable_json

EVALUATION_VERSION = 'ordered-gpu8-v1'


def evaluation_sources():
    return {str(ROOT/'scripts'/name): sha(ROOT/'scripts'/name) for name in
            ['capture_gpu_evaluation.py', 'parallel_capture.py', 'own_policy_capture.py',
             'restore_rank_contexts.py']}


def check_inputs(design):
    if evaluation_sources() != design['evaluation_sources']:
        raise ValueError('GPU evaluation capture source differs')
    for path, digest in design['frozen_files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen GPU evaluation input changed: '+path)
    for spec in design['models'].values():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('GPU evaluation model checksum differs')


def validate_snapshot(snapshot, design):
    if snapshot is None:
        return
    if len(snapshot['workers']) != 8 or len(snapshot['views']) != 8:
        raise ValueError('Snapshot must preserve all eight lanes')
    pieces = []
    for seed, worker, view in zip(design['seeds'], snapshot['workers'], snapshot['views']):
        state = worker['state']
        if (state['episode_seed'] != seed or state['episode_count'] != 1 or state['max_pieces'] != 20000 or
                type(worker['active']) is not bool or worker['active'] == state['finished'] or
                (view is not None) != worker['active'] or
                not 0 <= state['pieces'] <= snapshot['round'] <= 20000 or
                (worker['active'] and state['pieces'] != snapshot['round'])):
            raise ValueError('Cached seed order, episode, cap or active lane differs')
        pieces.append(state['pieces'])
    if sum(pieces) != snapshot['environment_steps']:
        raise ValueError('Worker pieces differ from committed unique environment work')


def prepare(folder):
    """Validate immutable inputs/runtime before cache pickle, locks, models or workers."""
    from training.train_vector_transaction import training_source_identity
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from training.native_feature_runtime import load_runtime
    design = json.loads((folder/'design.json').read_text())
    if (design.get('evaluation_version') != EVALUATION_VERSION or
            design.get('dispatch_version') != 'unique-seed-parallel-v1' or
            design.get('dispatch_mode') not in ('parallel', 'serial_oracle') or
            design.get('device') != 'cuda' or design.get('n_envs') != 8 or
            design.get('max_pieces') != 20000 or design.get('chunk_rounds') != 256 or
            set(design.get('models', {})) != {'candidate', 'shadow'}):
        raise ValueError('Explicit GPU8 evaluation schema, dispatch or cap differs')
    validate_seeds(design['seeds'], 8)
    check_inputs(design)
    if (training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('GPU evaluation numerical source/runtime differs')
    load_runtime(design['native_feature_runtime'])
    identity = dict(evaluation_version=EVALUATION_VERSION, design_sha256=sha(folder/'design.json'),
        evaluation_sources=design['evaluation_sources'],
        dispatch_version=design['dispatch_version'], dispatch_mode=design['dispatch_mode'],
        chunk_rounds=256, max_pieces=20000, ordered_lanes=design['seeds'], device='cuda',
        precision='highest', tf32=False, batching='active lanes in design order; no auto-reset')
    snapshot, previous, count = read_chain(folder, identity)
    validate_snapshot(snapshot, design)
    if (folder/'capture_summary.json').exists():
        raise FileExistsError('Evaluation already complete; inspect rather than starting it again')
    return design, identity, (snapshot, previous, count)


def reset_ordered(env, seeds):
    validate_seeds(seeds, env.num_envs)
    return [env.env_method('capture_reset', seed, indices=[lane])[0] for lane, seed in enumerate(seeds)]


def run(folder, max_chunks):
    import torch
    from sb3_contrib import MaskablePPO
    from training.train_transaction import global_rng_state, restore_global_rng
    from training.vector_env import make_vector_env
    from verify_feature_backend_benchmark import identical
    from resource_guard import now
    if type(max_chunks) is not int or not 1 <= max_chunks <= 20:
        raise ValueError('Each call permits one to twenty new bounded chunks')
    design, identity, _ = prepare(folder)
    def unchanged():
        check_inputs(design)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no own-policy capture fallback')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    lock = (folder/'capture.lock').open('a')
    env = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unchanged()
        snapshot, previous, number = read_chain(folder, identity)
        validate_snapshot(snapshot, design)
        if (folder/'capture_summary.json').exists():
            raise FileExistsError('Evaluation already complete; inspect rather than starting it again')
        reused_chunks = number
        models = {}
        for name in ['candidate', 'shadow']:
            spec = design['models'][name]
            if sha(spec['path']) != spec['sha256']:
                raise ValueError('Frozen capture model changed')
            model = MaskablePPO.load(spec['path'], device='cuda')
            assert model.get_env() is None and model.device.type == 'cuda'
            model.policy.set_training_mode(False)
            if [model.num_timesteps, model._n_updates] != spec['counts']:
                raise ValueError('Capture source model counters differ')
            models[name] = model
        expected = {name: (copy.deepcopy(model.policy.state_dict()),
            copy.deepcopy(model.policy.optimizer.state_dict()), model.num_timesteps, model._n_updates)
            for name, model in models.items()}
        candidate, shadow = models['candidate'], models['shadow']
        env = make_vector_env(8, base_seed=0, max_pieces=20000, env_class=ParallelNativeEnv,
            env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
        if snapshot is None:
            # Explicit resets happen AFTER both SB3.load calls, so seed42 cannot replace development seeds.
            views = reset_ordered(env, design['seeds'])
            snapshot = dict(round=0, environment_steps=0)
        else:
            if len(snapshot['workers']) != 8 or len(snapshot['views']) != 8:
                raise ValueError('Capture snapshot must contain all eight lanes')
            restore_global_rng(snapshot['rng'])
            views = [env.env_method('capture_restore', saved, indices=[i])[0]
                     for i, saved in enumerate(snapshot['workers'])]
            restored = env.env_method('capture_export')
            if not identical(restored, snapshot['workers']):
                raise ValueError('Complete worker state changed during capture restoration')
            if [None if v is None else observation_digest(**v) for v in views] != snapshot['views']:
                raise ValueError('Restored capture observations or masks differ')
        rng = global_rng_state(candidate)
        def audit():
            for name, model in models.items():
                weights, adam, steps, updates = expected[name]
                if (not identical(weights, model.policy.state_dict()) or
                        not identical(adam, model.policy.optimizer.state_dict()) or
                        (steps, updates) != (model.num_timesteps, model._n_updates)):
                    raise ValueError('Frozen model policy, Adam or counters changed')
            if not identical(rng, global_rng_state(candidate)):
                raise ValueError('Greedy capture or shadow changed global sampling RNG')
            return dict(policy_adam_counts_unchanged=True, global_sampling_rng_unchanged=True,
                model_counts={name: [m.num_timesteps, m._n_updates] for name, m in models.items()},
                optimizer_steps=0)
        def checkpoint():
            snapshot.update(workers=env.env_method('capture_export'), rng=global_rng_state(candidate),
                views=[None if view is None else observation_digest(**view) for view in views])
            validate_snapshot(snapshot, design)
        if number == 0:
            checkpoint()
            previous = save_commit(folder, 0, identity, None, snapshot, [], audit())
            number = 1
        computed = []
        process_steps = process_rounds = 0
        for _ in range(max_chunks):
            if not any(view is not None for view in views):
                break
            rows = []
            for _ in range(min(256, 20000-snapshot['round'])):
                if not any(view is not None for view in views):
                    break
                if design['dispatch_mode'] == 'parallel':
                    rows.extend(advance_parallel(env, views, candidate, shadow, snapshot['round'], design['seeds']))
                else:  # Explicit GPU serial oracle for functional acceptance, never a device fallback.
                    rows.extend(advance_round(env, views, candidate, shadow, snapshot['round']))
                snapshot['round'] += 1
                process_rounds += 1
            snapshot['environment_steps'] += len(rows)
            if snapshot['environment_steps'] > 160000 or not rows:
                raise ValueError('Capture work budget exceeded or made no progress')
            checkpoint(); result = audit(); unchanged()
            previous = save_commit(folder, number, identity, previous, snapshot, rows, result)
            computed.append(number); number += 1; process_steps += len(rows)
            durable_json(folder/'progress.json', dict(at=now(), pid=os.getpid(), identity=identity,
                last_manifest_sha256=previous, round=snapshot['round'],
                environment_steps=snapshot['environment_steps'],
                active_lanes=[i for i, v in enumerate(views) if v is not None],
                completed_capture=not any(v is not None for v in views)))
        checkpoint(); result = audit(); unchanged()
        process = dict(at=now(), pid=os.getpid(), reused_committed_chunks=reused_chunks,
            computed_chunks=computed, environment_steps_this_process=process_steps,
            candidate_forward_calls_this_process=process_rounds, shadow_forward_calls_this_process=process_rounds,
            optimizer_steps=0, model_audit=result, last_manifest_sha256=previous)
        durable_json(folder/f'process_{os.getpid()}.json', process)
        if not any(view is not None for view in views):
            endpoints = []
            for lane, worker in enumerate(snapshot['workers']):
                state = worker['state']
                endpoints.append(dict(lane=lane, seed=state['episode_seed'], steps=state['pieces'],
                    score=state['core'].score, lines=state['core'].lines,
                    game_over=state['core'].phase == 'over',
                    cap_censored=state['pieces'] == 20000 and state['core'].phase != 'over'))
            durable_json(folder/'capture_summary.json', dict(at=now(), status='passed', identity=identity,
                endpoints=endpoints, environment_steps=snapshot['environment_steps'],
                final_round=snapshot['round'], optimizer_steps=0, process=process,
                interpretation='CUDA8 evaluation completed for its declared seed order; completion is not quality acceptance. '
                    'Shadow choices never executed. No policy promotion or training.'))
        print(json.dumps(process))
    finally:
        if env is not None:
            env.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-chunks', type=int, default=20)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.max_chunks)
