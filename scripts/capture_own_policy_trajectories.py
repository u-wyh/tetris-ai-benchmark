"""Bounded, resumable CUDA8 own-policy capture; no training or CPU fallback."""
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

from own_policy_capture import CaptureNativeEnv, advance_round, save_commit, read_chain, sha
from restore_rank_contexts import observation_digest
from training.evaluation.evaluate_resumable import durable_json


def run(folder, max_chunks):
    import torch
    from sb3_contrib import MaskablePPO
    from training.train_vector_transaction import training_source_identity
    from training.train_transaction import global_rng_state, restore_global_rng
    from training.vector_env import make_vector_env
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from verify_feature_backend_benchmark import identical
    from resource_guard import now
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for path, digest in design['frozen_files'].items():
            if sha(path) != digest:
                raise ValueError('Frozen capture input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no own-policy capture fallback')
    if (design['seeds'] != list(range(100000, 100008)) or design['max_pieces'] != 20000 or
            design['chunk_rounds'] != 256 or not 1 <= max_chunks <= 20):
        raise ValueError('Capture bounded GPU8 protocol differs')
    if (training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('Capture numerical source/runtime differs')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = dict(design_sha256=sha(folder/'design.json'), chunk_rounds=256,
        ordered_lanes=design['seeds'], device='cuda', precision='highest', tf32=False,
        batching='active lanes only, in original lane order; no auto-reset')
    lock = (folder/'capture.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = None
    try:
        snapshot, previous, number = read_chain(folder, identity)
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
        env = make_vector_env(8, base_seed=100000, max_pieces=20000, env_class=CaptureNativeEnv,
            env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
        if snapshot is None:
            # Explicit resets happen AFTER both SB3.load calls, so seed42 cannot replace development seeds.
            views = [env.env_method('capture_reset', seed, indices=[i])[0]
                     for i, seed in enumerate(design['seeds'])]
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
            if any(w['state']['episode_seed'] != seed or w['state']['episode_count'] != 1
                   for w, seed in zip(snapshot['workers'], design['seeds'])):
                raise ValueError('Capture seed or episode count changed')
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
                interpretation='CUDA active-batch greedy development trajectories, not the prior CPU quality gate. '
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
