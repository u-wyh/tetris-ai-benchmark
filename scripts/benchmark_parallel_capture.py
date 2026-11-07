"""Frozen ABBA comparison of serial and concurrent complete GPU capture rounds."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from own_policy_capture import advance_round, greedy_details, save_commit, read_chain, sha
from parallel_capture import ParallelNativeEnv, advance_parallel, validate_seeds
from restore_own_death_contexts import load_verified_chunk, restore_vector
from restore_rank_contexts import observation_digest
from training.evaluation.evaluate_resumable import durable_json
from verify_feature_backend_benchmark import identical


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from training.vector_env import make_vector_env
    from training.train_transaction import global_rng_state, restore_global_rng
    from training.train_vector_transaction import training_source_identity
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from resource_guard import now
    if any(folder.glob('arm_*')) or (folder/'benchmark.json').exists():
        raise FileExistsError('ABBA already started; preserve work instead of repeating timings')
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for path, digest in design['frozen_files'].items():
            if sha(path) != digest:
                raise ValueError('Frozen parallel capture input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no parallel capture CPU fallback')
    if (training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('GPU comparison numerical source/runtime differs')
    if (design['order'] != ['serial', 'parallel', 'parallel', 'serial'] or
            design['rounds_per_arm'] != 512 or design['seeds'] != list(range(100000, 100008))):
        raise ValueError('Predeclared bounded ABBA protocol differs')
    validate_seeds(design['seeds'], 8)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    source = []
    previous = None
    for number, spec in enumerate(design['source_chunks']):
        snapshot, data, manifest = load_verified_chunk(Path(spec['directory']), spec['manifest_sha256'],
                                                       design['source_identity'])
        if (manifest['number'] != number or manifest['previous_manifest_sha256'] != previous or
                snapshot['round'] != number*256 or snapshot['environment_steps'] != number*2048):
            raise ValueError('Source prefix is not the exact first 512 CUDA8 rounds')
        source.append((snapshot, data)); previous = spec['manifest_sha256']
    if len(source) != 3:
        raise ValueError('Initial plus two immutable source chunks required')
    initial = source[0][0]
    if [w['state']['episode_seed'] for w in initial['workers']] != design['seeds']:
        raise ValueError('Worker seed order differs before dispatch')
    models = {}
    for name, spec in design['models'].items():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Capture checkpoint checksum differs')
        model = MaskablePPO.load(spec['path'], device='cuda')
        assert model.get_env() is None and model.device.type == 'cuda'
        assert [model.num_timesteps, model._n_updates] == spec['counts']
        model.policy.set_training_mode(False)
        models[name] = model
    expected = {name: (copy.deepcopy(m.policy.state_dict()), copy.deepcopy(m.policy.optimizer.state_dict()),
                       m.num_timesteps, m._n_updates) for name, m in models.items()}
    candidate, shadow = models['candidate'], models['shadow']
    def audit():
        if any(not identical(expected[name], (m.policy.state_dict(), m.policy.optimizer.state_dict(),
                                              m.num_timesteps, m._n_updates)) for name, m in models.items()):
            raise ValueError('Frozen model policy, Adam or counters changed')
        if not identical(initial['rng'], global_rng_state(candidate)):
            raise ValueError('Parallel or serial capture changed parent sampling RNG')
        return dict(policy_adam_counts_unchanged=True, global_sampling_rng_unchanged=True, optimizer_steps=0,
                    model_counts={name: [m.num_timesteps, m._n_updates] for name, m in models.items()})
    env = None
    arms = []
    try:
        env = make_vector_env(8, base_seed=100000, max_pieces=20000, env_class=ParallelNativeEnv,
            env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
        worker_pids = [p.pid for p in env.processes]
        for arm, dispatch in enumerate(design['order']):
            directory = folder/f'arm_{arm}_{dispatch}'; directory.mkdir()
            restore_global_rng(initial['rng'])
            views = restore_vector(env, initial)
            observations = {key: np.stack([v['observation'][key] for v in views])
                            for key in views[0]['observation']}
            masks = np.stack([v['mask'] for v in views])
            chosen = greedy_details(candidate.policy, observations, masks)
            greedy_details(shadow.policy, observations, masks, chosen['action'])
            audit()  # Warmup has no environment steps or sampling RNG effects.
            identity = dict(design_sha256=sha(folder/'design.json'), chunk_rounds=256,
                ordered_lanes=design['seeds'], device='cuda', precision='highest', tf32=False,
                dispatch=dispatch, arm=arm)
            snapshot = copy.deepcopy(initial)
            previous = save_commit(directory, 0, identity, None, snapshot, [], audit())
            torch.cuda.synchronize()
            started = time.perf_counter()
            for number in [1, 2]:
                rows = []
                for decision_round in range((number-1)*256, number*256):
                    if dispatch == 'serial':
                        rows.extend(advance_round(env, views, candidate, shadow, decision_round))
                    else:
                        rows.extend(advance_parallel(env, views, candidate, shadow, decision_round, design['seeds']))
                snapshot.update(round=number*256, environment_steps=number*2048,
                    workers=env.env_method('capture_export'), rng=global_rng_state(candidate),
                    views=[None if v is None else observation_digest(**v) for v in views])
                if len(rows) != 2048:
                    raise ValueError('Unexpected stopped lane or simulation budget')
                previous = save_commit(directory, number, identity, previous, snapshot, rows, audit())
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            recovered, final_sha, count = read_chain(directory, identity)
            if count != 3 or final_sha != previous or not identical(recovered, source[2][0]):
                raise ValueError('Complete workers, RNG, views or counters differ from frozen source')
            for number in [1, 2]:
                with np.load(directory/f'chunk_{number:04d}'/'rows.npz', allow_pickle=False) as archive:
                    observed = {key: archive[key] for key in archive.files}
                if not identical(observed, source[number][1]):
                    raise ValueError('Natural actions, probabilities, pre-views, features or outcomes differ')
            unchanged()
            arms.append(dict(arm=arm, dispatch=dispatch, seconds=elapsed, environment_steps=4096,
                candidate_forward_calls=512, shadow_forward_calls=512, warmup_forward_calls=2,
                final_manifest_sha256=previous, trace_and_complete_state_match_source=True))
            durable_json(folder/'progress.json', dict(at=now(), pid=os.getpid(), completed_arms=arms,
                actual_environment_steps=sum(a['environment_steps'] for a in arms), optimizer_steps=0))
        ratios = [arms[0]['seconds']/arms[1]['seconds'], arms[3]['seconds']/arms[2]['seconds']]
        speedup = statistics.geometric_mean(ratios)
        supported = speedup >= design['speedup_at_least'] and all(r > 1 for r in ratios)
        durable_json(folder/'benchmark.json', dict(at=now(), status='passed', pid=os.getpid(), worker_pids=worker_pids,
            arms=arms, paired_speedups=ratios, geometric_speedup=speedup,
            eligible_for_capture_integration=supported, optimizer_steps=0, actual_environment_steps=16384,
            actual_candidate_forward_calls=2048, actual_shadow_forward_calls=2048,
            warmup_forward_calls=8, policy_adam_counts_rng_unchanged=True,
            source_trajectories_and_complete_state_exact=True,
            timing_scope='Both GPU forwards, ordered environment steps and results, digest/trace assembly, '
                'full state exports, atomic 256-round commits, audits and compression/fsync. '
                'Excludes model/worker startup, initial restore/commit, warmup, source comparisons/hashes.',
            policy_promotion='none', training_speedup_claim=False,
            interpretation='Frozen single-session ABBA GPU evaluation comparison; not PPO training performance. '
                           'Integration requires separate actual process recovery verification.'))
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
