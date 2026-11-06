"""Profile a real CUDA8 rollout and independent PPO clones, preserving source."""

import argparse
import copy
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def profile_call(function):
    import torch
    torch.cuda.synchronize()
    profiler = cProfile.Profile()
    began = time.perf_counter()
    profiler.enable()
    try:
        result = function()
        torch.cuda.synchronize()
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - began
    rows = []
    for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        rows.append(dict(file=filename, line=line, function=name, calls=calls,
                         self_wall_seconds=own, cumulative_wall_seconds=cumulative))
    return result, dict(synchronized_wall_seconds=elapsed,
        top_self=sorted(rows, key=lambda r: r['self_wall_seconds'], reverse=True)[:40],
        top_cumulative=sorted(rows, key=lambda r: r['cumulative_wall_seconds'], reverse=True)[:40])


def rollout_digest(buffer):
    digest = hashlib.sha256()
    values = sorted(buffer.observations.items()) + [(name, getattr(buffer, name)) for name in
        ('actions', 'action_masks', 'returns', 'advantages', 'log_probs', 'values', 'rewards')]
    for name, value in values:
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run(folder):
    import torch
    from stable_baselines3.common.logger import configure
    from audit_ppo_update_path import clone_policy
    from training.train_transaction import global_rng_state, restore_global_rng
    from training.train_vector_transaction import restore_task, TransactionBitboardSearchCandidateEnv
    from training.vector_env import make_vector_env
    from verify_feature_backend_benchmark import identical
    from resource_guard import now, write_json

    if (folder/'training_profile.json').exists():
        raise FileExistsError('Preserve completed profile')
    design = json.loads((folder/'design.json').read_text())

    def unchanged():
        for name, sha in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Profile input changed: ' + name)

    unchanged()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    source = Path(design['source_context'])
    config = json.loads((source/'config.json').read_text())
    assert config['device'] == 'cuda' and config['ppo']['n_envs'] == 8
    env = make_vector_env(8, config['run_seed'], config['max_pieces'],
                          env_class=TransactionBitboardSearchCandidateEnv)
    model = None
    clones = []
    try:
        model, _, _ = restore_task(source, design['source_task'], config, env)
        policy = copy.deepcopy(model.policy.state_dict())
        adam = copy.deepcopy(model.policy.optimizer.state_dict())
        updates, steps = model._n_updates, model.num_timesteps
        model.set_logger(configure(str(folder/'logs/source'), []))
        _, callback = model._setup_learn(4096, reset_num_timesteps=False, progress_bar=False)
        callback.on_training_start({}, {})
        completed, rollout = profile_call(lambda: model.collect_rollouts(
            env, callback, model.rollout_buffer, model.n_steps, use_masking=True))
        callback.on_training_end()
        assert completed and model.num_timesteps - steps == 4096
        digest = rollout_digest(model.rollout_buffer)
        expected = json.loads(Path(design['reference_rollout_audit']).read_text())['rollout_sha256']
        assert digest == expected, 'Profiled rollout differs from independently audited rollout'
        rng = global_rng_state(model)
        update_profile = None
        try:
            for label in ('unprofiled', 'profiled'):
                clone = copy.copy(model)
                clone.policy = clone_policy(model.policy, model.device)
                clone.env = None
                clone.set_logger(configure(str(folder/'logs'/label), []))
                clones.append(clone)
                restore_global_rng(rng)
                if label == 'profiled':
                    _, update_profile = profile_call(clone.train)
                else:
                    clone.train()
                assert clone._n_updates - updates == 5
            assert identical(clones[0].policy.state_dict(), clones[1].policy.state_dict())
            assert identical(clones[0].policy.optimizer.state_dict(), clones[1].policy.optimizer.state_dict())
        finally:
            restore_global_rng(rng)
        assert identical(policy, model.policy.state_dict())
        assert identical(adam, model.policy.optimizer.state_dict()) and model._n_updates == updates
        unchanged()
        write_json(folder/'training_profile.json', dict(at=now(), status='passed', device='cuda', n_envs=8,
            source_optimizer_steps=0, samples=4096, rollout_sha256=digest,
            source_policy_adam_ppo_updates_unchanged=True, profile_and_control_update_exact=True,
            training_search_backend='bitboard', source_context=str(source),
            collection=rollout, ppo_train=update_profile,
            scope='cProfile wall time includes overhead; host-only function times include CUDA waits. '
                  'Workers are not profiled. Cumulative rows overlap; stage CUDA synchronization '
                  'does not measure GPU busy time. No throughput, policy quality or adoption claim.'))
    finally:
        for clone in clones:
            clone.logger.close()
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
