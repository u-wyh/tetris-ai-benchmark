"""Persist one immutable source rollout and describe critic target residuals."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def array_manifest(arrays):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                     sha256=hashlib.sha256(value.tobytes()).hexdigest())
            for key, value in sorted(arrays.items())}


def residual_metrics(values, returns):
    values, returns = np.asarray(values, dtype=np.float64), np.asarray(returns, dtype=np.float64)
    if values.size == 0:
        return dict(samples=0)
    if values.shape != returns.shape or not (np.isfinite(values).all() and np.isfinite(returns).all()):
        raise ValueError('Invalid calibration arrays')
    residual = returns - values
    mse, bias, variance = float(np.mean(residual**2)), float(residual.mean()), float(returns.var())
    return dict(samples=values.size, residual_mean=bias, residual_std=float(residual.std()),
                rmse=mse**0.5, mae=float(np.abs(residual).mean()),
                explained_variance=1-float(residual.var())/variance if variance else None,
                mean_bias_squared_over_mse=bias**2/mse if mse else None,
                value_mean=float(values.mean()), return_mean=float(returns.mean()),
                value_std=float(values.std()), return_std=variance**0.5)


def verify_targets(arrays, gamma, gae_lambda):
    """Reconstruct SB3 targets from durable rewards/starts/final bootstrap."""
    import torch
    from stable_baselines3.common.buffers import RolloutBuffer
    rewards, values = arrays['rewards'], arrays['values']
    buffer = SimpleNamespace(buffer_size=len(rewards), rewards=rewards, values=values,
        episode_starts=arrays['episode_starts'], gamma=gamma, gae_lambda=gae_lambda,
        advantages=np.zeros_like(values))
    RolloutBuffer.compute_returns_and_advantage(buffer,
        torch.from_numpy(arrays['last_values'].copy()), arrays['last_dones'])
    np.testing.assert_array_equal(buffer.advantages, arrays['advantages'])
    np.testing.assert_array_equal(buffer.returns, arrays['returns'])
    np.testing.assert_array_equal(arrays['episode_starts'][1:].astype(bool), arrays['step_dones'][:-1])
    np.testing.assert_array_equal(arrays['last_dones'], arrays['step_dones'][-1])


def save_dataset(folder, arrays, metadata):
    """Publish manifest only after an exact pickle-free round trip."""
    target = folder/'rollout.npz'
    if target.exists() or (folder/'dataset.json').exists():
        raise FileExistsError('Preserve existing durable rollout')
    manifest = array_manifest(arrays)
    temporary = folder/'rollout.tmp'
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as archive:
        reloaded = {key: archive[key] for key in archive.files}
    if array_manifest(reloaded) != manifest:
        raise RuntimeError('Dataset round trip differs')
    verify_targets(reloaded, metadata['gamma'], metadata['gae_lambda'])
    os.replace(temporary, target)
    from resource_guard import write_json
    write_json(folder/'dataset.json', dict(metadata, arrays=manifest,
        file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), round_trip_exact=True,
        saved_gae_reconstruction_exact=True))
    return reloaded


def run(folder):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from training.train_vector_transaction import restore_task, TransactionCompactSearchCandidateEnv
    from training.vector_env import make_vector_env
    from profile_training_stages import rollout_digest
    from verify_feature_backend_benchmark import identical
    from resource_guard import now, write_json

    if any((folder/name).exists() for name in ('rollout.npz', 'rollout.tmp', 'calibration.json')):
        raise FileExistsError('Inspect existing sample; never resample automatically')
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for name, sha in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Frozen calibration input changed: '+name)
    unchanged()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    source = Path(design['source_context'])
    config = json.loads((source/'config.json').read_text())
    assert config['device'] == 'cuda' and config['ppo']['n_envs'] == 8

    class Markers(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []
        def _on_step(self):
            infos = self.locals['infos']
            row = dict(step_dones=np.array(self.locals['dones'], dtype=bool, copy=True),
                       raw_env_rewards=np.array(self.locals['rewards'], copy=True),
                       step_truncated=np.array([i.get('TimeLimit.truncated', False) for i in infos]))
            for name in ('new_holes', 'holes_before', 'holes_after', 'height_after', 'pieces', 'cleared_lines'):
                row['info_'+name] = np.array([i[name] for i in infos])
            self.rows.append(row)
            return True

    env = make_vector_env(8, config['run_seed'], config['max_pieces'],
                          env_class=TransactionCompactSearchCandidateEnv)
    model = None
    try:
        model, _, _ = restore_task(source, design['source_task'], config, env)
        policy, adam = copy.deepcopy(model.policy.state_dict()), copy.deepcopy(model.policy.optimizer.state_dict())
        steps, updates = model.num_timesteps, model._n_updates
        model.set_logger(configure(str(folder/'logs'), []))
        markers = Markers()
        _, callback = model._setup_learn(4096, callback=markers, reset_num_timesteps=False, progress_bar=False)
        buffer = model.rollout_buffer
        original = buffer.compute_returns_and_advantage
        bootstrap = {}
        def capture(last_values, dones):
            bootstrap['last_values'] = last_values.detach().cpu().numpy().flatten().copy()
            bootstrap['last_dones'] = np.array(dones, copy=True)
            return original(last_values, dones)
        buffer.compute_returns_and_advantage = capture
        try:
            callback.on_training_start({}, {})
            assert model.collect_rollouts(env, callback, buffer, model.n_steps, use_masking=True)
            callback.on_training_end()
        finally:
            del buffer.compute_returns_and_advantage
        assert model.num_timesteps-steps == 4096 and len(markers.rows) == 512
        digest = rollout_digest(buffer)
        assert digest == json.loads(Path(design['reference_rollout_audit']).read_text())['rollout_sha256']
        assert identical(policy, model.policy.state_dict()) and identical(adam, model.policy.optimizer.state_dict())
        assert model._n_updates == updates
        arrays = {'obs_'+key: value.copy() for key, value in buffer.observations.items()}
        for name in ('actions', 'action_masks', 'returns', 'advantages', 'log_probs', 'values', 'rewards', 'episode_starts'):
            arrays[name] = getattr(buffer, name).copy()
        arrays.update(bootstrap)
        arrays.update({key: np.stack([r[key] for r in markers.rows]) for key in markers.rows[0]})
        arrays['step_terminated'] = arrays['step_dones'] & ~arrays['step_truncated']
        unchanged()
        arrays = save_dataset(folder, arrays, dict(at=now(), status='passed', samples=4096,
            n_envs=8, device='cuda', gamma=model.gamma, gae_lambda=model.gae_lambda,
            rollout_sha256=digest, source_context=str(source), source_task=design['source_task'],
            source_steps=steps, source_ppo_updates=updates, source_optimizer_updates=0,
            source_policy_adam_ppo_updates_unchanged=True, training_search_backend='bitboard_compact',
            runtime=design['runtime'], frozen_files=design['frozen_files'],
            scope='Historical source code SHA unavailable. Current runtime and exact historical sample pinned. '
                  'Rewards include SB3 timeout bootstrap; raw_env_rewards precede it. Array axes time,worker.'))
        values, returns = arrays['values'], arrays['returns']
        groups = {}
        for name, mask in [('terminated_step', arrays['step_terminated']),
            ('truncated_step', arrays['step_truncated']), ('continuing_step', ~arrays['step_dones']),
            ('new_holes_step', arrays['info_new_holes'] > 0), ('no_new_holes_step', arrays['info_new_holes'] == 0)]:
            groups[name] = residual_metrics(values[mask], returns[mask])
        unchanged()
        write_json(folder/'calibration.json', dict(at=now(), status='passed',
            overall=residual_metrics(values, returns), groups=groups,
            workers=[residual_metrics(values[:, i], returns[:, i]) for i in range(8)],
            scope='Residual=GAE return minus current value, not unbiased Monte Carlo truth. '
                  'One 512-step window per worker, correlated samples, bootstrap and truncation limits. '
                  'Groups label the current transition, not all preceding steps of an episode. '
                  'No learning, causal claim, quality promotion or automatic hyperparameter selection.'))
    finally:
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
