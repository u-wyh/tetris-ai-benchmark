"""Collect frozen CUDA8 calibration data with atomic, resumable segment commits."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from collect_critic_calibration import save_dataset, residual_metrics
from resource_guard import now, write_json, operation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fresh_model(path, env, base_seed, device='cuda'):
    from sb3_contrib import MaskablePPO
    from training.vector_env import worker_seeds
    model = MaskablePPO.load(path, env=env, device=device, force_reset=True)
    # _setup_model in load() reseeds env from the historical model's seed.
    expected = worker_seeds(base_seed, env.num_envs)
    if list(env.seed(base_seed)) != expected:
        raise RuntimeError('Fresh worker seed assignment differs')
    return model


def save_checkpoint(folder, model, env, identity):
    from training.train_transaction import global_rng_state
    from training.train_vector_transaction import observation_and_mask_digests, observations_equal
    observations, digests = observation_and_mask_digests(env)
    if not observations_equal(model._last_obs, observations):
        raise RuntimeError('Collector observation differs from workers')
    state = dict(trainer=global_rng_state(model), workers=env.env_method('get_state'))
    with (folder/'state.pkl').open('xb') as stream:
        pickle.dump(state, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    write_json(folder/'checkpoint.json', dict(identity, at=now(), state_sha256=sha(folder/'state.pkl'),
        num_timesteps=model.num_timesteps, worker_digests=digests))


def restore_checkpoint(folder, model, env, identity):
    from training.train_transaction import restore_global_rng, global_rng_state
    from training.train_vector_transaction import observation_and_mask_digests, observations_equal
    from verify_feature_backend_benchmark import identical
    receipt = json.loads((folder/'checkpoint.json').read_text())
    if any(receipt.get(k) != v for k, v in identity.items()) or sha(folder/'state.pkl') != receipt['state_sha256']:
        raise RuntimeError('Collector checkpoint identity or hash differs')
    # Only unpickle our hash-verified local commit, never an external dataset.
    state = pickle.loads((folder/'state.pkl').read_bytes())
    if len(state['workers']) != env.num_envs:
        raise RuntimeError('Incomplete worker checkpoint')
    for index, worker in enumerate(state['workers']):
        env.env_method('set_state', worker, indices=[index])
    trainer = state['trainer']
    model.num_timesteps = trainer['num_timesteps']
    model._last_obs = trainer['sb3_last_obs']
    model._last_episode_starts = trainer['sb3_last_episode_starts']
    observations, digests = observation_and_mask_digests(env)
    if not observations_equal(observations, model._last_obs) or digests != receipt['worker_digests']:
        raise RuntimeError('Restored worker observations/masks differ')
    if model.num_timesteps != receipt['num_timesteps']:
        raise RuntimeError('Restored sample counter differs')
    restore_global_rng(trainer)
    if not identical(global_rng_state(model), trainer):
        raise RuntimeError('Restored global RNG/counters differ')
    return receipt


def publish(pending, target):
    if target.exists():
        raise FileExistsError('Do not overwrite a committed segment')
    pending.rename(target)
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def staging(folder, name):
    pending = folder/('.'+name+'.pending')
    if pending.exists():
        preserved = folder/(pending.name+'.interrupted.'+str(__import__('time').time_ns()))
        pending.rename(preserved)
        operation('保留未提交采样片段，从最近完整提交恢复', preserved)
    pending.mkdir()
    return pending


def run(folder, through_chunk=4):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from training.train_vector_transaction import TransactionCompactSearchCandidateEnv
    from training.vector_env import make_vector_env, worker_seeds
    from profile_training_stages import rollout_digest
    from verify_feature_backend_benchmark import identical

    lock = (folder/'collector.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    design = json.loads((folder/'design.json').read_text())
    identity = dict(design_sha256=sha(folder/'design.json'))
    assert 1 <= through_chunk <= design['chunks'] == 4
    def unchanged():
        for name, digest in design['frozen_files'].items():
            if sha(Path(name)) != digest:
                raise RuntimeError('Frozen collector input changed: '+name)
    unchanged()
    if (folder/'collection_summary.json').exists():
        raise FileExistsError('Completed collection; preserve data instead of starting again')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    config = json.loads(Path(design['source_config']).read_text())
    assert config['device'] == 'cuda' and config['ppo']['n_envs'] == 8
    seeds = worker_seeds(design['base_seed'], 8)
    env = make_vector_env(8, design['base_seed'], config['max_pieces'],
                          env_class=TransactionCompactSearchCandidateEnv)
    model = None
    try:
        model = load_fresh_model(design['model'], env, design['base_seed'])
        before, adam = copy.deepcopy(model.policy.state_dict()), copy.deepcopy(model.policy.optimizer.state_dict())
        source_steps, updates = model.num_timesteps, model._n_updates
        assert model.n_steps == 512 and model.n_envs == 8 and model.policy.optimizer.state
        model.set_logger(configure(str(folder/'logs'), []))

        class Markers(BaseCallback):
            def __init__(self):
                super().__init__()
                self.rows = []
            def _on_step(self):
                infos = self.locals['infos']
                row = dict(step_dones=np.array(self.locals['dones'], dtype=bool, copy=True),
                    raw_env_rewards=np.array(self.locals['rewards'], copy=True),
                    step_truncated=np.array([i.get('TimeLimit.truncated', False) for i in infos]))
                for name in ('new_holes', 'holes_before', 'holes_after', 'height_after', 'pieces', 'cleared_lines', 'episode_seed'):
                    row['info_'+name] = np.array([i[name] for i in infos])
                self.rows.append(row)
                return True

        markers = Markers()
        _, callback = model._setup_learn(16384, callback=markers, reset_num_timesteps=False, progress_bar=False)
        assert env.get_attr('episode_seed') == seeds and env.get_attr('pieces') == [0]*8
        assert model._last_episode_starts.all()
        # Fresh action stream is applied AFTER SB3 model loading/setup.
        random.seed(design['action_seed'])
        np.random.seed(design['action_seed'])
        torch.manual_seed(design['action_seed'])
        torch.cuda.manual_seed_all(design['action_seed'])
        initial = folder/'initial'
        if not initial.exists():
            pending = staging(folder, 'initial')
            save_checkpoint(pending, model, env, identity)
            publish(pending, initial)
        latest, committed = initial, 0
        seen_gap = False
        for index in range(1, 5):
            segment = folder/f'chunk_{index:04d}'
            if not segment.exists():
                seen_gap = True
                continue
            if seen_gap:
                raise RuntimeError('Noncontiguous committed dataset')
            metadata = json.loads((segment/'dataset.json').read_text())
            checkpoint = json.loads((segment/'checkpoint.json').read_text())
            if (metadata['design_sha256'] != identity['design_sha256']
                    or sha(segment/'rollout.npz') != metadata['file_sha256']
                    or checkpoint['design_sha256'] != identity['design_sha256']
                    or sha(segment/'state.pkl') != checkpoint['state_sha256']):
                raise RuntimeError('Committed segment identity or data hash differs')
            latest, committed = segment, index
        restored = restore_checkpoint(latest, model, env, identity)
        assert model.num_timesteps == source_steps+committed*4096
        write_json(folder/f'restore_{committed:04d}_{os.getpid()}.json', dict(at=now(), pid=os.getpid(),
            restored_chunk=committed, sample_counter=model.num_timesteps,
            state_sha256=restored['state_sha256'], rng_observations_masks_exact=True))
        callback.on_training_start({}, {})
        for index in range(committed+1, through_chunk+1):
            unchanged()
            markers.rows.clear()
            buffer, bootstrap = model.rollout_buffer, {}
            original = buffer.compute_returns_and_advantage
            def capture(last_values, dones):
                bootstrap['last_values'] = last_values.detach().cpu().numpy().flatten().copy()
                bootstrap['last_dones'] = np.array(dones, copy=True)
                return original(last_values, dones)
            buffer.compute_returns_and_advantage = capture
            try:
                assert model.collect_rollouts(env, callback, buffer, model.n_steps, use_masking=True)
            finally:
                del buffer.compute_returns_and_advantage
            assert len(markers.rows) == 512 and model.num_timesteps == source_steps+index*4096
            assert identical(before, model.policy.state_dict()) and identical(adam, model.policy.optimizer.state_dict())
            assert model._n_updates == updates
            arrays = {'obs_'+key: value.copy() for key, value in buffer.observations.items()}
            for name in ('actions', 'action_masks', 'returns', 'advantages', 'log_probs', 'values', 'rewards', 'episode_starts'):
                arrays[name] = getattr(buffer, name).copy()
            arrays.update(bootstrap)
            arrays.update({name: np.stack([r[name] for r in markers.rows]) for name in markers.rows[0]})
            arrays['step_terminated'] = arrays['step_dones'] & ~arrays['step_truncated']
            pending = staging(folder, f'chunk_{index:04d}')
            reloaded = save_dataset(pending, arrays, dict(identity, at=now(), samples=4096, n_envs=8,
                device='cuda', gamma=model.gamma, gae_lambda=model.gae_lambda, chunk=index,
                source_steps=source_steps, collection_steps=index*4096,
                source_ppo_updates=updates, optimizer_steps=0,
                policy_adam_update_count_unchanged=True, rollout_sha256=rollout_digest(buffer),
                model_sha256=sha(Path(design['model'])), runtime=design['runtime'],
                worker_initial_seeds=seeds, fit_workers=[0, 1, 2, 3], validation_workers=[4, 5, 6, 7]))
            del arrays, reloaded
            save_checkpoint(pending, model, env, identity)
            unchanged()
            publish(pending, folder/f'chunk_{index:04d}')
            write_json(folder/'collection_progress.json', dict(at=now(), committed_chunks=index,
                collection_steps=index*4096, optimizer_steps=0))
        callback.on_training_end()
        unchanged()
        assert identical(before, model.policy.state_dict()) and identical(adam, model.policy.optimizer.state_dict())
        assert model._n_updates == updates
        if through_chunk == 4:
            summaries, episode_sets = {}, {}
            for group, workers in [('fit', [0, 1, 2, 3]), ('validation', [4, 5, 6, 7])]:
                values, returns, episodes, terminated, truncated = [], [], set(), 0, 0
                for index in range(1, 5):
                    with np.load(folder/f'chunk_{index:04d}'/'rollout.npz', allow_pickle=False) as data:
                        values.append(data['values'][:, workers].flatten())
                        returns.append(data['returns'][:, workers].flatten())
                        episodes.update(map(int, data['info_episode_seed'][:, workers].flatten()))
                        terminated += int(data['step_terminated'][:, workers].sum())
                        truncated += int(data['step_truncated'][:, workers].sum())
                episode_sets[group] = episodes
                summaries[group] = dict(residual_metrics(np.concatenate(values), np.concatenate(returns)),
                    terminated=terminated, truncated=truncated, episode_seeds=sorted(episodes), workers=workers)
            if episode_sets['fit'] & episode_sets['validation']:
                raise RuntimeError('Fit/validation episode seeds overlap')
            write_json(folder/'collection_summary.json', dict(identity, at=now(), status='passed',
                chunks=4, samples=16384, n_envs=8, device='cuda', optimizer_steps=0,
                source_steps=source_steps, final_collection_counter=model.num_timesteps,
                policy_adam_update_count_unchanged=True, groups=summaries,
                scope='Frozen original policy, fresh declared seeds; not historical source-context rollout. '
                      'Seed/worker-disjoint fit/validation, correlated within trajectories; shared batched action RNG. '
                      'Four 512-step GAE windows, bootstrap targets, no training/quality promotion. '
                      'Diagnostic worker/RNG checkpoints only; no new model checkpoint.'))
    finally:
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        env.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--through-chunk', type=int, default=4)
    parser.add_argument('--verify-restart', action='store_true')
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    if args.verify_restart:
        if (folder/'collection_summary.json').exists():
            raise FileExistsError('Completed collection; do not repeat restart test')
        children = []
        for endpoint in (2, 4):
            child = subprocess.Popen([sys.executable, __file__, '--run-dir', str(folder),
                                      '--through-chunk', str(endpoint)])
            children.append(dict(pid=child.pid, through_chunk=endpoint))
            if child.wait() != 0:
                raise RuntimeError('Segmented collector failed; preserve committed data')
        write_json(folder/'restart_receipt.json', dict(at=now(), status='passed', children=children,
            scope='Actual separate processes; second restores committed chunk2 RNG/worker observations/masks. '
                  'No continuous-vs-resumed duplicate collection comparison.'))
    else:
        run(folder, args.through_chunk)
