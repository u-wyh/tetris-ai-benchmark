"""Fresh seeds survive real SB3 load; collector restores trajectory and RNG."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_independent_critic_data import load_fresh_model, save_checkpoint, restore_checkpoint
from stable_baselines3.common.vec_env import DummyVecEnv
from training.train_vector_transaction import TransactionCompactSearchCandidateEnv


def environments():
    return DummyVecEnv([lambda: TransactionCompactSearchCandidateEnv(max_pieces=10)]*2)


def test_real_sb3_load_does_not_override_fresh_worker_seeds(tmp_path):
    from sb3_contrib import MaskablePPO
    from training.candidate_policy import CandidateScoringPolicy
    torch.set_num_threads(1)
    env = environments()
    try:
        source = MaskablePPO(CandidateScoringPolicy, env, n_steps=2, batch_size=4, seed=42, device='cpu')
        source.save(tmp_path/'source.zip')
        fresh = load_fresh_model(tmp_path/'source.zip', env, 2600000, device='cpu')
        fresh._setup_learn(4, reset_num_timesteps=False)
        assert env.get_attr('episode_seed') == [2600000, 2600001]
        assert env.get_attr('pieces') == [0, 0]
        assert fresh._last_episode_starts.all()
    finally:
        env.close()


def test_checkpoint_restores_workers_observations_and_rng_and_rejects_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    env = environments()
    try:
        env.seed(2600000)
        model = SimpleNamespace(num_timesteps=123, seed=42, _last_obs=env.reset(),
                                _last_episode_starts=np.ones(2, dtype=bool))
        identity = dict(design_sha256='test-design')
        save_checkpoint(tmp_path, model, env, identity)
        expected = torch.rand(4)
        actions = np.array([np.flatnonzero(mask)[0] for mask in env.env_method('action_masks')])
        next_obs, _, _, _ = env.step(actions)
        model._last_obs = next_obs
        model.num_timesteps = 999
        restore_checkpoint(tmp_path, model, env, identity)
        assert model.num_timesteps == 123
        assert env.get_attr('pieces') == [0, 0]
        assert torch.equal(torch.rand(4), expected)
        replay, _, _, _ = env.step(actions)
        for key in replay:
            np.testing.assert_array_equal(replay[key], next_obs[key])
        with pytest.raises(RuntimeError, match='identity or hash'):
            restore_checkpoint(tmp_path, model, env, dict(design_sha256='changed'))
        (tmp_path/'state.pkl').write_bytes(b'damaged')
        with pytest.raises(RuntimeError, match='identity or hash'):
            restore_checkpoint(tmp_path, model, env, identity)
    finally:
        env.close()
