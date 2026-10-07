"""Process isolation, worker seeding and MaskablePPO mask integration."""

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from training.env import TetrisEnv
from training.vector_env import make_vector_env, worker_seeds


class BoardProbeEnv(TetrisEnv):
    def fill_bottom(self):
        self.core.board[19] = ["I"] * 10


def test_worker_seeds():
    assert worker_seeds(42, 4) == [42, 43, 44, 45]
    with pytest.raises(ValueError):
        worker_seeds((1 << 32) - 1, 2)


def test_single_worker_matches_direct_environment():
    direct = TetrisEnv()
    vector = make_vector_env(1, base_seed=42)
    try:
        direct_obs, _ = direct.reset(seed=42)
        vector_obs = vector.reset()
        assert np.array_equal(direct_obs, vector_obs[0])
        for _ in range(5):
            mask = direct.action_masks()
            assert np.array_equal(mask, get_action_masks(vector)[0])
            action = int(np.argmax(mask))
            direct_obs, direct_reward, terminated, truncated, direct_info = direct.step(action)
            vector_obs, rewards, dones, infos = vector.step(np.array([action]))
            assert np.array_equal(direct_obs, vector_obs[0])
            assert direct_reward == rewards[0]
            assert dones[0] == (terminated or truncated)
            assert direct_info["score"] == infos[0]["score"]
            assert direct_info["lines"] == infos[0]["lines"]
    finally:
        direct.close()
        vector.close()


def test_masks_and_seeded_workers_are_independent():
    env = make_vector_env(2, base_seed=42, env_class=BoardProbeEnv)
    try:
        assert len({process.pid for process in env.processes}) == 2
        obs = env.reset()
        assert obs.shape == (2, 237)
        assert [info["episode_seed"] for info in env.reset_infos] == [42, 43]
        masks = get_action_masks(env)
        assert masks.shape == (2, 1840) and masks.dtype == np.bool_
        env.env_method("fill_bottom", indices=[0])
        changed = get_action_masks(env)
        assert not np.array_equal(masks[0], changed[0])
        assert np.array_equal(masks[1], changed[1])
    finally:
        env.close()


def test_auto_reset_truncation_and_unaffected_worker():
    env = make_vector_env(2, base_seed=42, max_pieces=1)
    try:
        env.reset()
        masks = get_action_masks(env)
        actions = np.argmax(masks, axis=1)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, 237)
        assert rewards.shape == dones.shape == (2,)
        assert dones.tolist() == [True, True]
        assert all(info["TimeLimit.truncated"] for info in infos)
        assert all(info["terminal_observation"].shape == (237,) for info in infos)
        assert env.reset_infos[0]["episode_seed"] != env.reset_infos[1]["episode_seed"]
        assert get_action_masks(env).shape == (2, 1840)
    finally:
        env.close()


def test_game_over_in_one_worker_does_not_end_the_other():
    env = make_vector_env(2, base_seed=42)
    try:
        env.reset()
        for _ in range(200):
            masks = get_action_masks(env)
            _, _, dones, infos = env.step(np.argmax(masks, axis=1))
            if any(done and info["game_over"] for done, info in zip(dones, infos)) and not all(dones):
                assert all(not info["TimeLimit.truncated"] for info in infos)
                assert all(info["terminal_observation"].shape == (237,)
                           for done, info in zip(dones, infos) if done)
                assert get_action_masks(env).shape == (2, 1840)
                break
        else:
            pytest.fail("Expected worker episodes to end independently")
    finally:
        env.close()


def test_maskable_ppo_smoke_with_subproc_masks():
    torch.set_num_threads(1)
    env = make_vector_env(2, base_seed=42)
    try:
        model = MaskablePPO(
            "MlpPolicy", env, policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]},
                                      "activation_fn": torch.nn.Tanh},
            n_steps=128, batch_size=256, n_epochs=1, seed=42, device="cpu", verbose=0)
        model.learn(total_timesteps=256, use_masking=True)
        assert model.num_timesteps == 256
        assert get_action_masks(env).shape == (2, 1840)
    finally:
        env.close()
