"""PPO-Shaped V2 changes only the new-hole reward coefficient."""

import json

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from training.env import TetrisEnv
from training.env.tetris_env import shape_reward
from training.step0_baseline import (compare_raw_initial_parameters,
                                     compare_v1_initial_parameters)
from training.train_ppo import ROOT
from training.train_vector_transaction import vector_config


def test_v2_preset_differs_from_v1_only_by_coefficient():
    v1 = json.loads((ROOT / "configs/ppo_hole_v1_2m_seed42.json").read_text())
    v2 = json.loads((ROOT / "configs/ppo_hole_v2_2m_seed42.json").read_text())
    assert v2 == dict(v1, hole_penalty_coef=0.02)
    config = vector_config(target_steps=v2["target_total_steps"],
                           device=v2["device"], n_envs=v2["n_envs"],
                           base_seed=v2["base_seed"], hole_penalty_coef=0.02)
    assert config["actual_target_committed_steps"] == 2_002_944
    assert config["ppo"]["n_steps"] == 512
    assert config["reward"]["new_holes_penalty_coef"] == 0.02


@pytest.mark.parametrize("before,after,penalty", [(0, 0, 0), (2, 5, 0.06), (5, 2, 0)])
def test_v2_reward_and_raw_zero_coefficient(before, after, penalty):
    shaped, new_holes, actual_penalty = shape_reward(1.001, before, after, 0.02)
    raw, _, raw_penalty = shape_reward(1.001, before, after, 0)
    assert new_holes == max(0, after - before)
    assert actual_penalty == pytest.approx(penalty)
    assert shaped == pytest.approx(raw - penalty)
    assert raw == pytest.approx(1.001)
    assert raw_penalty == 0


def test_v2_environment_preserves_observation_mask_and_checkpoint_coefficient():
    raw, v2 = TetrisEnv(), TetrisEnv(hole_penalty_coef=0.02)
    try:
        raw_obs, _ = raw.reset(seed=42)
        v2_obs, _ = v2.reset(seed=42)
        assert np.array_equal(raw_obs, v2_obs)
        for _ in range(4):
            raw_mask, v2_mask = raw.action_masks(), v2.action_masks()
            assert np.array_equal(raw_mask, v2_mask)
            action = int(np.flatnonzero(raw_mask)[0])
            a, b = raw.step(action), v2.step(action)
            assert np.array_equal(a[0], b[0])
            assert a[2:4] == b[2:4]
            assert b[1] == pytest.approx(a[1] - 0.02 * b[4]["new_holes"])
            assert raw.core.board == v2.core.board
            assert raw.core.score == v2.core.score
            if a[2] or a[3]:
                break
        state = v2.get_state()
        assert state["hole_penalty_coef"] == 0.02
        restored = TetrisEnv(hole_penalty_coef=0.02)
        restored.set_state(state)
        assert np.array_equal(restored.action_masks(), v2.action_masks())
        with pytest.raises(ValueError, match="reward configuration"):
            TetrisEnv(hole_penalty_coef=0.1).set_state(state)
    finally:
        raw.close()
        v2.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Formal Step-0 uses CUDA initialization")
def test_v2_fresh_cuda_step0_matches_raw_and_v1_exactly():
    torch.set_num_threads(1)
    config = vector_config(target_steps=2_000_000, hole_penalty_coef=0.02)
    ppo = config["ppo"]
    env = DummyVecEnv([lambda: TetrisEnv(hole_penalty_coef=0.02) for _ in range(8)])
    try:
        model = MaskablePPO(
            "MlpPolicy", env,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]},
                           "activation_fn": torch.nn.Tanh},
            learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
            n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"], ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"],
            seed=42, device="cuda", verbose=0)
        assert model.num_timesteps == 0
        assert compare_raw_initial_parameters(model, config)
        assert compare_v1_initial_parameters(model, config)
    finally:
        env.close()
