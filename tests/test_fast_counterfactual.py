import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.fast_counterfactual import fast_branch
from training.evaluation.hybrid_queue import PPO
from training.evaluation.ppo_fresh_counterfactual import branch


@pytest.fixture(scope="module")
def frozen_ppo():
    if not PPO.is_file():
        pytest.skip("Frozen PPO checkpoint is not installed")
    torch.set_num_threads(1)
    return MaskablePPO.load(PPO, device="cpu")


def test_fast_branch_matches_legacy_score_and_censoring(frozen_ppo):
    env = FastCandidateTetrisEnv(max_pieces=500)
    try:
        env.reset(seed=41)
        snapshot = env.get_state()
        action = int(np.flatnonzero(env.action_masks())[0])
    finally:
        env.close()
    models = {"ppo": frozen_ppo}
    assert fast_branch(snapshot, action, models, 300) == branch(snapshot, action, models, 300)


def test_fast_branch_keeps_public_trajectory_and_actions(frozen_ppo):
    slow = FastCandidateTetrisEnv(max_pieces=500)
    fast = LongCandidateTetrisEnv(max_pieces=500)
    try:
        slow.reset(seed=42)
        fast.reset(seed=42)
        for _ in range(200):
            old_obs, new_obs = slow.get_observation(), fast.get_observation()
            old_mask, new_mask = slow.action_masks(), fast.action_masks()
            assert np.array_equal(old_mask, new_mask)
            assert np.array_equal(old_obs["state"], new_obs["state"])
            assert np.array_equal(old_obs["candidate"], new_obs["candidate"])
            old_action = int(frozen_ppo.predict(old_obs, deterministic=True, action_masks=old_mask)[0])
            new_action = fast_action(frozen_ppo, new_obs, new_mask)
            assert old_action == new_action
            _, _, old_dead, old_capped, old_info = slow.step(old_action)
            _, _, new_dead, new_capped, new_info = fast.step(new_action)
            assert old_info["score"] == new_info["score"]
            assert old_info["cleared_lines"] == new_info["cleared_lines"]
            assert slow.core.get_public_observation() == fast.core.get_public_observation()
            assert (old_dead, old_capped) == (new_dead, new_capped)
            if old_dead or old_capped:
                break
    finally:
        slow.close()
        fast.close()
