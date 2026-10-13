"""Fast evaluation must produce the exact original Candidate trajectory."""

import numpy as np
import pytest

from training.env.candidate_env import CandidateTetrisEnv
from training.env.fast_candidate_env import FastCandidateTetrisEnv


@pytest.mark.parametrize("seed", [42, 1234000, 20261013])
def test_fast_candidate_matches_original(seed):
    slow = CandidateTetrisEnv(max_pieces=200)
    fast = FastCandidateTetrisEnv(max_pieces=200)
    try:
        slow_obs, slow_info = slow.reset(seed=seed)
        fast_obs, fast_info = fast.reset(seed=seed)
        assert slow_info == fast_info
        for step in range(200):
            np.testing.assert_array_equal(slow_obs["state"], fast_obs["state"])
            np.testing.assert_array_equal(slow_obs["candidate"], fast_obs["candidate"])
            mask = slow.action_masks()
            np.testing.assert_array_equal(mask, fast.action_masks())
            legal = np.flatnonzero(mask)
            action = int(legal[(step * 17 + seed) % len(legal)])
            slow_obs, slow_reward, slow_done, slow_truncated, slow_info = slow.step(action)
            fast_obs, fast_reward, fast_done, fast_truncated, fast_info = fast.step(action)
            assert (slow_reward, slow_done, slow_truncated, slow_info) == (
                fast_reward, fast_done, fast_truncated, fast_info)
            assert "get_legal_placements" not in fast.core.__dict__
            assert slow.core.get_public_observation() == fast.core.get_public_observation()
            if slow_done or slow_truncated:
                break
        else:
            pytest.fail("Expected the fixed cap to finish")
    finally:
        slow.close()
        fast.close()


def test_fast_candidate_snapshot_restores_original_state():
    fast = FastCandidateTetrisEnv(max_pieces=100)
    restored = CandidateTetrisEnv(max_pieces=100)
    try:
        fast.reset(seed=101)
        restored.reset(seed=101)
        action = int(np.flatnonzero(fast.action_masks())[0])
        fast.step(action)
        restored.set_state(fast.get_state())
        np.testing.assert_array_equal(fast.action_masks(), restored.action_masks())
        np.testing.assert_array_equal(fast.get_observation()["candidate"],
                                      restored.get_observation()["candidate"])
        assert "get_legal_placements" not in restored.core.__dict__
    finally:
        fast.close()
        restored.close()
