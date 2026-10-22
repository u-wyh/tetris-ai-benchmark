"""Optional training reward does not change formal Tetris transitions."""

import numpy as np
import pytest

from training.env.candidate_env import CandidateTetrisEnv
from training.env.tetris_env import BALANCED_REWARD_VERSION, TetrisEnv, placement_reward
from training.train_vector_transaction import vector_config


def test_balanced_reward_is_less_four_clear_dominant_and_penalizes_death():
    assert placement_reward(4, False, "survival_balanced_v1") == pytest.approx(5.201)
    assert placement_reward(4, False) == pytest.approx(8.001)
    assert placement_reward(0, True, "survival_balanced_v1") == pytest.approx(-9.999)
    assert placement_reward(0, True) == pytest.approx(-1.999)
    config = vector_config(device="cpu", reward_profile="survival_balanced_v1")
    assert config["reward_version"] == BALANCED_REWARD_VERSION
    assert config["reward"]["line_clear"] == [0.0, 1.0, 2.4, 3.8, 5.2]
    assert config["reward"]["game_over"] == -10.0


def test_balanced_reward_keeps_actions_and_core_identical():
    regular = CandidateTetrisEnv(max_pieces=100)
    balanced = CandidateTetrisEnv(max_pieces=100, reward_profile="survival_balanced_v1")
    try:
        regular.reset(seed=42)
        balanced.reset(seed=42)
        for _ in range(40):
            mask = regular.action_masks()
            assert np.array_equal(mask, balanced.action_masks())
            action = int(np.flatnonzero(mask)[0])
            _, _, term_a, trunc_a, info_a = regular.step(action)
            _, _, term_b, trunc_b, info_b = balanced.step(action)
            assert (term_a, trunc_a) == (term_b, trunc_b)
            assert regular.core.board == balanced.core.board
            assert (info_a["score"], info_a["lines"], info_a["pieces"]) == (
                info_b["score"], info_b["lines"], info_b["pieces"])
            if term_a or trunc_a:
                break
        with pytest.raises(ValueError, match="reward configuration"):
            regular.set_state(balanced.get_state())
    finally:
        regular.close()
        balanced.close()


def test_balanced_reward_rejects_combined_legacy_penalty():
    with pytest.raises(ValueError, match="cannot combine"):
        TetrisEnv(reward_profile="survival_balanced_v1", height_penalty_coef=0.1)
