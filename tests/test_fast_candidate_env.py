"""Fast evaluation must produce the exact original Candidate trajectory."""

import numpy as np
import pytest

from training.env.candidate_env import CandidateTetrisEnv
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv
from training.candidate_features import candidate_features
from training.evaluation.fast_candidate_features import candidate_features as fast_candidate_features
from training.tetris_core import TetrisCore
from training.tetris_core.pieces import CELLS, MATRICES, ROTATIONS, make_piece
from training.evaluation.fast_legal import get_legal_placements as fast_legal_placements


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


@pytest.mark.parametrize("seed", [42, 1234000, 20261013])
def test_long_candidate_bitboard_bfs_matches_official(seed):
    official = FastCandidateTetrisEnv(max_pieces=100)
    accelerated = LongCandidateTetrisEnv(max_pieces=100)
    try:
        left, _ = official.reset(seed=seed)
        right, _ = accelerated.reset(seed=seed)
        for step in range(100):
            np.testing.assert_array_equal(left["state"], right["state"])
            np.testing.assert_array_equal(left["candidate"], right["candidate"])
            mask = official.action_masks()
            np.testing.assert_array_equal(mask, accelerated.action_masks())
            assert official.core.get_legal_placements() == accelerated._placement_cache
            legal = np.flatnonzero(mask)
            action = int(legal[(step * 17 + seed) % len(legal)])
            left, _, done_a, capped_a, info_a = official.step(action)
            right, _, done_b, capped_b, info_b = accelerated.step(action)
            assert (done_a, capped_a, info_a) == (done_b, capped_b, info_b)
            if done_a or capped_a:
                break
    finally:
        official.close()
        accelerated.close()


@pytest.mark.parametrize("fixture", ["single_clear", "lock_out", "partial_lock"])
def test_fast_candidate_features_match_scoring_and_topout_edges(fixture):
    core = TetrisCore(42)
    if fixture == "single_clear":
        core.current = make_piece("O")
        for x in range(10):
            if x not in (3, 4):
                core.board[19][x] = "Z"
    elif fixture == "lock_out":
        core.current = make_piece("I")
        core.current["y"] = -2
        for x in range(3, 7):
            core.board[0][x] = "Z"
    else:
        core.current = make_piece("O")
        core.current["x"], core.current["y"] = 0, -1
        core.board[1][0] = "Z"
    public = core.get_public_observation()
    placements = core.get_legal_placements()
    for original, accelerated in zip(candidate_features(public, placements),
                                     fast_candidate_features(public, placements)):
        np.testing.assert_array_equal(original, accelerated)


@pytest.mark.parametrize("line_count", [2, 3, 4])
def test_fast_candidate_features_match_multiline_clears(line_count):
    core = TetrisCore(42)
    core.current = make_piece("I")
    for y in range(20 - line_count, 20):
        for x in range(10):
            if x != 4:
                core.board[y][x] = "Z"
    public = core.get_public_observation()
    placements = core.get_legal_placements()
    original, original_mask = candidate_features(public, placements)
    accelerated, accelerated_mask = fast_candidate_features(public, placements)
    np.testing.assert_array_equal(original_mask, accelerated_mask)
    np.testing.assert_array_equal(original, accelerated)
    assert np.any(original[:, 5] == line_count)


@pytest.mark.parametrize("kind", "IJLOSTZ")
@pytest.mark.parametrize("board_case", range(4))
@pytest.mark.parametrize("hold_case", range(3))
def test_fast_legal_matches_core_on_fixed_edge_boards(kind, board_case, hold_case):
    core = TetrisCore(42)
    core.current = make_piece(kind)
    core.hold = None if hold_case == 2 else "T"
    core.hold_used = hold_case == 0
    if board_case == 1:
        for y in range(15, 20):
            for x in range(10):
                if x not in ((y + 1) % 10, (y + 5) % 10):
                    core.board[y][x] = "Z"
    elif board_case == 2:
        for y in range(8, 20):
            for x in range(10):
                if x != 4 and (y + 2 * x) % 7 != 0:
                    core.board[y][x] = "Z"
    elif board_case == 3:
        for y in (2, 4, 7, 11, 16, 19):
            for x in range(10):
                if (3 * x + y) % 5 < 2:
                    core.board[y][x] = "Z"
    public = core.get_public_observation()
    expected = core.get_legal_placements()
    if board_case == 0:
        assert expected
        assert any(placement["hold"] for placement in expected) == (hold_case != 0)
    assert fast_legal_placements(public) == expected
    reference, reference_mask = candidate_features(public, expected)
    accelerated, accelerated_mask = fast_candidate_features(public, expected)
    np.testing.assert_array_equal(accelerated_mask, reference_mask)
    np.testing.assert_array_equal(accelerated, reference)


@pytest.mark.parametrize("kind", "IJLOSTZ")
@pytest.mark.parametrize("hold_case", range(3))
def test_fast_legal_matches_core_when_spawn_is_blocked(kind, hold_case):
    core = TetrisCore(42)
    core.current = make_piece(kind)
    core.hold = None if hold_case == 2 else "T"
    core.hold_used = hold_case == 0
    for y in (0, 1):
        core.board[y] = ["Z"] * 10
    public = core.get_public_observation()
    expected = core.get_legal_placements()
    assert expected == []
    assert fast_legal_placements(public) == expected
    reference, reference_mask = candidate_features(public, expected)
    accelerated, accelerated_mask = fast_candidate_features(public, expected)
    np.testing.assert_array_equal(accelerated_mask, reference_mask)
    np.testing.assert_array_equal(accelerated, reference)


@pytest.mark.parametrize("kind", "IJLOSTZ")
@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("board_case", range(2))
def test_fast_legal_matches_core_from_rotated_wall_states(kind, rotation, board_case):
    core = TetrisCore(42)
    if board_case:
        for y in range(11, 20):
            for x in range(10):
                if (x + 2 * y) % 5 < 2:
                    core.board[y][x] = "Z"
    core.current = make_piece(kind)
    core.current["rotation"] = ROTATIONS[rotation]
    core.current["matrix"] = [list(row) for row in MATRICES[kind, rotation]]
    core.current["y"] = 5
    core.hold_used = True
    xs = [x for x, _ in CELLS[kind, rotation]]
    left, right = -min(xs), 9 - max(xs)
    for x in (left, (left + right) // 2, right):
        core.current["x"] = x
        public = core.get_public_observation()
        assert fast_legal_placements(public) == core.get_legal_placements()
