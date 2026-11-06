"""Reuse archived placement edge coverage before adopting search in training.

Ported from codex/ppo-branch-throughput at5e7483b; production BFS unchanged.
"""
import numpy as np
import pytest
from training.candidate_features import candidate_features
from training.evaluation.fast_candidate_features import candidate_features as fast_candidate_features
from training.evaluation.fast_legal import get_legal_placements as reference_placements
from training.compact_legal import get_legal_placements as compact_placements


def fast_legal_placements(public):
    original = reference_placements(public)
    compact = compact_placements(public)
    assert compact == original
    return compact

from training.tetris_core import TetrisCore
from training.tetris_core.pieces import CELLS, MATRICES, ROTATIONS, make_piece

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


def test_fast_evaluation_matches_reachable_core_states():
    verified = 0
    actions = 0
    for seed in range(6100, 6112):
        core = TetrisCore(seed)
        for step in range(40):
            public = core.get_public_observation()
            placements = core.get_legal_placements()
            assert fast_legal_placements(public) == placements
            reference, reference_mask = candidate_features(public, placements)
            accelerated, accelerated_mask = fast_candidate_features(public, placements)
            np.testing.assert_array_equal(accelerated_mask, reference_mask)
            np.testing.assert_array_equal(accelerated, reference)
            verified += 1
            actions += len(placements)
            if not placements:
                break
            action = placements[(seed + 17 * step) % len(placements)]["actionId"]
            result = core.step(action)
            if result["gameOver"]:
                break
    assert verified >= 100
    assert actions >= 1000
    print(f"reachable states={verified}, legal actions={actions}")



@pytest.mark.parametrize("kind", "IJLOSTZ")
@pytest.mark.parametrize("rotation", range(4))
def test_compact_visited_negative_y_boundaries(kind, rotation):
    core = TetrisCore(42)
    core.current = make_piece(kind)
    core.current["rotation"] = ROTATIONS[rotation]
    core.current["matrix"] = [list(row) for row in MATRICES[kind, rotation]]
    core.hold_used = True
    for y in (-20, -12, -10, -9, -8, -7, -6, -5, -3, -1, 0, 17, 19, 20):
        for x in (-4, -3, -2, 0, 3, 7, 9, 10):
            core.current.update(x=x, y=y)
            public = core.get_public_observation()
            assert compact_placements(public) == reference_placements(public)


@pytest.mark.parametrize("kind", "IJLOSTZ")
def test_row_collision_bits_match_cell_geometry(kind):
    from training.compact_legal import free_rows
    from training.tetris_core.pieces import collides
    for pattern in range(4):
        core = TetrisCore(42)
        if pattern:
            for y in range(20):
                for x in range(10):
                    if ((x * 3 + y * 7) % (pattern * 3 + 2)) < pattern:
                        core.board[y][x] = "Z"
        rows = tuple(sum(1 << x for x, cell in enumerate(row) if cell)
                     for row in core.board)
        free = free_rows(kind, rows)
        for rotation in range(4):
            for y in range(-9, 21):
                for x in range(-3, 10):
                    assert bool(free[rotation][y + 9] & (1 << (x + 3))) == (
                        not collides(core.board, kind, rotation, x, y))
