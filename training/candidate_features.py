"""Public, deterministic one-placement features for the fixed 1840 actions."""

import numpy as np

from training.tetris_core import ACTION_COUNT
from training.tetris_core.legal_placements import get_legal_placements

FEATURE_VERSION = "candidate-placement-uint8-v1"
FEATURE_NAMES = (
    "piece_type", "hold", "rotation", "x", "y_plus_3", "cleared_lines",
    "holes_after", "aggregate_height_after", "max_height_after", "bumpiness_after",
    "wells_after", "new_holes", "removed_holes", "max_height_before",
    "top_four_occupied_after", "lock_out",
)
FEATURE_SCALE = np.asarray((6, 1, 3, 9, 22, 4, 190, 200, 20, 180,
                            200, 190, 190, 20, 40, 1), dtype=np.float32)
PIECES = "IJLOSTZ"


def board_features(board):
    heights = []
    holes = 0
    for x in range(10):
        first = next((y for y in range(20) if board[y][x] is not None), 20)
        heights.append(20 - first)
        if first < 20:
            holes += sum(board[y][x] is None for y in range(first + 1, 20))
    bumpiness = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    wells = sum(max(0, min(20 if x == 0 else heights[x - 1],
                            20 if x == 9 else heights[x + 1]) - height)
                for x, height in enumerate(heights))
    return holes, sum(heights), max(heights), bumpiness, wells


def candidate_board(board, occupied_cells):
    """Merge visible cells and clear lines, without spawning a hidden future piece."""
    result = [row[:] for row in board]
    visible = False
    for cell in occupied_cells:
        y = cell["y"]
        if y >= 0:
            result[y][cell["x"]] = "X"
            visible = True
    if not visible:
        return result, 0, True
    kept = [row for row in result if not all(row)]
    cleared = 20 - len(kept)
    return [[None] * 10 for _ in range(cleared)] + kept, cleared, False


def candidate_features(public, placements=None):
    """Return action-aligned features and mask using only the public observation."""
    placements = get_legal_placements(public) if placements is None else placements
    matrix = np.zeros((ACTION_COUNT, len(FEATURE_NAMES)), dtype=np.uint8)
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    board = public["board"]
    before_holes, _, before_height, _, _ = board_features(board)
    for placement in placements:
        action = placement["actionId"]
        if mask[action]:
            raise AssertionError("Duplicate legal action ID")
        mask[action] = True
        hold = placement["hold"]
        piece = (public["hold"] or public["next"][0]) if hold else public["currentPiece"]["type"]
        after, cleared, lock_out = candidate_board(board, placement["occupiedCells"])
        holes, aggregate, height, bumpiness, wells = board_features(after)
        values = (PIECES.index(piece), hold, placement["rotation"], placement["x"],
                  placement["y"] + 3, cleared, holes, aggregate, height, bumpiness,
                  wells, max(0, holes - before_holes), max(0, before_holes - holes),
                  before_height, sum(cell is not None for row in after[:4] for cell in row),
                  int(lock_out))
        if any(value < 0 or value > scale or value > 255
               for value, scale in zip(values, FEATURE_SCALE)):
            raise OverflowError(f"Candidate feature exceeds uint8 range: {values}")
        matrix[action] = values
    return matrix, mask
