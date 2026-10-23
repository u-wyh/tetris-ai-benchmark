"""Evaluation-only candidate features computed from public board bitmasks."""

import numpy as np

from training.candidate_features import FEATURE_NAMES, FEATURE_SCALE, PIECES
from training.tetris_core import ACTION_COUNT


def candidate_features(public, placements):
    board = public["board"]
    rows = []
    columns = [0] * 10
    for y, board_row in enumerate(board):
        bits = 0
        column_bit = 1 << (19 - y)
        for x, cell in enumerate(board_row):
            if cell is not None:
                bits |= 1 << x
                columns[x] |= column_bit
        rows.append(bits)
    before_heights = [col.bit_length() for col in columns]
    before_holes = sum(height - col.bit_count()
                       for height, col in zip(before_heights, columns))
    before_height = max(before_heights)
    matrix = np.zeros((ACTION_COUNT, len(FEATURE_NAMES)), dtype=np.uint8)
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    current_kind = public["currentPiece"]["type"] if placements else None
    held_kind = (public["hold"] or public["next"][0]) if placements else None
    for placement in placements:
        action = placement["actionId"]
        if mask[action]:
            raise AssertionError("Duplicate legal action ID")
        mask[action] = True
        placed_columns = columns.copy()
        placed_rows = rows.copy()
        visible = False
        for cell in placement["occupiedCells"]:
            x, y = cell["x"], cell["y"]
            if y >= 0:
                placed_columns[x] |= 1 << (19 - y)
                placed_rows[y] |= 1 << x
                visible = True
        cleared = 0
        if visible:
            kept = [row for row in placed_rows if row != 1023]
            cleared = 20 - len(kept)
            if cleared:
                placed_columns = [0] * 10
                for y, row in enumerate(([0] * cleared) + kept):
                    while row:
                        bit = row & -row
                        placed_columns[bit.bit_length() - 1] |= 1 << (19 - y)
                        row ^= bit
        heights = [col.bit_length() for col in placed_columns]
        holes = sum(height - col.bit_count()
                    for height, col in zip(heights, placed_columns))
        aggregate = sum(heights)
        height = max(heights)
        bumpiness = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
        wells = max(0, heights[1] - heights[0])
        for x in range(1, 9):
            depth = min(heights[x - 1], heights[x + 1]) - heights[x]
            if depth > 0:
                wells += depth
        wells += max(0, heights[8] - heights[9])
        top_four = sum((col >> 16).bit_count() for col in placed_columns)
        values = (PIECES.index(held_kind if placement["hold"] else current_kind),
                  placement["hold"], placement["rotation"], placement["x"],
                  placement["y"] + 3, cleared, holes, aggregate, height, bumpiness,
                  wells, max(0, holes - before_holes), max(0, before_holes - holes),
                  before_height, top_four, int(not visible))
        if any(value < 0 or value > scale or value > 255
               for value, scale in zip(values, FEATURE_SCALE)):
            raise OverflowError(f"Candidate feature exceeds uint8 range: {values}")
        matrix[action] = values
    return matrix, mask
