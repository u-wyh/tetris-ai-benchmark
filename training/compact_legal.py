"""Opt-in training BFS using row bitmaps and identical observable paths."""

from training.tetris_core.actions import encode_action
from training.tetris_core.pieces import CELLS, I_KICKS, JLSTZ_KICKS, ROTATIONS, make_piece

from training.evaluation.fast_legal import BOUNDS, _branch as reference_branch


ORIGIN_BITS = {}
for _kind in "IJLOSTZ":
    ORIGIN_BITS[_kind] = tuple(
        ((1 << (10 - BOUNDS[_kind, r][1] + BOUNDS[_kind, r][0])) - 1)
        << (3 - BOUNDS[_kind, r][0]) for r in range(4))


def free_rows(kind, board_rows):
    """Bit x+3 is set iff the piece at that origin does not collide.

    Rows index y+9. The final row is a y=20 sentinel for Down checks.
    This collision table does not apply the separate BFS top>=-6 bound.
    """
    result = []
    for rotation in range(4):
        cells = CELLS[kind, rotation]
        allowed = ORIGIN_BITS[kind][rotation]
        high_y = BOUNDS[kind, rotation][3]
        rows = []
        for y in range(-9, 21):
            blocked = 0
            if y + high_y >= 20:
                rows.append(0)
                continue
            for dx, dy in cells:
                if y + dy >= 0:
                    blocked |= board_rows[y + dy] << (3 - dx)
            rows.append(allowed & ~blocked)
        result.append(rows)
    return result


def _branch(observation, hold, board_rows):
    piece = (make_piece(observation["hold"] or observation["next"][0]) if hold
             else observation["currentPiece"])
    if piece is None:
        return []
    kind = piece["type"]
    shapes = [CELLS[kind, r] for r in range(4)]
    bounds = [BOUNDS[kind, r] for r in range(4)]
    rotation = ROTATIONS.index(piece["rotation"])
    # Keep the exact reference behavior for unusual initial poses.
    if not (-3 <= piece["x"] <= 9 and -9 <= piece["y"] <= 19):
        return reference_branch(observation, hold, board_rows)
    free = free_rows(kind, board_rows)

    def collides(rotation, x, y):
        if x < -3 or x > 9 or y > 19:
            return True
        bit = 1 << (x + 3)
        # A kick can first succeed above the table, then fail the BFS top
        # bound. Preserve that first-success choice instead of trying a later kick.
        if y < -9:
            return not (ORIGIN_BITS[kind][rotation] & bit)
        return not (free[rotation][y + 9] & bit)

    def rotate(rotation, x, y):
        target = (rotation + 1) % 4
        if kind == "O":
            return x, y, target
        kicks = I_KICKS if kind == "I" else JLSTZ_KICKS
        for dx, dy_up in kicks[rotation]:
            nx, ny = x + dx, y - dy_up
            if not collides(target, nx, ny):
                return nx, ny, target
        return None

    if collides(rotation, piece["x"], piece["y"]):
        return []
    nodes = [(piece["x"], piece["y"], rotation, -1, None)]
    visited = [[0] * 30 for _ in range(4)]
    visited[rotation][piece["y"] + 9] = 1 << (piece["x"] + 3)
    by_cells = {}
    head = 0
    while head < len(nodes):
        x, y, rotation, _, _ = nodes[head]
        row = y + 9
        bit = 1 << (x + 3)
        if not (free[rotation][row + 1] & bit):
            min_x, _, min_y, _ = bounds[rotation]
            final_x, final_y = x + min_x, y + min_y
            if 0 <= final_x < 10 and -3 <= final_y <= 19:
                cells = sorted(((x + dx, y + dy) for dx, dy in shapes[rotation]),
                               key=lambda cell: (cell[1], cell[0]))
                key = tuple(cy * 10 + cx for cx, cy in cells)
                action_id = encode_action(hold, rotation, final_x, final_y)
                previous = by_cells.get(key)
                if previous is None or action_id < previous["actionId"]:
                    steps = []
                    parent = head
                    while nodes[parent][3] != -1:
                        steps.append(nodes[parent][4])
                        parent = nodes[parent][3]
                    steps.reverse()
                    if hold:
                        steps.insert(0, "Hold")
                    steps.append("HardDrop")
                    by_cells[key] = {
                        "actionId": action_id, "hold": hold, "rotation": rotation,
                        "x": final_x, "y": final_y,
                        "occupiedCells": [{"x": cx, "y": cy} for cx, cy in cells],
                        "path": steps,
                    }
        seen = visited[rotation]
        available = free[rotation][row] & ~seen[row]
        top = y + bounds[rotation][2]
        # Neighbor order determines the exact public shortest path.
        if -6 <= top <= 19:
            left, right = bit >> 1, bit << 1
            if available & left:
                seen[row] |= left
                nodes.append((x - 1, y, rotation, head, "Left"))
            if available & right:
                seen[row] |= right
                nodes.append((x + 1, y, rotation, head, "Right"))
        rotated = rotate(rotation, x, y)
        if rotated is not None:
            nx, ny, nr = rotated
            top = ny + bounds[nr][2]
            if -6 <= top <= 19:
                target_row, target_bit = ny + 9, 1 << (nx + 3)
                if not (visited[nr][target_row] & target_bit):
                    visited[nr][target_row] |= target_bit
                    nodes.append((nx, ny, nr, head, "RotateCW"))
        if (-6 <= y + 1 + bounds[rotation][2] <= 19
                and free[rotation][row + 1] & bit and not (seen[row + 1] & bit)):
            seen[row + 1] |= bit
            nodes.append((x, y + 1, rotation, head, "Down"))
        head += 1
    return sorted(by_cells.values(), key=lambda placement: placement["actionId"])


def get_legal_placements(observation):
    if observation["currentPiece"] is None or observation["phase"] != "playing":
        return []
    board_rows = tuple(sum(1 << x for x, cell in enumerate(row) if cell is not None)
                       for row in observation["board"])
    placements = _branch(observation, 0, board_rows)
    if observation["holdAvailable"] and (observation["hold"] or observation["next"][0]):
        placements.extend(_branch(observation, 1, board_rows))
    return sorted(placements, key=lambda placement: placement["actionId"])
