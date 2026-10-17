"""Bitboard collision acceleration for the unchanged public placement BFS."""

from training.tetris_core.actions import encode_action
from training.tetris_core.pieces import CELLS, I_KICKS, JLSTZ_KICKS, ROTATIONS, make_piece

STEPS = ("Left", "Right", "RotateCW", "Down")
SHAPES = {}
BOUNDS = {}
for _piece in "IJLOSTZ":
    for _rotation in range(4):
        _cells = CELLS[_piece, _rotation]
        _rows = {}
        for _x, _y in _cells:
            _rows[_y] = _rows.get(_y, 0) | (1 << _x)
        SHAPES[_piece, _rotation] = tuple(sorted(_rows.items()))
        BOUNDS[_piece, _rotation] = (min(x for x, _ in _cells),
                                     max(x for x, _ in _cells),
                                     min(y for _, y in _cells),
                                     max(y for _, y in _cells))


def _branch(observation, hold, board_rows):
    piece = (make_piece(observation["hold"] or observation["next"][0]) if hold
             else observation["currentPiece"])
    if piece is None:
        return []
    kind = piece["type"]
    shapes = [CELLS[kind, r] for r in range(4)]
    bounds = [BOUNDS[kind, r] for r in range(4)]
    row_shapes = [SHAPES[kind, r] for r in range(4)]

    def collides(rotation, x, y):
        low_x, high_x, _, high_y = bounds[rotation]
        if x + low_x < 0 or x + high_x >= 10 or y + high_y >= 20:
            return True
        for dy, bits in row_shapes[rotation]:
            row = y + dy
            if row >= 0 and board_rows[row] & (bits << x if x >= 0 else bits >> -x):
                return True
        return False

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

    rotation = ROTATIONS.index(piece["rotation"])
    if collides(rotation, piece["x"], piece["y"]):
        return []
    nodes = [(piece["x"], piece["y"], rotation, -1, None)]
    visited = {(piece["x"], piece["y"], rotation)}
    by_cells = {}
    head = 0
    while head < len(nodes):
        x, y, rotation, _, _ = nodes[head]
        if collides(rotation, x, y + 1):
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
        for step in STEPS:
            nx, ny, nr = x, y, rotation
            if step == "Left":
                nx -= 1
                if collides(nr, nx, ny):
                    continue
            elif step == "Right":
                nx += 1
                if collides(nr, nx, ny):
                    continue
            elif step == "RotateCW":
                rotated = rotate(nr, nx, ny)
                if rotated is None:
                    continue
                nx, ny, nr = rotated
            else:
                ny += 1
                if collides(nr, nx, ny):
                    continue
            top = ny + bounds[nr][2]
            if not -6 <= top <= 19:
                continue
            state = nx, ny, nr
            if state not in visited:
                visited.add(state)
                nodes.append((nx, ny, nr, head, step))
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
