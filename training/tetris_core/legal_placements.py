"""Geometry-only BFS equivalent to tetris-game/legal-placements.js."""

from .actions import ACTION_COUNT, encode_action
from .pieces import CELLS, ROTATIONS, I_KICKS, JLSTZ_KICKS, collides, make_piece

STEPS = ("Left", "Right", "RotateCW", "Down")


def _branch(observation, hold):
    board = observation["board"]
    piece = (make_piece(observation["hold"] or observation["next"][0]) if hold
             else observation["currentPiece"])
    if piece is None:
        return []
    piece_type = piece["type"]
    rotation = ROTATIONS.index(piece["rotation"])
    collisions = {}

    def blocked(r, x, y):
        state = r, x, y
        result = collisions.get(state)
        if result is None:
            result = collides(board, piece_type, r, x, y)
            collisions[state] = result
        return result

    def clockwise(r, x, y):
        target = (r + 1) % 4
        if piece_type == "O":
            return x, y, target
        kicks = I_KICKS if piece_type == "I" else JLSTZ_KICKS
        for dx, dy_up in kicks[r]:
            nx, ny = x + dx, y - dy_up
            if not blocked(target, nx, ny):
                return nx, ny, target
        return None

    if blocked(rotation, piece["x"], piece["y"]):
        return []
    shapes = [CELLS[piece_type, index] for index in range(4)]
    bounds = [(min(x for x, _ in cells), min(y for _, y in cells)) for cells in shapes]
    # Each node contains geometry and a parent pointer; only geometry enters visited.
    nodes = [(piece["x"], piece["y"], rotation, -1, None)]
    visited = {(piece["x"], piece["y"], rotation)}
    by_cells = {}
    head = 0
    while head < len(nodes):
        x, y, rotation, _, _ = nodes[head]
        if blocked(rotation, x, y + 1):
            min_x, min_y = bounds[rotation]
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
                if blocked(nr, nx, ny):
                    continue
            elif step == "Right":
                nx += 1
                if blocked(nr, nx, ny):
                    continue
            elif step == "RotateCW":
                rotated = clockwise(nr, nx, ny)
                if rotated is None:
                    continue
                nx, ny, nr = rotated
            else:
                ny += 1
                if blocked(nr, nx, ny):
                    continue
            top = ny + bounds[nr][1]
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
    placements = _branch(observation, 0)
    if observation["holdAvailable"] and (observation["hold"] or observation["next"][0]):
        placements.extend(_branch(observation, 1))
    return sorted(placements, key=lambda placement: placement["actionId"])


def get_action_mask(observation):
    mask = [False] * ACTION_COUNT
    for placement in get_legal_placements(observation):
        mask[placement["actionId"]] = True
    return mask
