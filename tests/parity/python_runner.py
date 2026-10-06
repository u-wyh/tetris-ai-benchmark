"""JSON-lines-free command runner: one JSON request on stdin, one result on stdout."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.tetris_core import TetrisCore  # noqa: E402
from training.tetris_core.pieces import make_piece  # noqa: E402


def apply_fixture(core, fixture):
    if fixture in ("single_clear", "double_clear", "level_up"):
        core.current = make_piece("O")
        for x in range(10):
            if x not in (3, 4):
                core.board[19][x] = "Z"
                if fixture == "double_clear":
                    core.board[18][x] = "Z"
        if fixture == "level_up":
            core.lines, core.level = 9, 1
    elif fixture in ("triple_clear", "quad_clear"):
        core.current = make_piece("I")
        for y in range(17 if fixture == "triple_clear" else 16, 20):
            for x in range(10):
                if x != 4:
                    core.board[y][x] = "Z"
    elif fixture == "block_out":
        core.current = make_piece("O")
        core.board[2][4] = "Z"
    elif fixture == "lock_out":
        core.current = make_piece("I")
        core.current["y"] = -2
        for x in range(3, 7):
            core.board[0][x] = "Z"
    elif fixture == "partial_lock":
        core.current = make_piece("O")
        core.current["x"] = 0
        core.current["y"] = -1
        core.board[1][0] = "Z"
    elif fixture == "srs_floor":
        core.current = make_piece("T")
        core.current["y"] = 18
    elif fixture == "tuck":
        core.current = make_piece("T")
        rows = ("....#.....", "......#...", "#.......#.", ".##.......",
                "...##.....", ".....##...", ".......##.", ".#.......#")
        for y, row in enumerate(rows, 12):
            for x, cell in enumerate(row):
                if cell == "#":
                    core.board[y][x] = "Z"
    elif fixture is not None:
        raise ValueError(f"Unknown fixture: {fixture}")


def snapshot(core, step, include_placements=False):
    o = core.get_public_observation()
    result = {"step": step, "board": o["board"], "current": o["currentPiece"],
            "next3": o["next"], "hold": o["hold"],
            "holdAvailable": o["holdAvailable"], "score": o["score"],
            "lines": o["lines"], "level": o["level"], "gameOver": o["gameOver"],
            "actionMask": core.get_action_mask()}
    if include_placements:
        result["placements"] = core.get_legal_placements()
    return result


def run(request):
    core = TetrisCore(request["seed"])
    apply_fixture(core, request.get("fixture"))
    include = request.get("includePlacements", False)
    result = {"seed": request["seed"], "actions": [], "states": [snapshot(core, -1, include)]}
    for step, action in enumerate(request.get("actions", [])):
        if core.phase != "playing":
            break
        core.step(action)
        result["actions"].append(action)
        result["states"].append(snapshot(core, step, include))
    return result


if __name__ == "__main__":
    json.dump(run(json.load(sys.stdin)), sys.stdout, separators=(",", ":"))
