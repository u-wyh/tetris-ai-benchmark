"""Placement-level Benchmark v1.0 core; no browser timer or human lock delay."""

from .actions import ACTION_COUNT
from .legal_placements import get_action_mask, get_legal_placements
from .pieces import COLS, ROWS, MATRICES, ROTATIONS, CELLS, TYPES, collides, make_piece, rotate_cw
from .rng import Mulberry32


class TetrisCore:
    def __init__(self, seed):
        self.reset(seed)

    def reset(self, seed):
        self.initial_seed = seed
        self.rng = Mulberry32(seed)
        self.board = [[None] * COLS for _ in range(ROWS)]
        self.queue = []
        self.bag = []
        self.hold = None
        self.hold_used = False
        self.current = None
        self.score = 0
        self.lines = 0
        self.level = 1
        self.phase = "playing"
        self._spawn()
        return self.get_public_observation()

    def _fill_queue(self):
        while len(self.queue) < 5:
            if not self.bag:
                types = list(TYPES)
                for i in range(len(types) - 1, 0, -1):
                    j = int(self.rng.random() * (i + 1))
                    types[i], types[j] = types[j], types[i]
                self.bag = types
            self.queue.append(self.bag.pop(0))

    def _activate(self, piece_type):
        self.current = make_piece(piece_type)
        piece = self.current
        if collides(self.board, piece_type, 0, piece["x"], piece["y"]):
            self.phase = "over"  # Block Out

    def _spawn(self, reset_hold=True):
        self._fill_queue()
        piece_type = self.queue.pop(0)
        self._fill_queue()
        if reset_hold:
            self.hold_used = False
        self._activate(piece_type)

    def _hold(self):
        if self.phase != "playing" or self.hold_used:
            raise ValueError("Hold is unavailable")
        outgoing = self.current["type"]
        incoming = self.hold
        self.hold = outgoing
        self.hold_used = True
        if incoming is None:
            self._spawn(reset_hold=False)
        else:
            self._activate(incoming)

    def _merge(self):
        piece = self.current
        rotation = ROTATIONS.index(piece["rotation"])
        visible = False
        for dx, dy in CELLS[piece["type"], rotation]:
            x, y = piece["x"] + dx, piece["y"] + dy
            if y >= 0:
                self.board[y][x] = piece["type"]
                visible = True
        if not visible:
            self.phase = "over"  # Lock Out; do not spawn or clear rows.
            return
        cleared = 0
        y = ROWS - 1
        while y >= 0:
            if all(self.board[y]):
                self.board.pop(y)
                self.board.insert(0, [None] * COLS)
                cleared += 1
            else:
                y -= 1
        if cleared:
            old_level = self.level
            self.lines += cleared
            self.level = self.lines // 10 + 1
            self.score += (0, 100, 300, 500, 800)[cleared] * old_level
        self._spawn()

    def get_public_observation(self):
        piece = None
        if self.current is not None and self.phase != "over":
            piece = {"type": self.current["type"],
                     "matrix": [row[:] for row in self.current["matrix"]],
                     "rotation": self.current["rotation"], "x": self.current["x"],
                     "y": self.current["y"]}
        return {"board": [row[:] for row in self.board], "currentPiece": piece,
                "hold": self.hold, "holdAvailable": self.phase == "playing" and not self.hold_used,
                "next": self.queue[:3], "score": self.score, "level": self.level,
                "lines": self.lines, "phase": self.phase, "gameOver": self.phase == "over"}

    def get_legal_placements(self):
        return get_legal_placements(self.get_public_observation())

    def get_action_mask(self):
        return get_action_mask(self.get_public_observation())

    def step(self, action):
        if type(action) is not int or not 0 <= action < ACTION_COUNT:
            raise ValueError("Action ID must be an integer from 0 to 1839")
        placement = next((item for item in self.get_legal_placements()
                          if item["actionId"] == action), None)
        if placement is None:
            raise ValueError(f"Action {action} is not a legal placement")
        for step in placement["path"]:
            if step == "Hold":
                self._hold()
            elif step == "RotateCW":
                piece = self.current
                result = rotate_cw(self.board, piece["type"], ROTATIONS.index(piece["rotation"]),
                                   piece["x"], piece["y"])
                if result is None:
                    raise AssertionError("Legal placement path has a failed rotation")
                piece["x"], piece["y"], rotation = result
                piece["rotation"] = ROTATIONS[rotation]
                piece["matrix"] = [list(row) for row in MATRICES[piece["type"], rotation]]
            elif step in ("Left", "Right", "Down"):
                piece = self.current
                dx = -1 if step == "Left" else 1 if step == "Right" else 0
                dy = 1 if step == "Down" else 0
                rotation = ROTATIONS.index(piece["rotation"])
                if collides(self.board, piece["type"], rotation, piece["x"] + dx, piece["y"] + dy):
                    raise AssertionError("Legal placement path has a failed move")
                piece["x"] += dx
                piece["y"] += dy
            else:
                piece = self.current
                rotation = ROTATIONS.index(piece["rotation"])
                distance = 0
                while not collides(self.board, piece["type"], rotation, piece["x"], piece["y"] + 1):
                    piece["y"] += 1
                    distance += 1
                self.score += distance * 2
                self._merge()
        return self.get_public_observation()
