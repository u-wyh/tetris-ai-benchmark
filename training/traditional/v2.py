"""Deterministic public-observation traditional policy with legal BFS lookahead."""
from dataclasses import dataclass, field
from time import perf_counter_ns

from training.tetris_core.legal_placements import get_legal_placements
from training.tetris_core.pieces import CELLS, collides, make_piece, rotate_cw

VERSION = "v2-1"
DEFAULT_WEIGHTS = {
    "cleared": 8.2, "aggregate_height": -0.47, "holes": -4.2,
    "covered_holes": -0.16, "bumpiness": -0.28, "max_height": -0.18,
    "wells": -0.08, "row_transitions": -0.14,
    "column_transitions": -0.22, "top_risk": -0.9,
}


@dataclass(frozen=True)
class V2Config:
    mode: str = "beam"  # heuristic, hold, beam
    beam_width: int = 8
    lookahead: float = .58
    weights: dict = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    death_penalty: float = 1000000.0

    def __post_init__(self):
        if self.mode not in ("heuristic", "hold", "beam"):
            raise ValueError("Unknown V2 mode")
        if self.beam_width < 1 or not 0 <= self.lookahead <= 1:
            raise ValueError("Invalid search parameters")
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            raise ValueError("V2 weights must specify every feature")


def features(board):
    heights, holes, covered = [], 0, 0
    for x in range(10):
        first, above = 20, 0
        for y in range(20):
            occupied = bool(board[y][x])
            if occupied:
                first = min(first, y)
                above += 1
            elif first != 20:
                holes += 1
                covered += above
        heights.append(20 - first)
    wells = sum(max(0, min(heights[x-1] if x else 20,
                          heights[x+1] if x < 9 else 20) - heights[x]) for x in range(10))
    row_transitions = 0
    for row in board:
        previous = True  # side walls are occupied
        for cell in row:
            occupied = bool(cell)
            row_transitions += previous != occupied
            previous = occupied
        row_transitions += not previous
    column_transitions = 0
    for x in range(10):
        previous = False  # above the board is empty
        for y in range(20):
            occupied = bool(board[y][x])
            column_transitions += previous != occupied
            previous = occupied
        column_transitions += not previous  # floor is occupied
    maximum = max(heights)
    return dict(aggregate_height=sum(heights), holes=holes, covered_holes=covered,
                bumpiness=sum(abs(a-b) for a,b in zip(heights,heights[1:])),
                max_height=maximum, wells=wells, row_transitions=row_transitions,
                column_transitions=column_transitions,
                top_risk=max(0, maximum-13)**2)


def board_score(board, cleared, weights):
    values = features(board)
    values["cleared"] = cleared
    return sum(weights[name] * value for name,value in values.items())


def _drop_distance(observation, placement):
    board = observation["board"]
    piece = (make_piece(observation["hold"] or observation["next"][0]) if placement["hold"]
             else observation["currentPiece"])
    piece_type = piece["type"]
    x, y, rotation = piece["x"], piece["y"], "0R2L".index(piece["rotation"])
    for step in placement["path"]:
        if step == "Left":
            x -= 1
        elif step == "Right":
            x += 1
        elif step == "Down":
            y += 1
        elif step == "RotateCW":
            x, y, rotation = rotate_cw(board, piece_type, rotation, x, y)
    distance = 0
    while not collides(board, piece_type, rotation, x, y+1):
        y += 1
        distance += 1
    return distance


def simulate_public(observation, placement):
    """Apply one canonical placement using visible pieces only.

    After all visible Next pieces are consumed, phase='unknown': further
    lookahead must stop, while the real Core may draw another hidden piece.
    """
    visible = list(observation["next"])
    held = observation["hold"]
    current = observation["currentPiece"]["type"]
    if placement["hold"]:
        if not observation["holdAvailable"]:
            raise ValueError("Hold is unavailable")
        piece_type = held or visible.pop(0)
        held = current
    else:
        piece_type = current
    board = [row[:] for row in observation["board"]]
    score = observation["score"] + _drop_distance(observation, placement)*2
    lines = observation["lines"]
    level = observation["level"]
    occupied = placement["occupiedCells"]
    if not any(cell["y"] >= 0 for cell in occupied):
        return dict(board=board, currentPiece=None, hold=held, holdAvailable=False,
                    next=visible, score=score, lines=lines, level=level,
                    phase="over", gameOver=True), 0
    for cell in occupied:
        if cell["y"] >= 0:
            board[cell["y"]][cell["x"]] = piece_type
    remaining = [row for row in board if not all(row)]
    cleared = 20 - len(remaining)
    board = [[None]*10 for _ in range(cleared)] + remaining
    if cleared:
        lines += cleared
        score += (0,100,300,500,800)[cleared] * level
        level = lines//10 + 1
    if visible:
        following = make_piece(visible.pop(0))
        over = collides(board, following["type"], 0, following["x"], following["y"])
        phase = "over" if over else "playing"
        if over:
            following = None
    else:
        following, phase = None, "unknown"
    return dict(board=board, currentPiece=following, hold=held,
                holdAvailable=phase == "playing", next=visible,
                score=score, lines=lines, level=level, phase=phase,
                gameOver=phase == "over"), cleared


def _state_key(observation):
    return (tuple(tuple(bool(cell) for cell in row) for row in observation["board"]),
            observation["currentPiece"]["type"] if observation["currentPiece"] else None,
            observation["hold"], tuple(observation["next"]), observation["phase"])


class V2:
    def __init__(self, config=None):
        self.config = config or V2Config()
        self.last_timing_ms = {}

    def choose(self, observation, placements=None):
        started = perf_counter_ns()
        root = get_legal_placements(observation) if placements is None else placements
        legal_ms = (perf_counter_ns() - started)/1e6 if placements is None else 0.0
        allow_hold = self.config.mode != "heuristic"
        candidates = [p for p in root if allow_hold or not p["hold"]]
        if not candidates:
            raise ValueError("V2 has no allowed legal placement")
        evaluated, simulate_ns, future_legal_ns = [], 0, 0
        for placement in candidates:
            began = perf_counter_ns()
            state, cleared = simulate_public(observation, placement)
            value = board_score(state["board"], cleared, self.config.weights)
            if state["gameOver"]:
                value -= self.config.death_penalty
            simulate_ns += perf_counter_ns() - began
            evaluated.append((value, placement["actionId"], state))
        # Dedup equivalent visible states, retaining the smallest Action ID.
        unique = {}
        for value, action, state in evaluated:
            key = _state_key(state)
            old = unique.get(key)
            if old is None or (value, -action) > (old[0], -old[1]):
                unique[key] = (value, action, state)
        ranked = sorted(unique.values(), key=lambda item: (-item[0], item[1]))
        if self.config.mode != "beam":
            chosen = ranked[0][1]
        else:
            best = None
            cache = {}
            for value, action, state in ranked[:self.config.beam_width]:
                future = 0.0
                if state["phase"] == "playing":
                    key = _state_key(state)
                    if key not in cache:
                        began = perf_counter_ns()
                        cache[key] = get_legal_placements(state)
                        future_legal_ns += perf_counter_ns() - began
                    second = cache[key]
                    if second:
                        second_best = float("-inf")
                        for placement in second:
                            began = perf_counter_ns()
                            next_state, cleared = simulate_public(state, placement)
                            score = board_score(next_state["board"], cleared, self.config.weights)
                            if next_state["gameOver"]:
                                score -= self.config.death_penalty
                            simulate_ns += perf_counter_ns() - began
                            second_best = max(second_best, score)
                        future = self.config.lookahead * second_best
                    else:
                        future = -self.config.death_penalty
                score = value + future
                if best is None or (score, -action) > (best[0], -best[1]):
                    best = (score, action)
            chosen = best[1]
        total_ms = (perf_counter_ns() - started)/1e6
        self.last_timing_ms = dict(root_legal=legal_ms, simulation_features=simulate_ns/1e6,
                                   future_legal=future_legal_ns/1e6,
                                   search_overhead=max(0,total_ms-legal_ms-simulate_ns/1e6-future_legal_ns/1e6),
                                   policy_total=total_ms)
        return chosen
