"""Legal-placement adaptation of browser V1, not an original-browser measurement."""
from training.tetris_core.legal_placements import get_legal_placements
from training.tetris_core.pieces import make_piece
from training.tetris_core.rng import Mulberry32

VERSION = "v1-adapter-1"
WEIGHTS = dict(cleared=8.2, aggregate_height=-.47, holes=-4.2,
               covered_holes=-.16, bumpiness=-.28, max_height=-.18, wells=-.08)


def board_value(board, cleared=0):
    heights = []
    holes = covered = 0
    for x in range(10):
        first, above = 20, 0
        for y in range(20):
            if board[y][x]:
                first = min(first, y)
                above += 1
            elif first != 20:
                holes += 1
                covered += above
        heights.append(20 - first)
    wells = sum(max(0, min(heights[x-1] if x else 20,
                          heights[x+1] if x < 9 else 20) - heights[x]) for x in range(10))
    return (cleared * 8.2 - sum(heights) * .47 - holes * 4.2 - covered * .16
            - sum(abs(a-b) for a, b in zip(heights, heights[1:])) * .28
            - max(heights) * .18 - wells * .08)


def simulate(board, placement):
    # Browser V1 rejects any above-board cell, unlike Core's partial lock rule.
    if any(cell["y"] < 0 for cell in placement["occupiedCells"]):
        return None
    result = [row[:] for row in board]
    for cell in placement["occupiedCells"]:
        result[cell["y"]][cell["x"]] = "T"
    remaining = [row for row in result if not all(row)]
    cleared = 20 - len(remaining)
    return [[None] * 10 for _ in range(cleared)] + remaining, cleared


class V1Adapter:
    def __init__(self, policy_seed):
        self.rng = Mulberry32(policy_seed ^ 0x9e3779b9)

    def choose(self, observation, placements=None):
        placements = get_legal_placements(observation) if placements is None else placements
        normal = [p for p in placements if not p["hold"]]
        if not normal:
            raise ValueError("V1 has no legal non-Hold placement")
        best, best_value = normal[0], float("-inf")
        for placement in normal:
            simulation = simulate(observation["board"], placement)
            if simulation is None:
                continue
            board, cleared = simulation
            value = board_value(board, cleared)
            following = dict(board=board, currentPiece=make_piece(observation["next"][0]),
                             hold=None, holdAvailable=False, next=[], phase="playing")
            next_values = []
            for candidate in get_legal_placements(following):
                result = simulate(board, candidate)
                if result is not None:
                    next_values.append(board_value(*result))
            if next_values:
                value += .58 * max(next_values)
            value += self.rng.random() * .002
            if value > best_value:
                best, best_value = placement, value
        # If all candidates partially lock above the board, pick the first legal
        # action instead of stalling forever as the browser planner can do.
        return best["actionId"]
