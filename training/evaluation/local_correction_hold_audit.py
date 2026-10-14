"""Public one-step consequences of V1 trusted Hold and non-Hold labels."""
import gzip
import json

from training.evaluation.local_correction_active_v2 import V2
from training.local_correction_data import RUN, TRAIN, VALIDATION, _paths
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public
from training.train_ppo import atomic_json


def _heights(board):
    return [next((len(board) - y for y, row in enumerate(board) if row[x] is not None), 0)
            for x in range(len(board[0]))]


def _well(board):
    heights = _heights(board)
    depths = []
    for i, h in enumerate(heights):
        neighbors = [heights[j] for j in (i - 1, i + 1) if 0 <= j < len(heights)]
        if neighbors:
            depths.append(max(0, min(neighbors) - h))
    return max(depths, default=0)


def main():
    out = V2 / "hold_audit.json"
    if out.exists():
        return json.loads(out.read_text())
    records = []
    for seed in TRAIN + VALIDATION:
        shard, _, _ = _paths(seed)
        with gzip.open(shard, "rt", encoding="utf8") as stream:
            rows = {row["step"]: row for row in map(json.loads, stream)}
        labels = json.loads((RUN / "labels" / f"seed_{seed}.json").read_text())["labels"]
        for step, evidence in labels.items():
            if evidence["category"] != "A":
                continue
            row = rows[int(step)]
            public = row["public"]
            placements = {p["actionId"]: p for p in get_legal_placements(public)}
            states = {name: simulate_public(public, placements[row[f"{name}_action"]])[0]
                      for name in ("ppo", "bc")}
            pp, bc = states["ppo"], states["bc"]
            records.append({"seed": seed, "step": int(step),
                            "hold_flag_differs": evidence["hold_changed"],
                            "before_current": public["currentPiece"]["type"],
                            "before_hold": public["hold"], "before_next": public["next"],
                            "after_ppo": {"current": pp["currentPiece"]["type"] if pp["currentPiece"] else None,
                                          "hold": pp["hold"], "next": pp["next"], "well_depth": _well(pp["board"])},
                            "after_bc": {"current": bc["currentPiece"]["type"] if bc["currentPiece"] else None,
                                         "hold": bc["hold"], "next": bc["next"], "well_depth": _well(bc["board"])},
                            "future_piece_state_differs": (pp["hold"], pp["next"],
                                                             pp["currentPiece"]["type"] if pp["currentPiece"] else None)
                            != (bc["hold"], bc["next"],
                                bc["currentPiece"]["type"] if bc["currentPiece"] else None),
                            "board_differs": pp["board"] != bc["board"]})
    atomic_json(out, {"version": "public_one_step_hold_audit_v1", "records": records})
    return {"records": records}


if __name__ == "__main__":
    main()
