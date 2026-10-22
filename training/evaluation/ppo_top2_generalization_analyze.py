"""Seed-level summary of bounded Top-1/Top-2 comparisons."""

import hashlib
import json
from pathlib import Path

from training.evaluation.ppo_top2_generalization_capture import DEV_SEEDS, RUN, TRAIN_SEEDS
from training.train_ppo import atomic_json, now


def main():
    status = json.loads((RUN / "branch_status.json").read_text())
    if status["stage"] != "complete" or status["completed"] != len(TRAIN_SEEDS + DEV_SEEDS):
        raise RuntimeError("Paired branch study is incomplete")
    integrity = json.loads((RUN / "branch_integrity.json").read_text())
    for name, expected in integrity["files"].items():
        path = RUN / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Branch artifact checksum mismatch: {name}")
    groups = {}
    for name, seeds in (("candidate", TRAIN_SEEDS), ("replication", DEV_SEEDS)):
        rows = []
        for seed in seeds:
            path = RUN / "comparisons" / f"seed_{seed}.json"
            if not path.is_file():
                raise RuntimeError(f"Missing comparison: {seed}")
            row = json.loads(path.read_text())
            if row["seed"] != seed or row["group"] != name:
                raise RuntimeError(f"Comparison identity changed: {seed}")
            rows.append(row)
        deaths = [row for row in rows if row["original_game_over"]]
        controls = [row for row in rows if not row["original_game_over"]]
        def stats(items):
            return {"seeds": len(items),
                    "strict_rescues": sum(row["comparison"]["rescued_in_window"] for row in items),
                    "strict_harms": sum(row["comparison"]["harmed_in_window"] for row in items),
                    "top2_true_deaths": sum(row["top2_game_over"] for row in items),
                    "top2_right_censored": sum(row["top2_censored"] for row in items),
                    "both_right_censored": sum(row["comparison"]["ppo_right_censored"]
                                               and row["comparison"]["alternative_right_censored"]
                                               for row in items),
                    "mean_common_score_ratio": (sum(row["comparison"]["score_ratio"] for row in items)
                                                / len(items) if items else None),
                    "mean_common_four_delta": (sum(row["comparison"]["candidate_fours"]
                                                   - row["comparison"]["ppo_fours"] for row in items)
                                               / len(items) if items else None)}
        groups[name] = {"death_states": stats(deaths), "survival_controls": stats(controls),
                        "per_seed": [{"seed": row["seed"], "original_game_over": row["original_game_over"],
                                      "step": row["step"], "top1_action": row["top1_action"],
                                      "top2_action": row["top2_action"], "top2_pieces": row["top2_pieces"],
                                      "top2_game_over": row["top2_game_over"],
                                      "top2_censored": row["top2_censored"],
                                      "comparison": row["comparison"]} for row in rows]}
    atomic_json(RUN / "analysis.json", {"version": "top2_generalization_analysis_v1",
                                        "created_at": now(), "groups": groups,
                                        "branch_integrity_sha256": hashlib.sha256(
                                            (RUN / "branch_integrity.json").read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
