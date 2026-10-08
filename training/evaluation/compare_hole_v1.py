"""Paired Raw-2M versus Shaped-2M validation, only after Shaped has completed."""

import argparse
import json
import statistics
from pathlib import Path

from training.evaluation.evaluate import evaluate_model
from training.train_ppo import ROOT, atomic_json


def paired_summary(raw, shaped):
    if (raw["seeds"] != shaped["seeds"] or raw["max_pieces"] != shaped["max_pieces"]
            or len(raw["per_seed"]) != len(shaped["per_seed"])
            or [row["seed"] for row in raw["per_seed"]] != raw["seeds"]
            or [row["seed"] for row in shaped["per_seed"]] != shaped["seeds"]):
        raise ValueError("Paired evaluation needs identical seeds and cap")
    fields = ("pieces_survived", "lines", "score", "new_holes_total")
    result = {}
    for field in fields:
        differences = [b[field] - a[field]
                       for a, b in zip(raw["per_seed"], shaped["per_seed"])]
        result[field] = {"mean_difference": statistics.mean(differences),
                         "median_difference": statistics.median(differences),
                         "shaped_better": sum(value > 0 for value in differences),
                         "raw_better": sum(value < 0 for value in differences),
                         "ties": sum(value == 0 for value in differences)}
    return result


def compare(shaped_run, output):
    shaped_run = Path(shaped_run)
    state = json.loads((shaped_run / "transaction_state.json").read_text())
    config = json.loads((shaped_run / "config.json").read_text())
    if (state["status"] != "completed" or state["committed_steps"] != 2_002_944
            or config.get("reward_version") != "placement-reward-hole-v1"
            or config.get("hole_penalty_coef") != 0.1):
        raise ValueError("Expected completed 2M Shaped V1 run with coefficient 0.10")
    seeds = json.loads((ROOT / "training/evaluation/seeds.json").read_text())["validation"]
    if len(seeds) != 32:
        raise ValueError("Expected exactly 32 fixed validation seeds")
    raw_model = ROOT / "runs/ppo_raw_10m_seed42/milestones/step_002002944/model.zip"
    shaped_model = shaped_run / "final/model.zip"
    if not raw_model.is_file() or not shaped_model.is_file():
        raise FileNotFoundError("Raw or Shaped 2M checkpoint is missing")
    if Path(output).exists():
        raise FileExistsError("Comparison output already exists")
    raw = evaluate_model(raw_model, seeds, 5000, "paired_validation_2m", 2_002_944,
                         diagnostics=True)
    shaped = evaluate_model(shaped_model, seeds, 5000, "paired_validation_2m", 2_002_944,
                            diagnostics=True)
    result = {"seed_set": "validation", "max_pieces": 5000,
              "raw": raw, "shaped": shaped,
              "paired": paired_summary(raw, shaped)}
    atomic_json(Path(output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shaped-run", type=Path,
                        default=ROOT / "runs/ppo_hole_v1_2m_seed42_lambda010")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.shaped_run, args.output)


if __name__ == "__main__":
    main()
