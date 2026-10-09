"""Paired board diagnostics for five completed 2M PPO checkpoints."""
import gzip
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from analysis.ppo_raw_diagnostics import board_features, sha256
from training.env import TetrisEnv
from training.train_ppo import ROOT, atomic_json

RUN = ROOT / "runs/ppo_overnight_height_reward"
OLD = ROOT / "reports/experiments/ppo_hole_v1_failure_traces"
OUT = ROOT / "reports/experiments/ppo_five_model_board_diagnostics.json"
SEEDS = list(range(100000, 100032))
MODELS = {
    "raw_2m": ROOT / "runs/ppo_raw_10m_seed42/milestones/step_002002944/model.zip",
    "shaped_v1_2m": ROOT / "runs/ppo_hole_v1_2m_seed42_lambda010/milestones/step_002002944/model.zip",
    "shaped_v2_2m": ROOT / "runs/ppo_hole_v2_2m_seed42_lambda002/milestones/step_002002944/model.zip",
    "v3_2m": ROOT / "runs/ppo_hole_height_v3_2m_seed42/milestones/step_002002944/model.zip",
    "v4_2m": ROOT / "runs/ppo_height_v4_2m_seed42/milestones/step_002002944/model.zip",
}


def bootstrap_mean(values, draws=10000, seed=20261009):
    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(sample, [0.025, 0.975])]


def death_row(seed, pieces, lines, score, new_holes, board, first15, game_over):
    features = board_features(board)
    heights = [20 - next((y for y, row in enumerate(board) if row[x]), 20)
               for x in range(10)]
    tallest = max(range(10), key=heights.__getitem__)
    return {"seed": seed, "pieces_survived": pieces, "lines": lines, "score": score,
            "new_holes_total": new_holes, "game_over": game_over,
            "max_height_at_death": features["max_height"] if game_over else None,
            "aggregate_height_at_death": features["aggregate_height"] if game_over else None,
            "holes_at_death": features["holes"] if game_over else None,
            "tallest_column": tallest if game_over else None,
            "central_tallest": tallest in (4, 5) if game_over else None,
            "first_height_15": first15,
            "remaining_after_15": pieces - first15 if first15 is not None else None}


def old_rows(label):
    rows = []
    for seed in SEEDS:
        with gzip.open(OLD / f"{label}_seed_{seed}.json.gz", "rt") as stream:
            episode = json.load(stream)
        last = episode["steps"][-1]
        board = [[None if cell == "." else cell for cell in line]
                 for line in last["board_after"]]
        rows.append(death_row(seed, episode["pieces_survived"], episode["lines"],
                              episode["score"], sum(s["new_holes"] for s in episode["steps"]),
                              board, episode["first_max_height"]["15"], episode["game_over"]))
    return rows


def replay_rows(path):
    model = MaskablePPO.load(str(path), device="cpu")
    env = TetrisEnv(max_pieces=5000)
    rows = []
    try:
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            first15 = None
            new_holes = 0
            while True:
                mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                if not mask[int(action)]:
                    raise AssertionError("Model selected illegal placement")
                obs, _, terminated, truncated, info = env.step(int(action))
                new_holes += info["new_holes"]
                if first15 is None and board_features(env.core.board)["max_height"] >= 15:
                    first15 = info["pieces"]
                if terminated or truncated:
                    rows.append(death_row(seed, info["pieces"], info["lines"], info["score"],
                                          new_holes, env.core.board, first15, bool(terminated)))
                    break
    finally:
        env.close()
    return rows


def audit(rows, expected):
    if expected["seeds"] != SEEDS or expected["max_pieces"] != 5000:
        raise AssertionError("Validation protocol differs")
    for row, old in zip(rows, expected["per_seed"], strict=True):
        if any(row[field] != old[field] for field in
               ("seed", "pieces_survived", "lines", "score", "new_holes_total", "game_over")):
            raise AssertionError(f"Trajectory and existing evaluation differ at seed {row['seed']}")


def summarize(rows):
    deaths = [r for r in rows if r["game_over"]]
    reached = [r["remaining_after_15"] for r in rows if r["remaining_after_15"] is not None]
    return {"mean_pieces": statistics.mean(r["pieces_survived"] for r in rows),
            "median_pieces": statistics.median(r["pieces_survived"] for r in rows),
            "mean_lines": statistics.mean(r["lines"] for r in rows),
            "mean_score": statistics.mean(r["score"] for r in rows),
            "new_holes_per_100_pieces": 100 * sum(r["new_holes_total"] for r in rows)
            / sum(r["pieces_survived"] for r in rows),
            "deaths": len(deaths),
            "mean_death_max_height": statistics.mean(r["max_height_at_death"] for r in deaths),
            "mean_death_aggregate_height": statistics.mean(r["aggregate_height_at_death"] for r in deaths),
            "mean_death_holes": statistics.mean(r["holes_at_death"] for r in deaths),
            "central_tallest_count": sum(r["central_tallest"] for r in deaths),
            "central_tallest_rate": sum(r["central_tallest"] for r in deaths) / len(deaths),
            "reached_height_15": len(reached),
            "mean_remaining_after_15": statistics.mean(reached) if reached else None}


def paired(rows, baseline):
    if [r["seed"] for r in rows] != [r["seed"] for r in baseline]:
        raise AssertionError("Paired seed order differs")
    differences = [a["pieces_survived"] - b["pieces_survived"]
                   for a, b in zip(rows, baseline)]
    return {"mean_difference": statistics.mean(differences),
            "median_difference": statistics.median(differences),
            "bootstrap_95ci": bootstrap_mean(differences),
            "better": sum(d > 0 for d in differences),
            "worse": sum(d < 0 for d in differences), "ties": sum(d == 0 for d in differences),
            "per_seed": dict(zip(SEEDS, differences))}


def run():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {OUT}")
    torch.set_num_threads(1)
    old = json.loads((RUN / "evaluation_existing.json").read_text())
    old["v3_2m"] = json.loads((RUN / "evaluation_v3.json").read_text())
    old["v4_2m"] = json.loads((RUN / "evaluation_v4.json").read_text())
    rows = {}
    for label, path in MODELS.items():
        rows[label] = old_rows("raw" if label == "raw_2m" else "shaped") if label in (
            "raw_2m", "shaped_v1_2m") else replay_rows(path)
        audit(rows[label], old[label])
    result = {"seeds": SEEDS, "max_pieces": 5000, "deterministic": True,
              "checkpoint_sha256": {label: sha256(path) for label, path in MODELS.items()},
              "models": {label: {"summary": summarize(data), "per_seed": data}
                         for label, data in rows.items()},
              "paired_vs_raw": {label: paired(data, rows["raw_2m"])
                                for label, data in rows.items() if label != "raw_2m"},
              "paired_v3_vs_v2": paired(rows["v3_2m"], rows["shaped_v2_2m"]),
              "paired_v4_vs_v3": paired(rows["v4_2m"], rows["v3_2m"])}
    atomic_json(OUT, result)
    return result


if __name__ == "__main__":
    result = run()
    for name, data in result["models"].items():
        print(name, data["summary"])
