"""Read-only diagnostics for the completed PPO-Raw 10M run.

Outputs go to a separate ignored analysis directory and tracked report directory.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics as stats
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sb3_contrib import MaskablePPO

from training.env import TetrisEnv
from training.evaluation.evaluate import aggregate, evaluate_model
from training.train_ppo import atomic_json


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/ppo_raw_10m_seed42"
OUT = ROOT / "runs/ppo_raw_10m_seed42_analysis"
REPORT = ROOT / "reports/experiments"
SEEDS = ROOT / "training/evaluation/seeds.json"


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def board_features(board):
    heights = []
    holes = covered_holes = 0
    for x in range(10):
        first = 20
        blocks_above = 0
        for y in range(20):
            if board[y][x]:
                if first == 20:
                    first = y
                blocks_above += 1
            elif first != 20:
                holes += 1
                covered_holes += blocks_above
        heights.append(20 - first)
    wells = 0
    for x, height in enumerate(heights):
        left = 20 if x == 0 else heights[x - 1]
        right = 20 if x == 9 else heights[x + 1]
        wells += max(0, min(left, right) - height)
    return {"max_height": max(heights), "aggregate_height": sum(heights),
            "holes": holes, "covered_holes": covered_holes,
            "bumpiness": sum(abs(a - b) for a, b in zip(heights, heights[1:])),
            "well_depth": wells}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_run():
    state = json.loads((RUN / "transaction_state.json").read_text())
    meta = json.loads((RUN / "metadata.json").read_text())
    config = json.loads((RUN / "config.json").read_text())
    assert state["status"] == "completed"
    assert state["committed_steps"] == state["target_steps"] == 10002432
    assert state["committed_task"] == 2442
    assert config["requested_target_steps"] == 10000000
    assert sha256(SEEDS) == config["evaluation_seed_sha256"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", meta["git_commit"], head],
                   cwd=ROOT, check=True)
    checkpoints = {}
    for directory in sorted((RUN / "milestones").glob("step_*")):
        model = directory / "model.zip"
        checksum = json.loads((directory / "checksum.json").read_text())["model_sha256"]
        assert model.is_file() and sha256(model) == checksum, directory
        checkpoints[directory.name] = str(model)
    assert len(checkpoints) == 10
    best = RUN / "best/model.zip"
    assert sha256(best) == json.loads((RUN / "best/checksum.json").read_text())["model_sha256"]
    final = RUN / "committed/task_002442"
    manifest = json.loads((final / "manifest.json").read_text())["files"]
    for name, details in manifest.items():
        path = final / name
        assert path.stat().st_size == details["size"] and sha256(path) == details["sha256"]
    with (RUN / "evaluations/validation_summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    files = list((RUN / "evaluations/validation").glob("*.json"))
    assert len(summary) == len(files) == 51
    assert not any((RUN / "evaluations/working").iterdir())
    summary_keys = {(int(row["committed_steps"]), int(row["num_seeds"]),
                     round(float(row["mean_pieces"]), 8)) for row in summary}
    file_keys = set()
    for path in files:
        result = json.loads(path.read_text())
        assert len(result["per_seed"]) == result["num_seeds"]
        assert result["num_seeds"] in (16, 32)
        assert result["max_pieces"] in (5000, 10000)
        file_keys.add((result["committed_steps"], result["num_seeds"],
                       round(result["aggregate"]["mean_pieces"], 8)))
    assert summary_keys == file_keys and len(file_keys) == 51
    return {"run_name": RUN.name, "git_commit": meta["git_commit"],
            "git_head_at_audit": head, "head_matches_run_commit": head == meta["git_commit"],
            "requested_steps": config["requested_target_steps"],
            "actual_committed_steps": state["committed_steps"],
            "final_task": state["committed_task"], "best_step": 9502720,
            "milestones": len(checkpoints), "validation_records": len(files)}


def trace_model(label, model_path, seeds):
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(model_path), device="cpu")
    env = TetrisEnv(max_pieces=50000)
    rows = []
    death = []
    first_clear = []
    first_hole_to_death = []
    ten_holes_to_death = []
    last_hole_to_death = []
    placements = Counter()
    rotations = Counter()
    xs = Counter()
    ys = []
    legal_counts = []
    ranks = []
    entropies = []
    normalized_entropies = []
    line_clear_sizes = Counter()
    created_holes = 0
    sampled = 0
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed))
            first = None
            first_hole = None
            ten_holes = None
            last_hole = None
            done = False
            while not done:
                before = board_features(env.core.board)
                mask = env.action_masks()
                legal = np.flatnonzero(mask)
                action, _ = model.predict(observation, deterministic=True,
                                          action_masks=mask)
                action = int(action)
                assert mask[action]
                if sampled < 10000:
                    sampled += 1
                    legal_counts.append(len(legal))
                    ranks.append(int(np.searchsorted(legal, action)) / max(1, len(legal) - 1))
                    placement = next(p for p in env.core.get_legal_placements()
                                     if p["actionId"] == action)
                    placements["hold" if placement["hold"] else "no_hold"] += 1
                    rotations[placement["rotation"]] += 1
                    xs[placement["x"]] += 1
                    ys.append(placement["y"])
                    if sampled % 20 == 0:
                        with torch.no_grad():
                            tensor, _ = model.policy.obs_to_tensor(observation)
                            distribution = model.policy.get_distribution(tensor, action_masks=mask)
                            entropy = float(distribution.entropy().item())
                        entropies.append(entropy)
                        normalized_entropies.append(entropy / math.log(len(legal))
                                                    if len(legal) > 1 else 0)
                observation, reward, terminated, truncated, info = env.step(action)
                cleared = info["cleared_lines"]
                line_clear_sizes[cleared] += 1
                if cleared and first is None:
                    first = info["pieces"]
                after = board_features(env.core.board)
                if after["holes"] > before["holes"]:
                    created_holes += 1
                    last_hole = info["pieces"]
                    if first_hole is None:
                        first_hole = info["pieces"]
                if after["holes"] >= 10 and ten_holes is None:
                    ten_holes = info["pieces"]
                done = terminated or truncated
            if first is not None:
                first_clear.append(first)
            if terminated:
                death.append(after)
                if first_hole is not None:
                    first_hole_to_death.append(info["pieces"] - first_hole)
                if ten_holes is not None:
                    ten_holes_to_death.append(info["pieces"] - ten_holes)
                if last_hole is not None:
                    last_hole_to_death.append(info["pieces"] - last_hole)
            rows.append({"seed": int(seed), "pieces_survived": info["pieces"],
                         "lines": info["lines"], "score": info["score"],
                         "episode_reward": env.episode_reward,
                         "game_over": bool(terminated), "survived_cap": bool(truncated)})
    finally:
        env.close()
    pieces = [r["pieces_survived"] for r in rows]
    lines = [r["lines"] for r in rows]
    totals = {"placement": sum(pieces) * 0.001,
              "single": line_clear_sizes[1] * 1,
              "double": line_clear_sizes[2] * 3,
              "triple": line_clear_sizes[3] * 5,
              "tetris": line_clear_sizes[4] * 8,
              "game_over": -2 * sum(r["game_over"] for r in rows)}
    assert abs(sum(totals.values()) - sum(r["episode_reward"] for r in rows)) < 1e-5
    result = {"label": label, "checkpoint": str(model_path),
              "num_seeds": len(seeds), "max_pieces": 50000,
              "per_seed": rows, "aggregate": aggregate(rows, 50000),
              "p25_pieces": quantile(pieces, .25),
              "p75_pieces": quantile(pieces, .75),
              "std_pieces": stats.pstdev(pieces),
              "zero_line_rate": sum(x == 0 for x in lines) / len(lines),
              "one_plus_line_rate": sum(x >= 1 for x in lines) / len(lines),
              "ten_plus_line_rate": sum(x >= 10 for x in lines) / len(lines),
              "hundred_plus_line_rate": sum(x >= 100 for x in lines) / len(lines),
              "lines_per_100_pieces": 100 * sum(lines) / sum(pieces),
              "first_clear_mean_pieces_among_clearers": stats.mean(first_clear) if first_clear else None,
              "first_clear_coverage": len(first_clear) / len(rows),
              "first_hole_to_death_median": stats.median(first_hole_to_death) if first_hole_to_death else None,
              "ten_holes_to_death_median": stats.median(ten_holes_to_death) if ten_holes_to_death else None,
              "ten_holes_coverage": len(ten_holes_to_death) / len(death) if death else None,
              "death_board_mean": {key: stats.mean(d[key] for d in death)
                                   for key in death[0]} if death else {},
              "death_board_median": {key: stats.median(d[key] for d in death)
                                     for key in death[0]} if death else {},
              "death_board_ge10_holes": sum(d["holes"] >= 10 for d in death) / len(death) if death else None,
              "last_hole_increase_to_death_median": stats.median(last_hole_to_death) if last_hole_to_death else None,
              "decisions_with_hole_increase": created_holes,
              "sampled_decisions": sampled, "hold_rate": placements["hold"] / sampled,
              "rotation_counts": dict(rotations), "x_counts": dict(xs),
              "y_mean": stats.mean(ys), "y_median": stats.median(ys),
              "legal_mean": stats.mean(legal_counts), "legal_median": stats.median(legal_counts),
              "legal_p90": quantile(legal_counts, .9),
              "chosen_mask_rank_mean": stats.mean(ranks),
              "policy_entropy_mean": stats.mean(entropies),
              "policy_entropy_normalized_mean": stats.mean(normalized_entropies),
              "reward_components": totals,
              "line_clear_counts": {str(k): line_clear_sizes[k] for k in range(5)}}
    return result


def load_validation():
    with (RUN / "evaluations/validation_summary.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["num_seeds"]) == 16]
    rows.sort(key=lambda r: int(r["committed_steps"]))
    return rows


def evaluate_all():
    audit_run()
    seeds = json.loads(SEEDS.read_text())["final_test"][:100]
    checkpoints = {
        "step0": (0, RUN / "baseline/model.zip"),
        "1m": (1003520, RUN / "milestones/step_001003520/model.zip"),
        "2m": (2002944, RUN / "milestones/step_002002944/model.zip"),
        "5m": (5001216, RUN / "milestones/step_005001216/model.zip"),
        "best": (9502720, RUN / "best/model.zip"),
        "final": (10002432, RUN / "committed/task_002442/model.zip"),
    }
    folder = OUT / "final"
    folder.mkdir(parents=True, exist_ok=True)
    for label, (step, model) in checkpoints.items():
        destination = folder / f"{label}.json"
        if destination.exists():
            result = json.loads(destination.read_text())
        elif label == "best":
            result = json.loads((RUN / "evaluations/final/step_009502720.json").read_text())
            atomic_json(destination, result)
        else:
            result = evaluate_model(model, seeds, 50000, "final_test", step)
            atomic_json(destination, result)
        assert result["committed_steps"] == step
        assert result["protocol"] == "final_test" and result["max_pieces"] == 50000
        assert result["num_seeds"] == len(result["per_seed"]) == 100
        assert result["seeds"] == seeds
        print(label, result["aggregate"]["mean_pieces"], flush=True)


def plot_line(path, rows, key, title):
    image = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 50, 1060, 585
    draw.line([(left, top), (left, bottom), (right, bottom)], fill="black", width=2)
    xs = [float(row["committed_steps"]) / 1_000_000 for row in rows]
    ys = [float(row[key]) for row in rows]
    ymax = max(ys) * 1.08 if max(ys) > 0 else 1
    ymin = min(0, min(ys) * 1.08)
    points = [(left + int(x / 10.002432 * (right - left)),
               bottom - int((y - ymin) / (ymax - ymin) * (bottom - top)))
              for x, y in zip(xs, ys)]
    draw.line(points, fill="black", width=2)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="black")
    for step in range(0, 11, 2):
        x = left + int(step / 10.002432 * (right - left))
        draw.text((x - 8, bottom + 8), str(step), fill="black")
    for fraction in (0, .25, .5, .75, 1):
        value = ymin + fraction * (ymax - ymin)
        y = bottom - int(fraction * (bottom - top))
        draw.text((10, y - 7), f"{value:.1f}", fill="black")
    draw.text((left, 15), title, fill="black")
    draw.text((right - 150, bottom + 30), "Million steps", fill="black")
    image.save(path)


def build_plots():
    REPORT.mkdir(parents=True, exist_ok=True)
    figures = REPORT / "ppo_raw_10m_seed42_figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = load_validation()
    fields = ("committed_steps", "mean_pieces", "median_pieces", "mean_lines",
              "mean_score", "mean_reward", "survival_cap_rate")
    with (REPORT / "ppo_raw_10m_seed42_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in validation)
    for name, key, title in (
        ("training_curve_mean_pieces.png", "mean_pieces", "Validation mean pieces (16 fixed seeds)"),
        ("training_curve_mean_lines.png", "mean_lines", "Validation mean lines (16 fixed seeds)"),
        ("training_curve_mean_score.png", "mean_score", "Validation mean score (16 fixed seeds)"),
        ("training_curve_reward.png", "mean_reward", "Validation mean reward (16 fixed seeds)"),
    ):
        plot_line(figures / name, validation, key, title)
    with (RUN / "training_metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 2442
    entropy_rows = [{"committed_steps": row["end_step"], "entropy": row["entropy"]}
                    for row in metrics]
    plot_line(figures / "ppo_entropy.png", entropy_rows, "entropy", "Training policy entropy")

    labels = ("step0", "1m", "2m", "5m", "best", "final")
    results = {}
    for label in labels:
        path = (RUN / "evaluations/final/step_009502720.json" if label == "best"
                else OUT / "final" / f"{label}.json")
        results[label] = json.loads(path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ("seed", "checkpoint", "training_steps", "pieces_survived", "lines",
              "score", "episode_reward", "game_over", "survived_cap")
    with tempfile.NamedTemporaryFile("w", newline="", dir=OUT, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label, result in results.items():
            assert result["num_seeds"] == len(result["per_seed"]) == 100
            assert result["max_pieces"] == 50000
            for row in result["per_seed"]:
                writer.writerow({"seed": row["seed"], "checkpoint": label,
                                 "training_steps": result["committed_steps"],
                                 **{key: row[key] for key in fields[3:]}})
        pending = Path(stream.name)
    pending.replace(OUT / "final_test_per_seed.csv")
    image = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 50, 1060, 585
    draw.line([(left, top), (left, bottom), (right, bottom)], fill="black", width=2)
    ymax = max(result["aggregate"]["mean_pieces"] for result in results.values()) * 1.15
    width = (right - left) / len(labels)
    for index, label in enumerate(labels):
        value = results[label]["aggregate"]["mean_pieces"]
        x = int(left + index * width + width * .25)
        x2 = int(x + width * .5)
        y = bottom - int(value / ymax * (bottom - top))
        draw.rectangle((x, y, x2, bottom), outline="black")
        draw.text((x, y - 18), f"{value:.1f}", fill="black")
        draw.text((x, bottom + 10), label, fill="black")
    draw.text((left, 15), "Final test mean pieces (100 fixed seeds)", fill="black")
    image.save(figures / "final_test_checkpoint_comparison.png")
    print(figures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", choices=("best", "final"))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--evaluate-all", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_run(), indent=2))
        return
    if args.evaluate_all:
        evaluate_all()
        return
    if args.plots:
        build_plots()
        return
    if args.trace:
        OUT.mkdir(parents=True, exist_ok=True)
        seeds = json.loads(SEEDS.read_text())["final_test"][:100]
        model = (RUN / "best/model.zip" if args.trace == "best"
                 else RUN / "committed/task_002442/model.zip")
        result = trace_model(args.trace, model, seeds)
        atomic_json(OUT / f"trace_{args.trace}.json", result)
        print(args.trace, result["aggregate"], flush=True)
        return
    parser.error("specify --audit, --evaluate-all, --trace or --plots")


if __name__ == "__main__":
    main()
