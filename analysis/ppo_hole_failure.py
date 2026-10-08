"""Read-only, paired placement traces for completed Raw/Shaped 2M checkpoints."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from analysis.ppo_raw_diagnostics import board_features, sha256
from training.env import TetrisEnv
from training.tetris_core import decode_action
from training.train_ppo import ROOT, atomic_json

SEEDS_FILE = ROOT / "training/evaluation/seeds.json"
PAIRED_FILE = ROOT / "runs/ppo_hole_v1_2m_seed42_lambda010/paired_validation_2m.json"
MODELS = {
    "raw": ROOT / "runs/ppo_raw_10m_seed42/milestones/step_002002944/model.zip",
    "shaped": ROOT / "runs/ppo_hole_v1_2m_seed42_lambda010/final/model.zip",
}
FEATURES = ("aggregate_height", "max_height", "holes", "bumpiness", "covered_holes")
CAP = 5000


def visible_board(board):
    return ["".join(cell if cell is not None else "." for cell in row)
            for row in board]


def column_heights(rows):
    return [20 - next((y for y, row in enumerate(rows) if row[x] != "."), 20)
            for x in range(10)]


def trace_seed(model, env, seed, label):
    observation, _ = env.reset(seed=seed)
    steps = []
    while True:
        before = board_features(env.core.board)
        current = env.core.current["type"]
        held = env.core.hold
        next_piece = env.core.queue[0]
        mask = env.action_masks()
        action, _ = model.predict(observation, deterministic=True, action_masks=mask)
        action = int(action)
        if not mask[action]:
            raise AssertionError("Model selected an illegal action")
        placement = decode_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        after = board_features(env.core.board)
        if (info["holes_before"] != before["holes"]
                or info["holes_after"] != after["holes"]):
            raise AssertionError("Reward and board-feature hole definitions disagree")
        steps.append({
            "piece_index": info["pieces"], "action_id": action,
            "current_piece": current, "placed_piece": (
                held if held is not None else next_piece) if placement["hold"] else current,
            "hold_used": bool(placement["hold"]), "placement": placement,
            "cleared_lines": info["cleared_lines"], "lines_total": info["lines"],
            "before": {key: before[key] for key in FEATURES},
            "after": {key: after[key] for key in FEATURES},
            "new_holes": info["new_holes"], "score": info["score"],
            "raw_reward": info["raw_reward"],
            "shaped_reward": info["shaped_reward"] if label == "shaped" else None,
            "hole_penalty": info["hole_penalty"] if label == "shaped" else None,
            "game_over": bool(terminated), "survived_cap": bool(truncated),
            "board_after": visible_board(env.core.board),
        })
        if terminated or truncated:
            break
    raw_total = sum(step["raw_reward"] for step in steps)
    created = sum(step["new_holes"] for step in steps)
    penalty = 0.10 * created
    if label == "shaped" and abs(env.episode_reward - (raw_total - penalty)) > 1e-8:
        raise AssertionError("Shaped episode return does not match step diagnostics")
    if label == "raw" and abs(env.episode_reward - raw_total) > 1e-8:
        raise AssertionError("Raw episode return does not match step diagnostics")
    first_height = {str(height): next((step["piece_index"] for step in steps
                                      if step["after"]["max_height"] >= height), None)
                    for height in (15, 18, 19)}
    return {"seed": seed, "label": label, "pieces_survived": len(steps),
            "lines": info["lines"], "score": info["score"],
            "game_over": bool(terminated), "survived_cap": bool(truncated),
            "raw_reward_total": raw_total,
            "hole_penalty_at_lambda_010": penalty,
            "shaped_reward_at_lambda_010": raw_total - penalty,
            "actual_episode_reward": env.episode_reward,
            "first_max_height": first_height,
            "steps": steps}


def summary(episodes):
    steps = [step for episode in episodes for step in episode["steps"]]
    deaths = [episode for episode in episodes if episode["game_over"]]
    if not episodes or not steps:
        raise ValueError("Need at least one nonempty episode")
    lines = Counter(step["cleared_lines"] for step in steps)
    high_risk = [step for step in steps if step["before"]["max_height"] >= 15]
    death_heights = [column_heights(e["steps"][-1]["board_after"]) for e in deaths]
    def mean(values):
        return statistics.mean(values) if values else None
    def rate(rows, condition):
        return sum(condition(step) for step in rows) / len(rows) if rows else None
    result = {
        "num_seeds": len(episodes), "pieces_total": len(steps),
        "mean_pieces": mean([episode["pieces_survived"] for episode in episodes]),
        "median_pieces": statistics.median(e["pieces_survived"] for e in episodes),
        "game_over_rate": len(deaths) / len(episodes),
        "survival_cap_rate": 1 - len(deaths) / len(episodes),
        "mean_lines": mean([e["lines"] for e in episodes]),
        "mean_score": mean([e["score"] for e in episodes]),
        "lines_per_100_pieces": 100 * sum(e["lines"] for e in episodes) / len(steps),
        "new_holes_total": sum(s["new_holes"] for s in steps),
        "new_holes_events": sum(s["new_holes"] > 0 for s in steps),
        "new_holes_per_100_pieces": 100 * sum(s["new_holes"] for s in steps) / len(steps),
        "mean_raw_reward_total": mean([e["raw_reward_total"] for e in episodes]),
        "mean_hole_penalty_at_lambda_010": mean([
            e["hole_penalty_at_lambda_010"] for e in episodes]),
        "mean_shaped_reward_at_lambda_010": mean([
            e["shaped_reward_at_lambda_010"] for e in episodes]),
        "hold_rate": rate(steps, lambda s: s["hold_used"]),
        "placement_x_counts": {str(x): sum(s["placement"]["x"] == x for s in steps)
                               for x in range(10)},
        "clear_counts": {str(n): lines[n] for n in range(5)},
        "zero_new_holes_rate": rate(steps, lambda s: s["new_holes"] == 0),
        "height_increase_without_new_holes_rate": rate(
            steps, lambda s: s["new_holes"] == 0 and
            s["after"]["aggregate_height"] > s["before"]["aggregate_height"]),
        "mean_aggregate_height_after": mean([s["after"]["aggregate_height"] for s in steps]),
        "mean_max_height_after": mean([s["after"]["max_height"] for s in steps]),
        "mean_delta_aggregate_height": mean([
            s["after"]["aggregate_height"] - s["before"]["aggregate_height"] for s in steps]),
        "mean_delta_max_height": mean([
            s["after"]["max_height"] - s["before"]["max_height"] for s in steps]),
        "high_risk_preheight_15": {
            "actions": len(high_risk), "rate": len(high_risk) / len(steps),
            "hold_rate": rate(high_risk, lambda s: s["hold_used"]),
            "zero_new_holes_rate": rate(high_risk, lambda s: s["new_holes"] == 0),
            "height_increase_without_new_holes_rate": rate(
                high_risk, lambda s: s["new_holes"] == 0 and
                s["after"]["aggregate_height"] > s["before"]["aggregate_height"]),
            "mean_new_holes": mean([s["new_holes"] for s in high_risk]),
            "clear_rate": rate(high_risk, lambda s: s["cleared_lines"] > 0),
            "placement_x_counts": {str(x): sum(s["placement"]["x"] == x for s in high_risk)
                                   for x in range(10)}},
        "first_max_height": {
            str(h): {"reached": sum(e["first_max_height"][str(h)] is not None for e in episodes),
                     "mean_step_among_reached": mean([
                         e["first_max_height"][str(h)] for e in episodes
                         if e["first_max_height"][str(h)] is not None]),
                     "mean_remaining_pieces_among_reached": mean([
                         e["pieces_survived"] - e["first_max_height"][str(h)] for e in episodes
                         if e["first_max_height"][str(h)] is not None])}
            for h in (15, 18, 19)},
        "death_board_mean": {key: mean([e["steps"][-1]["after"][key] for e in deaths])
                             for key in FEATURES},
        "death_board_ge_15": sum(e["steps"][-1]["after"]["max_height"] >= 15
                                 for e in deaths) / len(deaths) if deaths else None,
        "death_board_ge_19": sum(e["steps"][-1]["after"]["max_height"] >= 19
                                 for e in deaths) / len(deaths) if deaths else None,
        "death_mean_column_heights": [mean([row[x] for row in death_heights])
                                      for x in range(10)],
        "death_max_column_counts": {str(x): sum(max(range(10), key=row.__getitem__) == x
                                           for row in death_heights) for x in range(10)},
        "death_mean_max_height_before_final_action": mean([
            e["steps"][-1]["before"]["max_height"] for e in deaths]),
    }
    for window in (10, 20):
        tails = [e["steps"][-window:] for e in episodes if len(e["steps"]) >= window]
        result[f"last_{window}"] = {
            "episodes": len(tails),
            "first_mean_aggregate_height": mean([t[0]["after"]["aggregate_height"] for t in tails]),
            "last_mean_aggregate_height": mean([t[-1]["after"]["aggregate_height"] for t in tails]),
            "first_mean_max_height": mean([t[0]["after"]["max_height"] for t in tails]),
            "last_mean_max_height": mean([t[-1]["after"]["max_height"] for t in tails]),
            "first_mean_holes": mean([t[0]["after"]["holes"] for t in tails]),
            "last_mean_holes": mean([t[-1]["after"]["holes"] for t in tails]),
        }
    return result


def curve_by_piece(episodes):
    length = max(len(e["steps"]) for e in episodes)
    return [{"piece_index": index, "survivors": len(rows),
             **{key: statistics.mean(row["after"][key] for row in rows)
                for key in FEATURES}}
            for index in range(1, length + 1)
            if (rows := [e["steps"][index - 1] for e in episodes
                         if len(e["steps"]) >= index])]


def curve_before_death(episodes, window=20):
    result = []
    for offset in range(-window + 1, 1):
        rows = [e["steps"][len(e["steps"]) + offset - 1]
                for e in episodes if e["game_over"] and len(e["steps"]) + offset - 1 >= 0]
        result.append({"steps_to_death": -offset, "episodes": len(rows),
                       **{key: statistics.mean(row["after"][key] for row in rows)
                          if rows else None for key in FEATURES}})
    return result


def audit_against_paired(label, episodes, paired):
    expected = paired[label]["per_seed"]
    if [e["seed"] for e in episodes] != [r["seed"] for r in expected]:
        raise AssertionError("Trace seed order differs from paired evaluation")
    for episode, row in zip(episodes, expected):
        if any(episode[field] != row[field] for field in ("pieces_survived", "lines", "score", "game_over")):
            raise AssertionError(f"Trace differs from prior paired result: {label} {episode['seed']}")
        if sum(step["new_holes"] for step in episode["steps"]) != row["new_holes_total"]:
            raise AssertionError("Trace new-hole total differs from prior paired result")


def plot_curves(output, episodes, curves):
    from PIL import Image, ImageDraw

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"raw": "#2266aa", "shaped": "#cc5522"}

    def draw_panel(draw, box, data, title, y_floor=None, y_ceiling=None):
        x0, y0, x1, y1 = box
        draw.text((x0, y0 - 17), title, fill="black")
        draw.line([(x0, y0), (x0, y1), (x1, y1)], fill="#777777", width=2)
        values = [value for points in data.values() for _, value in points if value is not None]
        xs = [x for points in data.values() for x, value in points if value is not None]
        if not values or not xs:
            return
        low = min(values) if y_floor is None else y_floor
        high = max(values) if y_ceiling is None else y_ceiling
        if high <= low:
            high = low + 1
        first, last = min(xs), max(xs)
        if last <= first:
            last = first + 1
        for label, points in data.items():
            coords = [(int(x0 + (x - first) / (last - first) * (x1 - x0)),
                       int(y1 - (value - low) / (high - low) * (y1 - y0)))
                      for x, value in points if value is not None]
            if len(coords) > 1:
                draw.line(coords, fill=colors[label], width=3)
            elif coords:
                draw.ellipse((coords[0][0] - 2, coords[0][1] - 2,
                              coords[0][0] + 2, coords[0][1] + 2), fill=colors[label])
        draw.text((x0, y1 + 4), str(first), fill="black")
        draw.text((x1 - 35, y1 + 4), str(last), fill="black")
        draw.text((x0 + 4, y0 + 3), f"{high:.1f}", fill="#555555")
        draw.text((x0 + 4, y1 - 16), f"{low:.1f}", fill="#555555")

    for key, title, filename in (
        ("aggregate_height", "Aggregate height", "aggregate_height.png"),
        ("max_height", "Maximum column height", "max_height.png"),
        ("holes", "Board holes", "holes.png"),
    ):
        image = Image.new("RGB", (1000, 680), "white")
        draw = ImageDraw.Draw(image)
        draw.text((70, 20), title + " by placement (survivors only)", fill="black")
        draw.text((70, 44), "Raw = blue; Shaped = orange. Lower panel shows surviving sample count.",
                  fill="#555555")
        draw_panel(draw, (70, 105, 950, 445), {
            label: [(r["piece_index"], r[key]) for r in curves[label]["by_piece"]]
            for label in colors}, "Mean " + title.lower())
        draw_panel(draw, (70, 520, 950, 635), {
            label: [(r["piece_index"], r["survivors"])
                    for r in curves[label]["by_piece"]] for label in colors},
            "Surviving episodes at each placement", 0, 32)
        image.save(figures / filename)

    image = Image.new("RGB", (1000, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 20), "Last 20 placements before Game Over", fill="black")
    draw.text((70, 44), "Raw = blue; Shaped = orange; 0 = final placement.", fill="#555555")
    for index, (key, title) in enumerate((
        ("aggregate_height", "Aggregate height"), ("max_height", "Maximum height"),
        ("holes", "Board holes"), ("episodes", "Episodes contributing"))):
        draw_panel(draw, (70, 120 + 250 * index, 950, 295 + 250 * index), {
            label: [(-r["steps_to_death"], r[key])
                    for r in curves[label]["before_death"]] for label in colors},
            title, 0 if key == "episodes" else None,
            32 if key == "episodes" else None)
    image.save(figures / "before_death_20.png")

    image = Image.new("RGB", (840, 640), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 20), "Final visible-board occupancy (fraction of 32 deaths)", fill="black")
    for index, label in enumerate(("raw", "shaped")):
        finished = [e["steps"][-1]["board_after"] for e in episodes[label] if e["game_over"]]
        density = np.mean([[[cell != "." for cell in row] for row in board]
                           for board in finished], axis=0)
        origin_x = 60 + index * 400
        draw.text((origin_x, 65), label, fill=colors[label])
        for y in range(20):
            for x in range(10):
                fraction = float(density[y, x])
                color = (int(255 * fraction), int(110 * (1 - fraction)),
                         int(180 * (1 - fraction)))
                x0, y0 = origin_x + 28 * x, 95 + 24 * y
                draw.rectangle((x0, y0, x0 + 26, y0 + 22), fill=color)
        draw.text((origin_x, 590), f"n={len(finished)}; darker/purple = less occupied",
                  fill="#555555")
    image.save(figures / "death_board_occupancy.png")


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Analysis output already exists; refusing to overwrite traces")
    seeds = json.loads(SEEDS_FILE.read_text())["validation"]
    if seeds != list(range(100000, 100032)):
        raise ValueError("Expected the 32 fixed validation seeds")
    paired = json.loads(PAIRED_FILE.read_text())
    if paired["raw"]["seeds"] != seeds or paired["shaped"]["seeds"] != seeds:
        raise ValueError("Existing paired result uses different seeds")
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    all_episodes = {}
    curves = {}
    results = {"seeds": seeds, "max_pieces": CAP, "models": {},
               "checkpoint_sha256": {key: sha256(path) for key, path in MODELS.items()}}
    for label, path in MODELS.items():
        model = MaskablePPO.load(str(path), device="cpu")
        env = TetrisEnv(max_pieces=CAP, hole_penalty_coef=0.1 if label == "shaped" else 0.0)
        episodes = []
        try:
            for seed in seeds:
                episode = trace_seed(model, env, seed, label)
                atomic_json(output / f"{label}_seed_{seed}.json", episode)
                episodes.append(episode)
        finally:
            env.close()
        audit_against_paired(label, episodes, paired)
        all_episodes[label] = episodes
        curves[label] = {"by_piece": curve_by_piece(episodes),
                         "before_death": curve_before_death(episodes)}
        results["models"][label] = summary(episodes)
    atomic_json(output / "curves.json", curves)
    atomic_json(output / "summary.json", results)
    plot_curves(output, all_episodes, curves)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/ppo_hole_v1_failure_analysis")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate figures from saved trace and curve data")
    parser.add_argument("--from-traces", action="store_true",
                        help="Rebuild summary and figures from saved per-seed traces")
    args = parser.parse_args()
    if args.plots_only or args.from_traces:
        seeds = json.loads((args.output / "summary.json").read_text())["seeds"]
        episodes = {label: [json.loads((args.output / f"{label}_seed_{seed}.json").read_text())
                            for seed in seeds] for label in MODELS}
        if args.from_traces:
            curves = {label: {"by_piece": curve_by_piece(rows),
                              "before_death": curve_before_death(rows)}
                      for label, rows in episodes.items()}
            atomic_json(args.output / "curves.json", curves)
            result = json.loads((args.output / "summary.json").read_text())
            result["models"] = {label: summary(rows) for label, rows in episodes.items()}
            atomic_json(args.output / "summary.json", result)
        else:
            curves = json.loads((args.output / "curves.json").read_text())
            result = json.loads((args.output / "summary.json").read_text())
        plot_curves(args.output, episodes, curves)
    else:
        result = run(args.output)
    for label, summary_data in result["models"].items():
        print(label, summary_data["mean_pieces"], summary_data["death_board_mean"])


if __name__ == "__main__":
    main()
