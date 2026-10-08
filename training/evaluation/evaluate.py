"""Evaluate a committed model in an independent process; never call learn()."""

import argparse
import json
import statistics
import time
from pathlib import Path

from training.env import TetrisEnv
from training.train_ppo import atomic_json, now


def best_key(result):
    metrics = result["aggregate"] if "aggregate" in result else result
    return (metrics["mean_pieces"], metrics["mean_lines"], metrics["mean_score"])


def aggregate(rows, max_pieces):
    if not rows:
        raise ValueError("Evaluation needs at least one seed")
    fields = (("pieces_survived", "pieces"), ("lines", "lines"),
              ("score", "score"), ("episode_reward", "reward"))
    result = {}
    for source, label in fields:
        values = [row[source] for row in rows]
        result[f"mean_{label}"] = statistics.mean(values)
        if label != "reward":
            result[f"median_{label}"] = statistics.median(values)
    pieces = sorted(row["pieces_survived"] for row in rows)
    result.update({"p90_pieces": pieces[max(0, (9 * len(pieces) + 9) // 10 - 1)],
                   "best_pieces": pieces[-1],
                   "game_over_rate": sum(row["game_over"] for row in rows) / len(rows),
                   "survival_cap_rate": sum(row["survived_cap"] for row in rows) / len(rows)})
    result["evaluation_saturated"] = result["survival_cap_rate"] >= 0.5
    result["metric_censored_by_cap"] = any(row["survived_cap"] for row in rows)
    if all("new_holes_total" in row for row in rows):
        total_pieces = sum(row["pieces_survived"] for row in rows)
        deaths = [row["holes_at_end"] for row in rows if row["game_over"]]
        result.update(new_holes_events=sum(row["new_holes_events"] for row in rows),
                      new_holes_total=sum(row["new_holes_total"] for row in rows),
                      new_holes_per_100_pieces=(100 * sum(row["new_holes_total"] for row in rows)
                                                / total_pieces),
                      mean_holes_at_death=statistics.mean(deaths) if deaths else None)
    return result


def evaluate_model(model_path, seeds, max_pieces, protocol, committed_steps,
                   diagnostics=False):
    from sb3_contrib import MaskablePPO
    import torch

    torch.set_num_threads(1)
    model = MaskablePPO.load(str(model_path), device="cpu")
    began = time.monotonic()
    rows = []
    env = TetrisEnv(max_pieces=max_pieces)
    try:
        for seed in seeds:
            observation, info = env.reset(seed=int(seed))
            ended = False
            hole_events = hole_total = 0
            while not ended:
                action, _ = model.predict(observation, deterministic=True,
                                          action_masks=env.action_masks())
                observation, reward, terminated, truncated, info = env.step(int(action))
                if diagnostics:
                    hole_events += int(info["new_holes"] > 0)
                    hole_total += info["new_holes"]
                ended = terminated or truncated
            row = {"seed": int(seed), "pieces_survived": info["pieces"],
                   "lines": info["lines"], "score": info["score"],
                   "episode_reward": env.episode_reward,
                   "game_over": bool(terminated), "survived_cap": bool(truncated)}
            if diagnostics:
                row.update(new_holes_events=hole_events, new_holes_total=hole_total,
                           holes_at_end=info["holes_after"])
            rows.append(row)
    finally:
        env.close()
    return {"protocol": protocol, "committed_steps": committed_steps,
            "model_checkpoint": str(model_path), "num_seeds": len(seeds),
            "max_pieces": max_pieces, "seeds": list(seeds), "per_seed": rows,
            "aggregate": aggregate(rows, max_pieces),
            "wall_seconds": time.monotonic() - began, "completed_at": now()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds-file", type=Path, required=True)
    parser.add_argument("--seed-set", choices=("validation", "final_test"), required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--max-pieces", type=int, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--committed-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = json.loads(args.seeds_file.read_text())[args.seed_set][:args.seed_count]
    if len(seeds) != args.seed_count or len(set(seeds)) != len(seeds):
        raise ValueError("Seed count or uniqueness mismatch")
    atomic_json(args.output, evaluate_model(args.model, seeds, args.max_pieces,
                                            args.protocol, args.committed_steps))


if __name__ == "__main__":
    main()
