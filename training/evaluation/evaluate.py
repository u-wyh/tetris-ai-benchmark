"""Evaluate a committed model in an independent process; never call learn()."""

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import statistics
import time
from pathlib import Path

from gymnasium import spaces

from training.env import TetrisEnv
from training.train_ppo import atomic_json, now
from training.evaluation.worker_budget import automatic_workers, detect_resources
from training.evaluation.feature_backend import (BACKENDS, backend_name, inference_source_identity,
                                                 evaluation_runtime_identity)


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


_WORKER_MODEL = None
_WORKER_BACKEND = 'official'


def _init_worker(model_path, candidate_feature_backend='official'):
    global _WORKER_MODEL, _WORKER_BACKEND
    from sb3_contrib import MaskablePPO
    import torch
    torch.set_num_threads(1)
    _WORKER_BACKEND = backend_name(candidate_feature_backend)
    _WORKER_MODEL = MaskablePPO.load(str(model_path), device="cpu")


def _evaluate_seed(model, seed, max_pieces, diagnostics, candidate_feature_backend='official'):
    if model is None:
        raise RuntimeError("Evaluation worker has no model")
    from training.env.fast_candidate_env import FastCandidateTetrisEnv
    env_class = (FastCandidateTetrisEnv if isinstance(getattr(model, "observation_space", None), spaces.Dict)
                 else TetrisEnv)
    if backend_name(candidate_feature_backend) == 'bitboard':
        if env_class is TetrisEnv:
            raise ValueError('Bitboard evaluation requires a Candidate model')
        from training.env.bitboard_candidate_env import BitboardCandidateTetrisEnv
        env_class = BitboardCandidateTetrisEnv
    env = env_class(max_pieces=max_pieces)
    try:
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
        return row
    finally:
        env.close()


def _evaluate_worker(seed, max_pieces, diagnostics):
    return _evaluate_seed(_WORKER_MODEL, seed, max_pieces, diagnostics, _WORKER_BACKEND)


def evaluate_model(model_path, seeds, max_pieces, protocol, committed_steps,
                   diagnostics=False, workers=1, candidate_feature_backend='official'):
    candidate_feature_backend = backend_name(candidate_feature_backend)
    if workers < 1:
        raise ValueError("Evaluation workers must be positive")
    began = time.monotonic()
    if workers == 1:
        _init_worker(model_path, candidate_feature_backend)
        rows = [_evaluate_worker(seed, max_pieces, diagnostics) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds)),
                                 mp_context=get_context("spawn"),
                                 initializer=_init_worker, initargs=(model_path,candidate_feature_backend)) as pool:
            # executor.map preserves the frozen seed order in the result file.
            rows = list(pool.map(_evaluate_worker, seeds,
                                 [max_pieces] * len(seeds), [diagnostics] * len(seeds)))
    return {"protocol": protocol, "committed_steps": committed_steps,
            "candidate_feature_backend": candidate_feature_backend,
            **evaluation_runtime_identity(diagnostics),
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
    parser.add_argument("--workers", type=int, default=0,
                        help="0 selects up to eight workers within affinity, CPU quota and memory headroom")
    parser.add_argument('--candidate-feature-backend', choices=BACKENDS, default='official')
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    seeds = json.loads(args.seeds_file.read_text())[args.seed_set][:args.seed_count]
    if len(seeds) != args.seed_count or len(set(seeds)) != len(seeds):
        raise ValueError("Seed count or uniqueness mismatch")
    budget = detect_resources() if args.workers == 0 else None
    workers = automatic_workers(len(seeds), budget) if args.workers == 0 else args.workers
    result = evaluate_model(args.model, seeds, args.max_pieces,
                            args.protocol, args.committed_steps, workers=workers,
                            diagnostics=args.diagnostics,
                            candidate_feature_backend=args.candidate_feature_backend)
    result["evaluation_workers"] = workers
    result["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    result["evaluation_source_sha256"] = inference_source_identity()
    if budget is not None:
        result["automatic_worker_budget"] = budget
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
