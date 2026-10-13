"""Resumable masked Candidate comparisons on a fixed, shared seed/cap protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.champion_pipeline import evaluate_seed
from training.imitation_dataset import digest
from training.long_run import SEEDS_FILE
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/candidate_bc_next_stage"
WORKER_TUNING = RUN / "long_worker_tuning.json"


def probe_one(model_path, seed):
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(model_path), device="cpu")
    return evaluate_seed(model, seed, 3000)["pieces_survived"]


def choose_workers(timings):
    return min(timings, key=lambda count: (timings[count], count))


def long_workers():
    """Time independent short games once, after training, on this host."""
    logical = len(os.sched_getaffinity(0))
    candidates = [n for n in (4, 6, 8, 10, 12) if n <= logical]
    if not candidates:
        return max(1, logical)
    if WORKER_TUNING.exists():
        record = json.loads(WORKER_TUNING.read_text())
        if (record.get("version") == 1 and record.get("logical_cpus") == logical
                and record.get("candidates") == candidates
                and record.get("selected") in candidates):
            return record["selected"]
        raise RuntimeError("Long-run worker tuning differs from the current host or protocol")
    seeds = list(range(1270000, 1270012))
    model_path = RUN / "champion/model.zip"
    timings = {}
    for count in candidates:
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as pool:
            pieces = list(pool.map(probe_one, [model_path] * len(seeds), seeds))
        if pieces != [3000] * len(seeds):
            raise RuntimeError("BC throughput pilot did not reach its fixed cap")
        timings[count] = time.monotonic() - start
    selected = choose_workers(timings)
    atomic_json(WORKER_TUNING, {"version": 1, "logical_cpus": logical,
                "candidates": candidates,
                "pilot_model": str(model_path), "pilot_seeds": seeds,
                "pilot_max_pieces": 3000, "wall_seconds": timings,
                "selection_rule": "shortest batch wall time; lower workers on tie",
                "selected": selected, "selected_at": now()})
    return selected


def one(model_path, seed, cap, output):
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(model_path), device="cpu")
    row = evaluate_seed(model, seed, cap)
    atomic_json(output / f"seed_{seed}.json", row)
    return seed, row["pieces_survived"]


def evaluate(label, model_path, seeds, cap, workers=6):
    model_path, seeds = Path(model_path), list(seeds)
    if len(seeds) != len(set(seeds)) or not seeds or cap < 1:
        raise ValueError("Invalid comparison protocol")
    output = RUN / "comparisons" / label
    output.mkdir(parents=True, exist_ok=True)
    expected = {"label": label, "model": str(model_path.resolve()),
                "model_sha256": digest(model_path), "seeds": seeds,
                "max_pieces": cap, "deterministic": True, "action_mask": True}
    config = output / "config.json"
    if config.exists() and json.loads(config.read_text()) != expected:
        raise RuntimeError("Comparison protocol changed")
    if not config.exists():
        atomic_json(config, expected)
    pending = []
    for seed in seeds:
        path = output / f"seed_{seed}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["seed"] != seed or row["pieces_survived"] > cap:
                raise RuntimeError(f"Invalid saved comparison: {path}")
        else:
            pending.append(seed)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(one, model_path, seed, cap, output) for seed in pending]
            for future in as_completed(futures):
                seed, pieces = future.result()
                atomic_json(output / "status.json", {"state": "running", "last_seed": seed,
                            "completed": len(seeds) - sum(not (output / f"seed_{s}.json").exists()
                                                          for s in seeds), "total": len(seeds),
                            "updated_at": now()})
                print(f"{label}: seed={seed}, pieces={pieces}", flush=True)
        rows = [json.loads((output / f"seed_{seed}.json").read_text()) for seed in seeds]
        pieces = sorted(row["pieces_survived"] for row in rows)
        result = {"mean_pieces": statistics.mean(pieces), "median_pieces": statistics.median(pieces),
                  "p10_pieces": pieces[max(0, math.ceil(.1 * len(pieces)) - 1)],
                  "capped_count": sum(p == cap for p in pieces),
                  "mean_lines": statistics.mean(row["lines"] for row in rows),
                  "mean_score": statistics.mean(row["score"] for row in rows),
                  "mean_new_holes": statistics.mean(row["new_holes_total"] for row in rows),
                  "new_holes_per_100_pieces": (100 * sum(row["new_holes_total"] for row in rows)
                                               / sum(row["pieces_survived"] for row in rows)),
                  "mean_holes_at_end": statistics.mean(row["holes_at_end"] for row in rows),
                  "mean_height_at_end": statistics.mean(row["height_at_end"] for row in rows),
                  "max_pieces": cap, "seeds": seeds, "completed_at": now()}
        atomic_json(output / "summary.json", result)
        atomic_json(output / "status.json", {"state": "completed", "completed": len(seeds),
                    "total": len(seeds), "updated_at": now()})
        return result
    except Exception:
        atomic_json(output / "status.json", {"state": "failed", "error": traceback.format_exc(),
                    "updated_at": now()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=20000)
    parser.add_argument("--seed-set", choices=["validation", "tuning"], default="validation")
    parser.add_argument("--workers", type=int, default=0,
                        help="0 uses a measured worker count for 100k games, else six")
    args = parser.parse_args()
    seeds = (json.loads(SEEDS_FILE.read_text())["validation"] if args.seed_set == "validation"
             else list(range(1260000, 1260008)))
    workers = (long_workers() if args.cap >= 100000 else 6) if args.workers == 0 else args.workers
    evaluate(args.label, args.model, seeds, args.cap, workers)


if __name__ == "__main__":
    main()
