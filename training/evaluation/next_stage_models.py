"""Resumable masked Candidate comparisons on a fixed, shared seed/cap protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
from pathlib import Path
import statistics
import traceback

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.champion_pipeline import evaluate_seed
from training.imitation_dataset import digest
from training.long_run import SEEDS_FILE
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/candidate_bc_next_stage"


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
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    seeds = (json.loads(SEEDS_FILE.read_text())["validation"] if args.seed_set == "validation"
             else list(range(1260000, 1260008)))
    evaluate(args.label, args.model, seeds, args.cap, args.workers)


if __name__ == "__main__":
    main()
