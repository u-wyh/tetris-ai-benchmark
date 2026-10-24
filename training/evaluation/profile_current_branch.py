"""Profile current frozen PPO evaluation on three archived branch snapshots."""

import cProfile
import gzip
import hashlib
import json
import pickle
import pstats
import time

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.fast_counterfactual import fast_branch
from training.evaluation.hybrid_queue import PPO, digest
from training.train_ppo import ROOT, atomic_json, now


CASES = ((1550002, 3875), (1550005, 4900), (1550007, 6075))
MODEL_HASH = "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34"
DATA = ROOT / "runs/ppo_clear_cadence_counterfactual"
OUT = ROOT / "runs/ppo_current_branch_profile_20261024"


def main():
    torch.set_num_threads(1)
    if digest(PPO) != MODEL_HASH:
        raise RuntimeError("Frozen PPO model changed")
    OUT.mkdir(parents=True, exist_ok=True)
    model = MaskablePPO.load(PPO, device="cpu")
    rows = []
    for seed, step in CASES:
        name = f"seed_{seed}_step_{step}"
        snapshot_path = DATA / "states" / f"{name}.pkl.gz"
        candidate_path = DATA / "candidates" / f"{name}.json"
        candidate = json.loads(candidate_path.read_text())
        if digest(snapshot_path) != candidate["snapshot_sha256"]:
            raise RuntimeError(f"Snapshot changed: {name}")
        snapshot = pickle.loads(gzip.decompress(snapshot_path.read_bytes()))
        if (snapshot["episode_seed"], snapshot["pieces"]) != (seed, step):
            raise RuntimeError(f"Snapshot identity changed: {name}")
        action = candidate["original_action"]
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        result = fast_branch(snapshot, action, {"ppo": model}, 500)
        profiler.disable()
        elapsed = time.perf_counter() - started
        archive_path = DATA / "branches" / f"{name}_action_{action}.json"
        archive = json.loads(archive_path.read_text())
        expected = archive["outcome"]
        if result["scores"] != expected["scores"][:500] or result["fours"] != expected["fours"][:500]:
            raise RuntimeError(f"Archived branch prefix differs: {name}")
        profiler.dump_stats(str(OUT / f"{name}.prof"))
        stats = pstats.Stats(profiler)
        functions = {}
        for (path, _line, function), (_cc, calls, own, cumulative, _callers) in stats.stats.items():
            if path.endswith(("fast_legal.py", "fast_candidate_features.py")) and function in (
                "get_legal_placements", "_branch", "candidate_features", "collides", "rotate"
            ):
                functions[function] = {"calls": calls, "own_seconds": own,
                                       "cumulative_seconds": cumulative}
        row = {"seed": seed, "step": step, "snapshot_sha256": digest(snapshot_path),
               "candidate_sha256": digest(candidate_path), "action": action,
               "model_sha256": MODEL_HASH, "horizon": 500,
               "archive_sha256": digest(archive_path), "archived_prefix_equal": True,
               "elapsed_seconds": elapsed, "profile_total_seconds": stats.total_tt,
               "functions": functions,
               "result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
               "completed_at": now()}
        atomic_json(OUT / f"{name}.json", row)
        rows.append(row)
        print(name, round(elapsed, 3), functions, flush=True)
    atomic_json(OUT / "summary.json", {"cases": rows, "completed_at": now()})


if __name__ == "__main__":
    main()
