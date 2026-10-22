"""Paired wall-time benchmark for exact frozen-PPO branch outcomes."""

import gzip
import hashlib
import json
import pickle
import resource
import statistics
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.fast_counterfactual import fast_branch
from training.evaluation.hybrid_queue import PPO, digest
from training.evaluation.ppo_fresh_counterfactual import branch
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/ppo_branch_throughput_benchmark"
DATA = ROOT / "runs/ppo30m_top2_generalization"
SEEDS = (1540000, 1540004, 1541002)
HORIZON = 1500
REPEATS = 3
PPO_SHA256 = "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34"


def source_hash(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True)
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO checkpoint changed")
    cases = []
    for seed in SEEDS:
        manifest_path = DATA / "selected_states" / f"seed_{seed}.json"
        manifest = json.loads(manifest_path.read_text())
        snapshot_path = DATA / "selected_states" / f"seed_{seed}.pkl.gz"
        if digest(snapshot_path) != manifest["sha256"]:
            raise RuntimeError(f"Benchmark snapshot changed: {seed}")
        candidates_path = DATA / "candidates" / f"seed_{seed}.json"
        candidates = json.loads(candidates_path.read_text())["choices"]
        top2 = next(item for item in candidates if "ppo_top2" in item["sources"])
        cases.append({"seed": seed, "step": manifest["step"], "action": top2["action"],
                      "snapshot_sha256": manifest["sha256"], "candidate_sha256": digest(candidates_path)})
    protocol = {"version": "fast_counterfactual_benchmark_v1", "cases": cases,
                "horizon": HORIZON, "repeats": REPEATS, "warmup_horizon": 100,
                "device": "cpu", "torch_threads": 1, "ppo_sha256": PPO_SHA256,
                "source_sha256": {relative: source_hash(relative) for relative in (
                    "training/evaluation/fast_counterfactual.py",
                    "training/evaluation/ppo_fresh_counterfactual.py",
                    "training/evaluation/fast_legal.py",
                    "training/evaluation/fast_candidate_features.py",
                    "training/evaluation/benchmark_fast_counterfactual.py")}}
    frozen = RUN / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Benchmark protocol changed")
    else:
        atomic_json(frozen, protocol)
    model = MaskablePPO.load(PPO, device="cpu")
    models = {"ppo": model}
    all_rows = []
    for case in cases:
        seed = case["seed"]
        path = RUN / f"seed_{seed}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row.get("case") != case or len(row.get("timings", [])) != 2 * REPEATS:
                raise RuntimeError(f"Existing benchmark case is invalid: {seed}")
            all_rows.append(row)
            continue
        snapshot_path = DATA / "selected_states" / f"seed_{seed}.pkl.gz"
        snapshot = pickle.loads(gzip.decompress(snapshot_path.read_bytes()))
        if snapshot["episode_seed"] != seed or snapshot["pieces"] != case["step"]:
            raise RuntimeError(f"Benchmark snapshot identity changed: {seed}")
        action = case["action"]
        if branch(snapshot, action, models, 100) != fast_branch(snapshot, action, models, 100):
            raise RuntimeError(f"Warmup branch output differs: {seed}")
        expected = None
        timings = []
        for rep in range(REPEATS):
            order = (("legacy", branch), ("fast", fast_branch)) if rep % 2 == 0 else (
                ("fast", fast_branch), ("legacy", branch))
            for name, runner in order:
                started = time.perf_counter()
                result = runner(snapshot, action, models, HORIZON)
                seconds = time.perf_counter() - started
                if expected is None:
                    expected = result
                elif result != expected:
                    raise RuntimeError(f"Full branch output differs: {seed}:{rep}:{name}")
                timings.append({"repeat": rep, "variant": name, "seconds": seconds})
        row = {"case": case, "timings": timings, "outcome": expected, "completed_at": now()}
        atomic_json(path, row)
        all_rows.append(row)
        print(f"seed={seed} pieces={expected['pieces']} "
              f"legacy={sum(x['seconds'] for x in timings if x['variant']=='legacy'):.3f}s "
              f"fast={sum(x['seconds'] for x in timings if x['variant']=='fast'):.3f}s", flush=True)
    old = [x["seconds"] for row in all_rows for x in row["timings"] if x["variant"] == "legacy"]
    fast = [x["seconds"] for row in all_rows for x in row["timings"] if x["variant"] == "fast"]
    summary = {"version": protocol["version"], "cases": len(cases), "timed_branches_per_variant": len(old),
               "horizon": HORIZON, "all_outputs_identical": True,
               "legacy_seconds_total": sum(old), "fast_seconds_total": sum(fast),
               "legacy_seconds_median": statistics.median(old),
               "fast_seconds_median": statistics.median(fast),
               "speedup_total": sum(old) / sum(fast),
               "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "file_sha256": {f"seed_{seed}.json": digest(RUN / f"seed_{seed}.json") for seed in SEEDS},
               "completed_at": now()}
    atomic_json(RUN / "summary.json", summary)


if __name__ == "__main__":
    main()
