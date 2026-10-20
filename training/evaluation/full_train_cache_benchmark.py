"""Paired, transaction-complete CPU benchmark for cached legal placements."""

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.hybrid_queue import PPO, digest, verify_models
from training.train_ppo import ROOT, atomic_json, now
from training.train_vector_transaction import verify_task

RUN = ROOT / "runs/autonomous_rl_research/full_train_cache_benchmark"
TASKS = 2
SEED = 1493000


def preset(fast):
    return {"target_total_steps": TASKS * 4096, "task_steps": 4096,
            "device": "cpu", "n_envs": 4, "n_steps": 1024,
            "base_seed": SEED, "candidate_policy": True,
            "fast_candidate_env": fast, "max_pieces": 100000,
            "overnight_monitor": True, "step0_baseline": False,
            "transaction_retention": 3, "milestone_interval": 100000000,
            "validation_interval": 100000000, "final_test_seeds": 100,
            "final_test_max_pieces": 50000,
            "ppo_overrides": {"gamma": 0.999, "gae_lambda": 0.98},
            "initial_model": {"path": str(PPO), "sha256": digest(PPO)},
            "kl_reference": {"path": str(PPO), "sha256": digest(PPO), "beta": 0.03}}


def record(task_dir, number):
    config = json.loads((task_dir.parent.parent / "config.json").read_text())
    verify_task(task_dir, number, config)
    metrics = json.loads((task_dir / "metrics.json").read_text())
    return {"wall_seconds": metrics["wall_seconds"],
            "model_sha256": hashlib.sha256((task_dir / "model.zip").read_bytes()).hexdigest(),
            "metrics": {k: v for k, v in metrics.items() if k != "wall_seconds"}}


def train(label, number):
    directory = RUN / label
    configuration = RUN / f"{label}_preset.json"
    command = [sys.executable, "-m", "training.train_vector_transaction",
               "--run-dir", str(directory), "--config-file", str(configuration),
               "--max-tasks", "1"]
    if number == 2:
        command.append("--resume")
    started = time.monotonic()
    with (RUN / f"{label}_task_{number}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Training {label} task {number} failed; inspect its log")
    task_dir = directory / f"committed/task_{number:06d}"
    row = record(task_dir, number)
    row["process_seconds"] = time.monotonic() - started
    atomic_json(RUN / f"{label}_task_{number}.json", row)
    return row


def compare(number):
    ordinary = RUN / f"ordinary/committed/task_{number:06d}/model.zip"
    cached = RUN / f"cached/committed/task_{number:06d}/model.zip"
    left = MaskablePPO.load(ordinary, device="cpu").policy.state_dict()
    right = MaskablePPO.load(cached, device="cpu").policy.state_dict()
    if left.keys() != right.keys():
        raise RuntimeError("Cached training changed policy structure")
    different = [name for name in left if not torch.equal(left[name], right[name])]
    if different:
        raise RuntimeError(f"Cached training changed model parameters: {different[:3]}")
    a = json.loads((RUN / f"ordinary_task_{number}.json").read_text())
    b = json.loads((RUN / f"cached_task_{number}.json").read_text())
    if a["metrics"] != b["metrics"]:
        raise RuntimeError("Cached training changed task metrics")
    return {"task": number, "policy_exact": True, "task_metrics_exact": True,
            "ordinary_wall_seconds": a["wall_seconds"],
            "cached_wall_seconds": b["wall_seconds"],
            "speedup": a["wall_seconds"] / b["wall_seconds"]}


def main():
    verify_models()
    RUN.mkdir(parents=True, exist_ok=True)
    for seed in range(SEED, SEED + 4):
        for root in (ROOT / "runs", ROOT / "reports"):
            for path in root.rglob(f"seed_{seed}.*"):
                if not path.is_relative_to(RUN):
                    raise RuntimeError(f"Benchmark training seed already used: {path}")
    identity = {"created_at": now(), "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": SEED, "tasks_per_mode": TASKS,
        "trainer_sha256": hashlib.sha256((ROOT / "training/train_vector_transaction.py").read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol = RUN / "protocol.json"
    if protocol.exists():
        old = json.loads(protocol.read_text())
        for key in ("seed", "tasks_per_mode", "trainer_sha256", "benchmark_sha256"):
            if old[key] != identity[key]:
                raise RuntimeError("Frozen full-training benchmark changed")
    else:
        atomic_json(protocol, identity)
    for label, fast in (("ordinary", False), ("cached", True)):
        path = RUN / f"{label}_preset.json"
        value = preset(fast)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise RuntimeError("Benchmark preset changed")
        else:
            atomic_json(path, value)
    # Alternate order across complete tasks to reduce order effects.
    rows = []
    for number, order in ((1, ("ordinary", "cached")),
                          (2, ("cached", "ordinary"))):
        for label in order:
            result = RUN / f"{label}_task_{number}.json"
            if not result.exists():
                train(label, number)
            else:
                record(RUN / label / f"committed/task_{number:06d}", number)
        rows.append(compare(number))
    result = {"protocol": identity, "paired_tasks": rows,
              "total_steps_per_mode": TASKS * 4096,
              "speedup": sum(r["ordinary_wall_seconds"] for r in rows) /
                         sum(r["cached_wall_seconds"] for r in rows),
              "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    atomic_json(RUN / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
