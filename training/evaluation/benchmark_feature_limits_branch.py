"""Paired archived branch benchmark for the evaluation feature limit change."""

import gzip
import hashlib
import json
import pickle
import statistics
import subprocess
import time

import torch
from sb3_contrib import MaskablePPO

import training.evaluation.failure_curriculum_eval as evaluation
from training.evaluation.fast_candidate_features import candidate_features as current_features
from training.evaluation.fast_counterfactual import fast_branch
from training.evaluation.audit_feature_limits_branch import check_case
from training.evaluation.hybrid_queue import PPO, digest
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/ppo_fast_feature_limits_branch_20261024"
DATA = ROOT / "runs/ppo_clear_cadence_counterfactual"
CASES = ((1550002, 3875), (1550005, 4900), (1550007, 6075))
HORIZON = 500
REPEATS = 2
MODEL_HASH = "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34"


def main():
    torch.set_num_threads(1)
    if digest(PPO) != MODEL_HASH:
        raise RuntimeError("Frozen PPO checkpoint changed")
    old_source = subprocess.check_output([
        "git", "show", "65e5c0c:training/evaluation/fast_candidate_features.py"], cwd=ROOT)
    old_namespace = {"__name__": "old_fast_candidate_features"}
    exec(compile(old_source, "baseline_fast_candidate_features.py", "exec"), old_namespace)
    old_features = old_namespace["candidate_features"]
    protocol = {
        "version": 1, "cases": CASES, "horizon": HORIZON, "repeats": REPEATS,
        "model_sha256": MODEL_HASH, "old_source_sha256": hashlib.sha256(old_source).hexdigest(),
        "new_source_sha256": digest(ROOT / "training/evaluation/fast_candidate_features.py"),
        "branch_source_sha256": digest(ROOT / "training/evaluation/fast_counterfactual.py"),
        "cpu_threads": 1, "order": "old,new,new,old", "metric": "full fast_branch wall time",
    }
    RUN.mkdir(parents=True, exist_ok=True)
    frozen = RUN / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise RuntimeError("Existing protocol differs")
    if not frozen.exists():
        atomic_json(frozen, protocol)
    model = MaskablePPO.load(PPO, device="cpu")
    rows = []
    try:
        for seed, step in CASES:
            path = RUN / f"seed_{seed}_step_{step}.json"
            if path.exists():
                row = json.loads(path.read_text())
                check_case(row, seed, step, data=DATA, horizon=HORIZON)
                rows.append(row)
                continue
            snapshot_path = DATA / "states" / f"seed_{seed}_step_{step}.pkl.gz"
            candidate_path = DATA / "candidates" / f"seed_{seed}_step_{step}.json"
            candidate = json.loads(candidate_path.read_text())
            if digest(snapshot_path) != candidate["snapshot_sha256"]:
                raise RuntimeError("Archived snapshot hash mismatch")
            snapshot = pickle.loads(gzip.decompress(snapshot_path.read_bytes()))
            if snapshot["episode_seed"] != seed or snapshot["pieces"] != step:
                raise RuntimeError("Archived snapshot identity mismatch")
            action = candidate["original_action"]
            timings, expected = [], None
            for repeat in range(REPEATS):
                for variant, feature_fn in (("old", old_features), ("new", current_features)) if repeat == 0 else (("new", current_features), ("old", old_features)):
                    evaluation.fast_candidate_features = feature_fn
                    started = time.perf_counter()
                    result = fast_branch(snapshot, action, {"ppo": model}, HORIZON)
                    seconds = time.perf_counter() - started
                    if expected is None:
                        expected = result
                    elif result != expected:
                        raise RuntimeError(f"Branch mismatch: {seed}:{step}:{variant}")
                    timings.append({"repeat": repeat, "variant": variant, "seconds": seconds})
            row = {"seed": seed, "step": step, "action": action,
                   "snapshot_sha256": candidate["snapshot_sha256"],
                   "candidate_sha256": digest(candidate_path), "timings": timings,
                   "outcome_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                   "pieces": expected["pieces"], "game_over": expected["game_over"],
                   "completed_at": now()}
            atomic_json(path, row)
            rows.append(row)
            print(seed, [(x["variant"], round(x["seconds"], 3)) for x in timings], flush=True)
    finally:
        evaluation.fast_candidate_features = current_features
    old = [x["seconds"] for row in rows for x in row["timings"] if x["variant"] == "old"]
    new = [x["seconds"] for row in rows for x in row["timings"] if x["variant"] == "new"]
    summary = {"protocol_sha256": digest(frozen), "cases": len(rows), "paired_runs": len(old),
               "outputs_identical": True, "old_total_seconds": sum(old), "new_total_seconds": sum(new),
               "speedup_total": sum(old) / sum(new), "paired_speedups": [a / b for a, b in zip(old, new)],
               "old_median_seconds": statistics.median(old), "new_median_seconds": statistics.median(new),
               "result_sha256": {f"seed_{row['seed']}_step_{row['step']}.json": digest(RUN / f"seed_{row['seed']}_step_{row['step']}.json") for row in rows},
               "completed_at": now()}
    atomic_json(RUN / "summary.json", summary)


if __name__ == "__main__":
    main()
