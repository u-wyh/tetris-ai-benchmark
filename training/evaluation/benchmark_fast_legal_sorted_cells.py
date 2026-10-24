"""Bounded paired branch check for redundant fast placement visited lookups."""

import gzip
import hashlib
import json
import pickle
import time

import torch
from sb3_contrib import MaskablePPO

import training.evaluation.failure_curriculum_eval as evaluation
from training.evaluation.fast_counterfactual import fast_branch
from training.evaluation.fast_legal import get_legal_placements as current_placements
from training.evaluation.hybrid_queue import PPO, digest
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/ppo_fast_legal_sorted_cells_20261024"
DATA = ROOT / "runs/ppo_clear_cadence_counterfactual"
CASES = ((1550002, 3875), (1550005, 4900), (1550007, 6075))
MODEL_HASH = "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34"


def main():
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True)
    if digest(PPO) != MODEL_HASH:
        raise RuntimeError("Frozen model changed")
    old_source = (RUN / "old_fast_legal.py").read_bytes()
    namespace = {"__name__": "old_fast_legal"}
    exec(compile(old_source, "old_fast_legal.py", "exec"), namespace)
    candidate_source = (RUN / "candidate_fast_legal.py").read_bytes()
    candidate_namespace = {"__name__": "candidate_fast_legal"}
    exec(compile(candidate_source, "candidate_fast_legal.py", "exec"), candidate_namespace)
    protocol = {
        "cases": [list(case) for case in CASES], "horizon": 500, "repeats": 2,
        "order": "old,new,new,old", "cpu_threads": 1,
        "model_sha256": MODEL_HASH,
        "old_source_sha256": hashlib.sha256(old_source).hexdigest(),
        "new_source_sha256": hashlib.sha256(candidate_source).hexdigest(),
        "branch_source_sha256": digest(ROOT / "training/evaluation/fast_counterfactual.py"),
    }
    frozen = RUN / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise RuntimeError("Protocol changed")
    if not frozen.exists():
        atomic_json(frozen, protocol)
    model = MaskablePPO.load(PPO, device="cpu")
    rows = []
    try:
        for seed, step in CASES:
            path = RUN / f"seed_{seed}_step_{step}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            snapshot_path = DATA / "states" / f"seed_{seed}_step_{step}.pkl.gz"
            candidate_path = DATA / "candidates" / f"seed_{seed}_step_{step}.json"
            candidate = json.loads(candidate_path.read_text())
            if digest(snapshot_path) != candidate["snapshot_sha256"]:
                raise RuntimeError("Snapshot changed")
            snapshot = pickle.loads(gzip.decompress(snapshot_path.read_bytes()))
            if (snapshot["episode_seed"], snapshot["pieces"]) != (seed, step):
                raise RuntimeError("Snapshot identity changed")
            action = candidate["original_action"]
            timings, expected = [], None
            for repeat in range(2):
                variants = (("old", namespace["get_legal_placements"]),
                            ("new", candidate_namespace["get_legal_placements"]))
                if repeat:
                    variants = variants[::-1]
                for label, placement_fn in variants:
                    evaluation.fast_placements = placement_fn
                    started = time.perf_counter()
                    result = fast_branch(snapshot, action, {"ppo": model}, 500)
                    seconds = time.perf_counter() - started
                    if expected is None:
                        expected = result
                    elif result != expected:
                        raise RuntimeError("Branch result changed")
                    timings.append({"variant": label, "repeat": repeat, "seconds": seconds})
            row = {"seed": seed, "step": step, "snapshot_sha256": digest(snapshot_path),
                   "candidate_sha256": digest(candidate_path), "action": action,
                   "timings": timings, "outcome_sha256": hashlib.sha256(
                       json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                   "completed_at": now()}
            atomic_json(path, row)
            rows.append(row)
            print(seed, [(x["variant"], round(x["seconds"], 3)) for x in timings], flush=True)
    finally:
        evaluation.fast_placements = current_placements
    old = [x["seconds"] for row in rows for x in row["timings"] if x["variant"] == "old"]
    new = [x["seconds"] for row in rows for x in row["timings"] if x["variant"] == "new"]
    atomic_json(RUN / "summary.json", {
        "cases": len(rows), "paired_runs": len(old), "outputs_identical": True,
        "old_seconds": sum(old), "new_seconds": sum(new), "speedup": sum(old) / sum(new),
        "pair_speedups": [a / b for a, b in zip(old, new)],
        "protocol_sha256": digest(frozen), "completed_at": now(),
    })


if __name__ == "__main__":
    main()
