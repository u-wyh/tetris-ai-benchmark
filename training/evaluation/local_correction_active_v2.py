"""Bounded, resumable V2 counterfactuals from immutable PPO trajectory shards."""
import gzip
import hashlib
import json
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch

from training.evaluation.hybrid_queue import HASHES, load_models, verify_models
from training.imitation_dataset import digest
from training.local_correction_data import (RUN, TRAIN, VALIDATION, HORIZON, _paths,
                                            _replay_snapshots, classify)
from training.train_ppo import atomic_json, now

V2 = RUN / "active_v2"
SEEDS = TRAIN + VALIDATION
PER_SEED = 30
RANDOM_QUOTA = 6
MIN_SPACING = 10
VERSION = "active_candidates_v2_up_to_30_per_seed_random_6_spacing_10_original_grade_v4"
MODELS = None


def _row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_seed(seed):
    shard, _, meta_path = _paths(seed)
    meta = json.loads(meta_path.read_text())
    if meta["models"] != HASHES or digest(shard) != meta["shard_sha256"]:
        raise RuntimeError(f"Source mismatch {seed}")
    with gzip.open(shard, "rt", encoding="utf8") as stream:
        rows = {row["step"]: row for row in map(json.loads, stream)}
    old = json.loads((RUN / "labels" / f"seed_{seed}.json").read_text())
    if old["source_sha256"] != meta["shard_sha256"]:
        raise RuntimeError(f"V1 label source mismatch {seed}")
    return rows, meta, old


def _priority(row):
    p, b = row["ppo_risk"], row["bc_risk"]
    d = row["death_distance"]
    near_death = d is not None and d <= HORIZON
    return (0 if near_death else 1,
            abs(d - HORIZON // 2) if near_death else 0,
            -int(p["lock_out"]), -int(p["height_growth"] >= 3),
            -max(0, p["height"] - b["height"]), -p["new_holes"], row["step"])


def choose(seed):
    rows, meta, old = _load_seed(seed)
    path = V2 / "selection" / f"seed_{seed}.json"
    excluded = set(old["branch_steps"])
    pool = [r for r in rows.values() if r["category"] == "candidate" and r["step"] not in excluded]
    rng = random.Random(20261015 + seed)
    chosen = []
    def add(row):
        if all(abs(row["step"] - other["step"]) >= MIN_SPACING for other in chosen):
            chosen.append(row)
            return True
        return False
    for row in sorted(pool, key=_priority):
        if len(chosen) >= PER_SEED - RANDOM_QUOTA:
            break
        add(row)
    high_count = len(chosen)
    remaining = [r for r in pool if r["step"] not in {x["step"] for x in chosen}]
    rng.shuffle(remaining)
    for row in remaining:
        if len(chosen) >= PER_SEED:
            break
        add(row)
    if not chosen:
        raise RuntimeError(f"No diverse risk states in {seed}")
    selection = {"seed": seed, "version": VERSION, "source_sha256": meta["shard_sha256"],
                 "old_labels_sha256": digest(RUN / "labels" / f"seed_{seed}.json"),
                 "candidate_pool": len(pool), "priority_count": high_count,
                 "random_count": len(chosen) - high_count,
                 "steps": sorted(r["step"] for r in chosen),
                 "row_hashes": {str(r["step"]): _row_hash(r) for r in chosen}}
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise RuntimeError(f"Frozen selection changed {seed}")
    else:
        atomic_json(path, selection)
    return selection


def _init():
    global MODELS
    torch.set_num_threads(1)
    MODELS = load_models()


def _run_seed(seed):
    selection = choose(seed)
    rows, meta, _ = _load_seed(seed)
    targets = selection["steps"]
    states = _replay_snapshots(seed, rows, targets, MODELS, meta["shard_sha256"],
                               cache_root=V2 / "snapshots")
    completed = 0
    for step in targets:
        path = V2 / "branches" / f"seed_{seed}" / f"step_{step}.json"
        identity = {"seed": seed, "step": step, "models": HASHES, "horizon": HORIZON,
                    "version": VERSION, "source_sha256": meta["shard_sha256"],
                    "selection_sha256": digest(V2 / "selection" / f"seed_{seed}.json"),
                    "row_sha256": selection["row_hashes"][str(step)]}
        if path.exists():
            prior = json.loads(path.read_text())
            if any(prior.get(k) != v for k, v in identity.items()):
                raise RuntimeError(f"Completed branch identity mismatch {seed}:{step}")
        else:
            evidence = classify(rows[step], states[step], MODELS)
            atomic_json(path, {**identity, "row": rows[step], "evidence": evidence,
                               "completed_at": now()})
        completed += 1
    return {"seed": seed, "completed": completed}


def _build_dataset():
    files = [V2 / "branches" / f"seed_{seed}" / f"step_{step}.json"
             for seed in SEEDS for step in choose(seed)["steps"]]
    if not all(path.exists() for path in files):
        raise RuntimeError("Incomplete V2 branch set")
    records = [json.loads(path.read_text()) for path in files]
    if any(record["version"] != VERSION or record["models"] != HASHES for record in records):
        raise RuntimeError("V2 branch identity mismatch")
    manifest = {"version": VERSION, "models": HASHES, "horizon": HORIZON,
                "train_seeds": TRAIN, "validation_seeds": VALIDATION,
                "branch_count": len(files),
                "branch_hashes": {str(path.relative_to(V2)): digest(path) for path in files},
                "selection_hashes": {str(seed): digest(V2 / "selection" / f"seed_{seed}.json")
                                     for seed in SEEDS}}
    path = V2 / "dataset_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Immutable V2 dataset manifest changed")
    else:
        atomic_json(path, manifest)
    train_a = [r for r in records if r["seed"] in TRAIN and r["evidence"]["category"] == "A"]
    val_a = [r for r in records if r["seed"] in VALIDATION and r["evidence"]["category"] == "A"]
    train_seed_count = len({r["seed"] for r in train_a})
    non_hold = sum(not r["evidence"]["hold_changed"] for r in train_a)
    decision = {"state": "ready_for_smoke_only" if len(train_a) >= 100 and len(val_a) >= 30
                and train_seed_count >= 8 and non_hold >= 30 else "insufficient_quality_data",
                "train_A_new": len(train_a), "validation_A_new": len(val_a),
                "train_A_episodes": train_seed_count, "train_A_non_hold": non_hold,
                "verified_new": len(records), "completed_at": now()}
    atomic_json(V2 / "readiness.json", decision)
    return records, decision


def main():
    V2.mkdir(parents=True, exist_ok=True)
    try:
        verify_models()
        protocol = {"version": VERSION, "models": HASHES,
                    "seeds": {"train": TRAIN, "validation": VALIDATION},
                    "per_seed": PER_SEED, "random_quota": RANDOM_QUOTA,
                    "min_step_spacing": MIN_SPACING, "horizon": HORIZON,
                    "grade": "unchanged_local_correction_v4"}
        protocol_path = V2 / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Frozen V2 protocol changed")
        else:
            atomic_json(protocol_path, protocol)
        if not (V2 / "diagnosis.json").exists():
            raise RuntimeError("Training mechanism diagnosis must finish before active sampling")
        for seed in SEEDS:
            choose(seed)
        atomic_json(V2 / "status.json", {"state": "running", "stage": "counterfactual_validation_v2",
                    "completed_seeds": 0, "total_seeds": len(SEEDS), "updated_at": now()})
        with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=_init) as pool:
            futures = [pool.submit(_run_seed, seed) for seed in SEEDS]
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                atomic_json(V2 / "status.json", {"state": "running", "stage": "counterfactual_validation_v2",
                            "completed_seeds": index, "total_seeds": len(SEEDS),
                            "last_seed": result["seed"], "updated_at": now()})
                print(result, flush=True)
        _, decision = _build_dataset()
        atomic_json(V2 / "status.json", {"state": "completed", "decision": decision,
                    "updated_at": now()})
    except Exception:
        atomic_json(V2 / "status.json", {"state": "failed", "traceback": traceback.format_exc(),
                    "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
