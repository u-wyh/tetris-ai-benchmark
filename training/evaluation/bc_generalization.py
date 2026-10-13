"""Frozen BC versus V2 Hold on 30 previously unused 100k-piece seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.champion_pipeline import evaluate_seed
from training.evaluation.traditional_v2 import play
from training.imitation_bc import batch, records
from training.imitation_dataset import TEACHER_HASH, digest
from training.traditional.v2 import V2Config, VERSION
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/candidate_bc_next_stage"
CHAMPION = RUN / "champion/model.zip"
SOURCE = ROOT / "runs/imitation_v2_hold_20m/bc/epoch_03.zip"
SEEDS = list(range(1234000, 1234030))
CAP = 100000
WORKERS = 6


def code_digest(directory):
    h = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def freeze():
    source_config = json.loads((ROOT / "runs/ppo_candidate_raw_20m_seed42/config.json").read_text())
    bc_config = json.loads((SOURCE.parent / "config.json").read_text())
    expected = {"policy": "CandidateScoringPolicy", "reward_version": "placement-reward-v1",
                "observation_version": "candidate-dict-state237-features16-v1",
                "candidate_feature_version": "candidate-placement-uint8-v1",
                "action_space_version": "1840-legal-placement-v1"}
    if any(source_config.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Source PPO config differs from Candidate protocol")
    if (bc_config["label"] != "20M PPO + BC" or bc_config["teacher_hash"] != TEACHER_HASH
            or bc_config["ppo_environment_steps_added"] != 0 or bc_config["epochs"] != 3
            or Path(bc_config["source_model"]) !=
            ROOT / "runs/ppo_candidate_raw_20m_seed42/milestones/step_020000768/model.zip"
            or digest(Path(bc_config["source_model"])) != bc_config["source_sha256"]):
        raise RuntimeError("BC source or teacher identity differs")
    dataset = Path(bc_config["dataset"])
    manifests = sorted(dataset.glob("seed_*.json"))
    if len(manifests) != 108:
        raise RuntimeError("Expert shard count changed")
    manifest_hash = hashlib.sha256()
    sample_count = 0
    for path in manifests:
        meta = json.loads(path.read_text())
        shard = path.with_suffix(".jsonl.gz")
        if meta["teacher_hash"] != TEACHER_HASH or digest(shard) != meta["sha256"]:
            raise RuntimeError(f"Expert shard integrity failed: {path}")
        sample_count += meta["samples"]
        manifest_hash.update(path.name.encode())
        manifest_hash.update(meta["sha256"].encode())
    if sample_count != 214809:
        raise RuntimeError("Expert sample count changed")
    CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    if not CHAMPION.exists():
        temporary = CHAMPION.with_suffix(".tmp")
        shutil.copy2(SOURCE, temporary)
        if digest(temporary) != digest(SOURCE):
            raise RuntimeError("BC Champion copy changed")
        os.replace(temporary, CHAMPION)
    if digest(CHAMPION) != digest(SOURCE):
        raise RuntimeError("Frozen BC Champion changed")
    record = {"label": "20M PPO + BC epoch 3", "source": str(SOURCE),
              "model_sha256": digest(CHAMPION), "source_ppo_steps": 20000768,
              "bc_epochs": 3, "bc_samples_total": sample_count,
              "dataset_manifest_sha256": manifest_hash.hexdigest(),
              "teacher_version": VERSION, "teacher_config_sha256": TEACHER_HASH,
              "core_code_sha256": code_digest(ROOT / "training/tetris_core"),
              "source_config": {k: source_config[k] for k in expected},
              "bc_config": bc_config}
    path = CHAMPION.parent / "provenance.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError("Frozen Champion provenance changed")
    if not path.exists():
        atomic_json(path, record)
    sample = next(records(dataset, [600100], 1))
    obs, mask, _ = batch([sample], "cpu")
    torch.set_num_threads(1)
    left, right = MaskablePPO.load(str(SOURCE), device="cpu"), MaskablePPO.load(str(CHAMPION), device="cpu")
    with torch.no_grad():
        a = left.policy.get_distribution(obs, action_masks=mask).distribution.logits
        b = right.policy.get_distribution(obs, action_masks=mask).distribution.logits
    if not torch.equal(a, b):
        raise RuntimeError("Champion inference changed after copy")
    subprocess.run([sys.executable, "-m", "training.evaluation.bc_generalization", "--verify"],
                   cwd=ROOT, check=True)
    return record


def verify_independent():
    torch.set_num_threads(1)
    if digest(CHAMPION) != json.loads((CHAMPION.parent / "provenance.json").read_text())["model_sha256"]:
        raise RuntimeError("Champion digest failed in independent process")
    model = MaskablePPO.load(str(CHAMPION), device="cpu")
    sample = next(records(ROOT / "runs/imitation_v2_hold_20m/data", [600100], 1))
    obs, mask, _ = batch([sample], "cpu")
    with torch.no_grad():
        action = int(model.policy.get_distribution(obs, action_masks=mask).get_actions(deterministic=True)[0])
    if not bool(mask[0, action]):
        raise RuntimeError("Frozen Champion selected illegal action")


def seed_audit():
    # Existing fixed sets, demonstration sets, smoke seeds, and previous independent tests.
    occupied = (set(range(42, 50)) | set(range(100000, 100032)) |
                set(range(200000, 200100)) | set(range(300000, 300100)) |
                set(range(400000, 400100)) | set(range(500000, 500002)) |
                set(range(600000, 600108)) | {800000, 900001})
    if set(SEEDS) & occupied:
        raise RuntimeError("Fresh seeds overlap known sets")
    for base in (ROOT / "runs", ROOT / "reports"):
        for seed in SEEDS:
            matches = list(base.rglob(f"seed_{seed}.json")) + list(base.rglob(f"seed_{seed}.jsonl.gz"))
            if any(not p.is_relative_to(RUN) for p in matches):
                raise RuntimeError(f"Fresh seed was already used: {seed}")
    return {"seeds": SEEDS, "max_pieces": CAP, "purpose": "one-shot generalization test",
            "no_hyperparameter_selection": True, "audited_at": now()}


def evaluate_one(kind, seed):
    torch.set_num_threads(1)
    if kind == "bc":
        model = MaskablePPO.load(str(CHAMPION), device="cpu")
        row = evaluate_seed(model, seed, CAP)
    else:
        row = play(seed, CAP, V2Config(mode="hold"))
        row.pop("actions")
        row.pop("decision_ms")
    atomic_json(RUN / "generalization" / kind / f"seed_{seed}.json", row)
    return kind, seed, row["pieces_survived"]


def summarize(rows):
    pieces = sorted(r["pieces_survived"] for r in rows)
    at = lambda f: pieces[max(0, math.ceil(f * len(pieces)) - 1)]
    return {"mean_pieces": statistics.mean(pieces), "median_pieces": statistics.median(pieces),
            "p10_pieces": at(.1), "p90_pieces": at(.9),
            "capped_count": sum(v == CAP for v in pieces),
            "mean_lines": statistics.mean(r["lines"] for r in rows),
            "mean_score": statistics.mean(r["score"] for r in rows),
            "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in rows),
            "total_worker_seconds": sum(r["wall_seconds"] for r in rows)}


def generalization():
    RUN.mkdir(parents=True, exist_ok=True)
    freeze()
    audit = seed_audit()
    protocol = {**audit, "bc_sha256": digest(CHAMPION), "teacher_version": VERSION,
                "teacher_config_sha256": TEACHER_HASH,
                "core_code_sha256": code_digest(ROOT / "training/tetris_core"),
                "deterministic": True, "action_mask": True, "workers": WORKERS}
    path = RUN / "generalization/protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        original = json.loads(path.read_text())
        if {k: v for k, v in original.items() if k != "audited_at"} != {k: v for k, v in protocol.items() if k != "audited_at"}:
            raise RuntimeError("Frozen generalization protocol changed")
    else:
        atomic_json(path, protocol)
    for kind in ("bc", "v2_hold"):
        (path.parent / kind).mkdir(exist_ok=True)
    pending = []
    for kind in ("bc", "v2_hold"):
        for seed in SEEDS:
            target = path.parent / kind / f"seed_{seed}.json"
            if target.exists():
                row = json.loads(target.read_text())
                if row["seed"] != seed or row["pieces_survived"] > CAP:
                    raise RuntimeError(f"Invalid saved result: {target}")
            else:
                pending.append((kind, seed))
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(evaluate_one, kind, seed): (kind, seed) for kind, seed in pending}
            for future in as_completed(futures):
                kind, seed, pieces = future.result()
                atomic_json(path.parent / "status.json", {"state": "running", "last_kind": kind,
                    "last_seed": seed, "completed": 60 - sum(not (path.parent / k / f"seed_{s}.json").exists()
                    for k in ("bc", "v2_hold") for s in SEEDS), "total": 60, "updated_at": now()})
                print(f"{kind} seed={seed} pieces={pieces}", flush=True)
        rows = {kind: [json.loads((path.parent / kind / f"seed_{seed}.json").read_text()) for seed in SEEDS]
                for kind in ("bc", "v2_hold")}
        paired = {"bc_wins": 0, "v2_wins": 0, "both_capped": 0, "equal_uncapped": 0}
        for a, b in zip(rows["bc"], rows["v2_hold"]):
            if a["pieces_survived"] == b["pieces_survived"] == CAP:
                paired["both_capped"] += 1
            elif a["pieces_survived"] > b["pieces_survived"]:
                paired["bc_wins"] += 1
            elif a["pieces_survived"] < b["pieces_survived"]:
                paired["v2_wins"] += 1
            else:
                paired["equal_uncapped"] += 1
        result = {"bc": summarize(rows["bc"]), "v2_hold": summarize(rows["v2_hold"]),
                  "paired": paired, "wall_seconds": time.monotonic()-started, "completed_at": now()}
        atomic_json(path.parent / "summary.json", result)
        atomic_json(path.parent / "status.json", {"state": "completed", "completed": 60,
                    "total": 60, "updated_at": now()})
    except Exception:
        atomic_json(path.parent / "status.json", {"state": "failed", "error": traceback.format_exc(),
                    "updated_at": now()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--wait-existing", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_independent()
    elif args.freeze:
        freeze()
    else:
        if args.wait_existing:
            old = ROOT / "runs/imitation_v2_hold_20m/status.json"
            while json.loads(old.read_text())["state"] == "running":
                time.sleep(30)
            if json.loads(old.read_text())["state"] != "completed":
                raise RuntimeError("Existing comparison did not complete normally")
        generalization()


if __name__ == "__main__":
    main()
