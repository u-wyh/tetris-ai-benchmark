"""Resumable V2 labels on genuine frozen-BC-visited TetrisCore states."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import board_features, candidate_features
from training.env.tetris_env import TetrisEnv, count_holes
from training.imitation_dataset import TEACHER, TEACHER_HASH, digest
from training.tetris_core import TetrisCore
from training.traditional.v2 import V2, VERSION
from training.train_ppo import ROOT, atomic_json, now

CHAMPION = ROOT / "runs/candidate_bc_next_stage/champion/model.zip"


def capture(seed, output, model, limit=1000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shard = output / f"seed_{seed}.jsonl.gz"
    manifest = output / f"seed_{seed}.json"
    model_hash = digest(CHAMPION)
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if (meta["seed"] != seed or meta["limit"] != limit or
                meta["learner_sha256"] != model_hash or
                meta["teacher_hash"] != TEACHER_HASH or
                digest(shard) != meta["sha256"]):
            raise RuntimeError(f"DAgger shard integrity failed: {shard}")
        return meta
    temporary = shard.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    core, teacher = TetrisCore(seed), V2(TEACHER)
    hashes = set()
    counts = {"samples": 0, "duplicate_states": 0, "height_ge_12": 0,
              "height_ge_16": 0, "holes_ge_4": 0, "teacher_hold": 0,
              "learner_teacher_agreement": 0}
    buckets = [0] * 5
    with gzip.open(temporary, "wt", encoding="utf8", compresslevel=3) as stream:
        while core.phase == "playing" and counts["samples"] < limit:
            public = core.get_public_observation()
            placements = core.get_legal_placements()
            features, mask = candidate_features(public, placements)
            if not mask.any():
                break
            public_hash = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
            if public_hash in hashes:
                counts["duplicate_states"] += 1
            else:
                hashes.add(public_hash)
            teacher_action = teacher.choose(public, placements)
            if not bool(mask[teacher_action]):
                raise RuntimeError("V2 DAgger label is illegal")
            obs = {"state": TetrisEnv._encode_observation(public), "candidate": features}
            learner_action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            learner_action = int(learner_action)
            if not bool(mask[learner_action]):
                raise RuntimeError("Frozen BC learner action is illegal")
            _, holes, height, _, _ = board_features(public["board"])
            counts["height_ge_12"] += height >= 12
            counts["height_ge_16"] += height >= 16
            counts["holes_ge_4"] += count_holes(public["board"]) >= 4
            counts["teacher_hold"] += teacher_action >= 920
            counts["learner_teacher_agreement"] += learner_action == teacher_action
            buckets[min(height // 4, 4)] += 1
            ids = mask.nonzero()[0]
            row = {"seed": seed, "step": counts["samples"], "public": public,
                   "state": obs["state"].tolist(), "legal_ids": ids.tolist(),
                   "legal_features": features[ids].tolist(), "action": teacher_action,
                   "teacher_version": VERSION, "teacher_hash": TEACHER_HASH,
                   "state_hash": public_hash, "learner_action": learner_action,
                   "learner_sha256": model_hash, "source": "dagger_bc_visited"}
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            core.step(learner_action)
            counts["samples"] += 1
    os.replace(temporary, shard)
    meta = {"seed": seed, "limit": limit, **counts,
            "height_buckets_0_3_4_7_8_11_12_15_16_plus": buckets,
            "unique_public_states": len(hashes), "game_over": core.phase == "over",
            "sha256": digest(shard), "bytes": shard.stat().st_size,
            "learner_sha256": model_hash, "teacher_version": VERSION,
            "teacher_hash": TEACHER_HASH, "completed_at": now()}
    atomic_json(manifest, meta)
    return meta


def collect(output, seeds, limit=1000, target=20000):
    output = Path(output)
    if not CHAMPION.exists():
        raise FileNotFoundError(CHAMPION)
    config = {"seeds": list(seeds), "limit": limit, "target_samples": target,
              "learner_checkpoint": str(CHAMPION), "learner_sha256": digest(CHAMPION),
              "teacher_version": VERSION, "teacher_config": asdict(TEACHER),
              "teacher_hash": TEACHER_HASH, "source": "real learner-visited TetrisCore",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        # Preserve the original code commit as provenance, while allowing a
        # documentation-only Git commit between interrupted collection runs.
        if {key: value for key, value in saved.items() if key != "git_commit"} != {
                key: value for key, value in config.items() if key != "git_commit"}:
            raise RuntimeError("DAgger collection configuration changed")
    if not config_path.exists():
        atomic_json(config_path, config)
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(CHAMPION), device="cpu")
    total = 0
    for seed in seeds:
        meta = capture(seed, output, model, limit)
        total += meta["samples"]
        print(f"DAgger seed={seed} samples={meta['samples']} total={total}", flush=True)
        if total >= target:
            break
    if total < target:
        raise RuntimeError(f"DAgger collected only {total} of {target} required samples")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--target", type=int, default=20000)
    args = parser.parse_args()
    collect(args.output, range(args.first_seed, args.first_seed + args.seeds), args.limit, args.target)


if __name__ == "__main__":
    main()
