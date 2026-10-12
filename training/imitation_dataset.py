"""Resumable, episode-isolated V2 Hold demonstrations from public Core states."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

from training.candidate_features import board_features, candidate_features
from training.env.tetris_env import TetrisEnv
from training.tetris_core import TetrisCore
from training.traditional.v2 import V2, V2Config, VERSION
from training.train_ppo import atomic_json, now

ROOT = Path(__file__).resolve().parents[1]
TEACHER = V2Config(mode="hold")
TEACHER_CONFIG = asdict(TEACHER)
TEACHER_HASH = hashlib.sha256(json.dumps(TEACHER_CONFIG, sort_keys=True).encode()).hexdigest()


def label_public(public, placements=None):
    """DAgger hook: label a learner-visited public state with the same teacher."""
    return V2(TEACHER).choose(public, placements)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(seed, output, limit=2000):
    """A single seed is an atomic shard; partial files are discarded on resume."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shard = output / f"seed_{seed}.jsonl.gz"
    manifest = output / f"seed_{seed}.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if (meta["seed"] != seed or meta["limit"] != limit or
                meta["teacher_hash"] != TEACHER_HASH or digest(shard) != meta["sha256"]):
            raise RuntimeError(f"Invalid completed shard: {shard}")
        return meta
    temporary = shard.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    core, teacher, encoder = TetrisCore(seed), V2(TEACHER), TetrisEnv()
    count = hold = danger = 0
    state_hashes = set()
    heights = [0] * 5
    with gzip.open(temporary, "wt", encoding="utf8", compresslevel=3) as stream:
        while core.phase == "playing" and count < limit:
            public = core.get_public_observation()
            placements = core.get_legal_placements()
            features, mask = candidate_features(public, placements)
            action = teacher.choose(public, placements)
            if not mask[action]:
                raise AssertionError("Teacher chose an illegal action")
            ids = mask.nonzero()[0]
            _, _, height, _, _ = board_features(public["board"])
            bucket = min(height // 4, 4)
            heights[bucket] += 1
            danger += height >= 12
            hold += action >= 920
            public_hash = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
            state_hashes.add(public_hash)
            record = {"seed": seed, "step": count, "public": public,
                      "state": encoder._encode_observation(public).tolist(),
                      "legal_ids": ids.tolist(), "legal_features": features[ids].tolist(),
                      "action": action, "teacher_version": VERSION,
                      "teacher_hash": TEACHER_HASH, "state_hash": public_hash}
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            core.step(action)
            count += 1
    os.replace(temporary, shard)
    meta = {"seed": seed, "limit": limit, "samples": count, "hold_actions": hold,
            "danger_height_ge_12": danger, "height_buckets_0_3_4_7_8_11_12_15_16_plus": heights,
            "unique_public_states": len(state_hashes), "game_over": core.phase == "over",
            "sha256": digest(shard), "bytes": shard.stat().st_size,
            "teacher_version": VERSION, "teacher_hash": TEACHER_HASH,
            "completed_at": now()}
    atomic_json(manifest, meta)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"first_seed": args.first_seed, "seeds": args.seeds, "limit": args.limit,
              "teacher_version": VERSION, "teacher_config": TEACHER_CONFIG,
              "teacher_hash": TEACHER_HASH, "core": "TetrisCore",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Dataset config changed")
    if not config_path.exists():
        atomic_json(config_path, config)
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        print(capture(seed, args.output, args.limit), flush=True)


if __name__ == "__main__":
    main()
