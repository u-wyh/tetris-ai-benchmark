"""Independent structural audit of new PPO episode checkpoints."""

import gzip
import hashlib
import json
import pickle
from pathlib import Path

from training.evaluation.hybrid_queue import PPO, digest
from training.evaluation.ppo_transition_capture import PPO_SHA256, public_hash
from training.evaluation.ppo_top2_generalization_capture import (CAP, CHECKPOINT_EVERY, DEV_SEEDS,
                                                                   RESERVED_SEEDS, RUN, TRAIN_SEEDS,
                                                                   checkpoint_paths, episode_path)
from training.train_ppo import atomic_json, now


def audit_capture():
    status = json.loads((RUN / "capture_status.json").read_text())
    if status["stage"] != "complete" or status["completed"] != len(TRAIN_SEEDS + DEV_SEEDS):
        raise RuntimeError("Capture has not completed")
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO checkpoint changed")
    protocol = json.loads((RUN / "capture_protocol.json").read_text())
    source = Path(__file__).with_name("ppo_top2_generalization_capture.py")
    if protocol["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Capture source changed after freeze")
    seed_audit = json.loads((RUN / "seed_audit.json").read_text())
    if (seed_audit["candidate"] != list(TRAIN_SEEDS)
            or seed_audit["replication"] != list(DEV_SEEDS)
            or seed_audit["reserved"] != list(RESERVED_SEEDS)
            or seed_audit["exact_collision_count"] != 0):
        raise RuntimeError("Seed audit changed")
    files = {}
    counts = {"candidate_deaths": 0, "candidate_capped": 0,
              "replication_deaths": 0, "replication_capped": 0, "checkpoints": 0}
    for seed in TRAIN_SEEDS + DEV_SEEDS:
        path = episode_path(seed)
        row = json.loads(path.read_text())
        if (row["seed"] != seed or row["cap"] != CAP or row["ppo_sha256"] != PPO_SHA256
                or row["game_over"] == row["capped"] or not 0 < row["pieces"] <= CAP
                or (row["capped"] and row["pieces"] != CAP)
                or sum(row["clears_0_to_4"]) != row["pieces"]):
            raise RuntimeError(f"Episode identity, censoring, or clear counts invalid: {seed}")
        expected_steps = list(range(0, row["pieces"], CHECKPOINT_EVERY))
        actual_steps = [item["step"] for item in row["checkpoints"]]
        if actual_steps != expected_steps:
            raise RuntimeError(f"Sparse checkpoint schedule changed: {seed}")
        group = "candidate" if seed in TRAIN_SEEDS else "replication"
        counts[f"{group}_{'deaths' if row['game_over'] else 'capped'}"] += 1
        files[str(path.relative_to(RUN))] = digest(path)
        for item in row["checkpoints"]:
            step = item["step"]
            snapshot_path, manifest_path = checkpoint_paths(seed, step)
            if (item["seed"] != seed or item["ppo_sha256"] != PPO_SHA256
                    or item != json.loads(manifest_path.read_text())
                    or digest(snapshot_path) != item["sha256"]):
                raise RuntimeError(f"Checkpoint identity changed: {seed}:{step}")
            state = pickle.loads(gzip.decompress(snapshot_path.read_bytes()))  # locally captured, hashed
            if (state["episode_seed"] != seed or state["pieces"] != step
                    or state["max_pieces"] != CAP
                    or public_hash(state["core"].get_public_observation()) != item["public_sha256"]):
                raise RuntimeError(f"Checkpoint Core/public state mismatch: {seed}:{step}")
            files[str(snapshot_path.relative_to(RUN))] = item["sha256"]
            files[str(manifest_path.relative_to(RUN))] = digest(manifest_path)
            counts["checkpoints"] += 1
    summary = json.loads((RUN / "capture_summary.json").read_text())
    if any(summary[key] != counts[key] for key in ("candidate_deaths", "candidate_capped",
                                                   "replication_deaths", "replication_capped")):
        raise RuntimeError("Capture summary changed")
    for seed in RESERVED_SEEDS:
        if any(RUN.rglob(f"seed_{seed}*")):
            raise RuntimeError(f"Reserved seed unexpectedly used: {seed}")
    result = {"version": "top2_capture_audit_v1", "audited_at": now(), "episodes": len(TRAIN_SEEDS + DEV_SEEDS),
              **counts, "file_sha256": files}
    path = RUN / "capture_integrity.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {key: value for key, value in old.items() if key != "audited_at"} != {
                key: value for key, value in result.items() if key != "audited_at"}:
            raise RuntimeError("Capture integrity manifest changed")
    else:
        atomic_json(path, result)
    return result


if __name__ == "__main__":
    print(audit_capture())
