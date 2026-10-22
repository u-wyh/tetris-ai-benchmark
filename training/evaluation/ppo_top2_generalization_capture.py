"""Fixed-seed PPO episodes with sparse verifiable replay checkpoints."""

import gzip
import hashlib
import json
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.hybrid_queue import PPO, digest
from training.evaluation.ppo_transition_capture import PPO_SHA256, public_hash
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now

RUN = ROOT / "runs/ppo30m_top2_generalization"
TRAIN_SEEDS = tuple(range(1540000, 1540016))
DEV_SEEDS = tuple(range(1541000, 1541008))
RESERVED_SEEDS = tuple(range(1542000, 1542016))
SEEDS = TRAIN_SEEDS + DEV_SEEDS
CAP = 50000
CHECKPOINT_EVERY = 10000
VERSION = "top2_generalization_capture_v1"


def checkpoint_paths(seed, step):
    stem = f"seed_{seed}_step_{step}"
    return RUN / "checkpoints" / f"{stem}.pkl.gz", RUN / "checkpoints" / f"{stem}.json"


def episode_path(seed):
    return RUN / "episodes" / f"seed_{seed}.json"


def verify_checkpoint(seed, step, public_sha256):
    path, manifest_path = checkpoint_paths(seed, step)
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Checkpoint pair incomplete: {seed}:{step}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["seed"] != seed or manifest["step"] != step
            or manifest["public_sha256"] != public_sha256
            or manifest["ppo_sha256"] != PPO_SHA256 or digest(path) != manifest["sha256"]):
        raise RuntimeError(f"Checkpoint changed: {seed}:{step}")
    return manifest


def verify_episode(seed, row):
    if (row["version"] != VERSION or row["seed"] != seed or row["cap"] != CAP
            or row["ppo_sha256"] != PPO_SHA256 or row["pieces"] > CAP
            or row["game_over"] == row["capped"] or sum(row["clears_0_to_4"]) != row["pieces"]):
        raise RuntimeError(f"Episode identity or censoring invalid: {seed}")
    for item in row["checkpoints"]:
        if verify_checkpoint(seed, item["step"], item["public_sha256"]) != item:
            raise RuntimeError(f"Episode checkpoint changed: {seed}:{item['step']}")
    return row


def save_checkpoint(seed, step, env):
    public_sha256 = public_hash(env.core.get_public_observation())
    path, manifest_path = checkpoint_paths(seed, step)
    if path.exists() and not manifest_path.exists():
        raw = path.read_bytes()
        state = pickle.loads(gzip.decompress(raw))  # locally generated interrupted checkpoint
        if (state["episode_seed"] != seed or state["pieces"] != step
                or public_hash(state["core"].get_public_observation()) != public_sha256):
            raise RuntimeError(f"Orphan checkpoint disagrees with replay: {seed}:{step}")
        atomic_json(manifest_path, {"seed": seed, "step": step,
                                    "sha256": hashlib.sha256(raw).hexdigest(),
                                    "public_sha256": public_sha256, "ppo_sha256": PPO_SHA256})
    if path.exists() or manifest_path.exists():
        return verify_checkpoint(seed, step, public_sha256)
    raw = gzip.compress(pickle.dumps(env.get_state(), protocol=pickle.HIGHEST_PROTOCOL), compresslevel=3)
    manifest = {"seed": seed, "step": step, "sha256": hashlib.sha256(raw).hexdigest(),
                "public_sha256": public_sha256, "ppo_sha256": PPO_SHA256}
    atomic_bytes(path, raw)
    atomic_json(manifest_path, manifest)
    return verify_checkpoint(seed, step, public_sha256)


def capture(seed):
    torch.set_num_threads(1)
    path = episode_path(seed)
    if path.exists():
        return verify_episode(seed, json.loads(path.read_text()))
    model = MaskablePPO.load(PPO, device="cpu")
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    checkpoints = []
    clears = [0] * 5
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            step = env.pieces
            if step % CHECKPOINT_EVERY == 0:
                checkpoints.append(save_checkpoint(seed, step, env))
            mask = env.action_masks()
            action = fast_action(model, obs, mask)
            if not mask[action]:
                raise RuntimeError(f"Illegal PPO action: {seed}:{step}")
            obs, _, dead, capped, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            if dead or capped:
                break
        row = {"version": VERSION, "seed": seed, "cap": CAP, "ppo_sha256": PPO_SHA256,
               "pieces": env.pieces, "game_over": bool(dead), "capped": bool(capped),
               "score": env.core.score, "clears_0_to_4": clears,
               "final_public_sha256": public_hash(env.core.get_public_observation()),
               "checkpoints": checkpoints, "completed_at": now()}
        atomic_json(path, row)
        return verify_episode(seed, row)
    finally:
        env.close()


def audit_seeds():
    pattern = (r'"(?:seed|base_seed|run_seed|episode_seed)"\s*:\s*'
               r'154(?:000(?:[0-9])|001[0-5]|100[0-7]|20(?:0[0-9]|1[0-5]))\b|'
               r'seed_154(?:000[0-9]|001[0-5]|100[0-7]|20(?:0[0-9]|1[0-5]))\b')
    historical_root = Path("/home/wyh/projects/tetris-ai-benchmark")
    audit = subprocess.run(["rg", "-n", "-P", pattern, "training", "tests", "reports", "runs",
                            "--glob", "*.json", "--glob", "*.md", "--glob", "*.py",
                            "--glob", "*.txt", "--glob", "*.log"],
                           cwd=historical_root, text=True, capture_output=True, check=False)
    if audit.returncode != 1:
        raise RuntimeError(f"Exact-seed audit failed: {audit.stdout[:250]} {audit.stderr[:250]}")
    atomic_json(RUN / "seed_audit.json", {"candidate": list(TRAIN_SEEDS), "replication": list(DEV_SEEDS),
                                                 "reserved": list(RESERVED_SEEDS),
                                                 "exact_collision_count": 0, "pattern": pattern,
                                                 "searched_root": str(historical_root), "at": now()})


def main():
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO changed")
    for name in ("checkpoints", "episodes"):
        (RUN / name).mkdir(parents=True, exist_ok=True)
    protocol = {"version": VERSION, "candidate": list(TRAIN_SEEDS), "replication": list(DEV_SEEDS),
                "reserved": list(RESERVED_SEEDS), "cap": CAP, "checkpoint_every": CHECKPOINT_EVERY,
                "max_workers": 4, "ppo_sha256": PPO_SHA256,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    frozen = RUN / "capture_protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Capture protocol changed")
    else:
        audit_seeds()
        atomic_json(frozen, protocol)
    completed = 0
    for group_name, group in (("candidate", TRAIN_SEEDS), ("replication", DEV_SEEDS)):
        if group_name == "replication":
            candidate = [verify_episode(seed, json.loads(episode_path(seed).read_text()))
                         for seed in TRAIN_SEEDS]
            if sum(row["game_over"] for row in candidate) < 3:
                atomic_json(RUN / "capture_status.json", {"stage": "insufficient_candidate_deaths",
                                                            "completed": len(candidate), "total": len(SEEDS), "at": now()})
                return
        with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(capture, seed): seed for seed in group}
            for future in as_completed(futures):
                seed = futures[future]
                row = future.result()
                completed += 1
                atomic_json(RUN / "capture_status.json", {"stage": "running", "group": group_name,
                                                            "completed": completed, "total": len(SEEDS),
                                                            "last_seed": seed, "last_pieces": row["pieces"], "at": now()})
                print(f"seed={seed} pieces={row['pieces']} dead={row['game_over']}", flush=True)
    rows = [verify_episode(seed, json.loads(episode_path(seed).read_text())) for seed in SEEDS]
    atomic_json(RUN / "capture_summary.json", {"episodes": len(rows),
                                                "candidate_deaths": sum(x["game_over"] for x in rows[:len(TRAIN_SEEDS)]),
                                                "replication_deaths": sum(x["game_over"] for x in rows[len(TRAIN_SEEDS):]),
                                                "candidate_capped": sum(x["capped"] for x in rows[:len(TRAIN_SEEDS)]),
                                                "replication_capped": sum(x["capped"] for x in rows[len(TRAIN_SEEDS):])})
    atomic_json(RUN / "capture_status.json", {"stage": "complete", "completed": len(rows),
                                                "total": len(rows), "at": now()})


if __name__ == "__main__":
    main()
