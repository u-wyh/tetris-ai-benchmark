"""Prospective public-state sampling on new frozen PPO episodes."""

import gzip
import hashlib
import json
import pickle
import random
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import board_features
from training.env.tetris_env import board_max_height
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.hybrid_queue import PPO, digest
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now

RUN = ROOT / "runs/ppo30m_transition_counterfactual"
SEEDS = tuple(range(1530000, 1530008))
CAP = 50000
VERSION = "transition_capture_v1"
PPO_SHA256 = "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34"


def public_hash(public):
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def prospective_stratum(height, rise, step, last_accepted):
    if step < 100 or height < 15:
        return None
    kind = "rapid_stack" if rise >= 5 else "high_control" if rise <= 2 else None
    if kind is None or step - last_accepted[kind] < 200:
        return None
    return kind


def reservoir_add(pool, item, eligible_count, rng, limit=2):
    if len(pool) < limit:
        pool.append(item)
    else:
        index = rng.randrange(eligible_count)
        if index < limit:
            pool[index] = item


def snapshot_path(seed, step):
    return RUN / "states" / f"seed_{seed}_step_{step}.pkl.gz"


def episode_path(seed):
    return RUN / "episodes" / f"seed_{seed}.json"


def verify_episode(seed, row):
    if (row["version"] != VERSION or row["seed"] != seed or row["cap"] != CAP
            or row["ppo_sha256"] != PPO_SHA256 or row["pieces"] > CAP):
        raise RuntimeError(f"Episode identity mismatch: {seed}")
    if len({item["step"] for item in row["states"]}) != len(row["states"]):
        raise RuntimeError(f"Duplicate selected state: {seed}")
    for item in row["states"]:
        path = snapshot_path(seed, item["step"])
        if not path.is_file() or digest(path) != item["snapshot_sha256"]:
            raise RuntimeError(f"Selected snapshot changed: {path}")
    return row


def capture(seed):
    torch.set_num_threads(1)
    path = episode_path(seed)
    if path.exists():
        return verify_episode(seed, json.loads(path.read_text()))
    model = MaskablePPO.load(PPO, device="cpu")
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    rng = random.Random(seed ^ 0x5A17C0DE)
    pools = {"rapid_stack": [], "high_control": []}
    counts = {"rapid_stack": 0, "high_control": 0}
    last_accepted = {"rapid_stack": -10000, "high_control": -10000}
    recent_heights = deque(maxlen=50)
    routine = None
    clears = [0] * 5
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            step = env.pieces
            public = env.core.get_public_observation()
            height = board_max_height(public["board"])
            rise = height - min(recent_heights) if recent_heights else 0
            kind = prospective_stratum(height, rise, step, last_accepted)
            if kind or step == 5000:
                item = {"step": step, "stratum": kind or "routine_control",
                        "height": height, "rise_50": rise,
                        "holes": board_features(public["board"])[0],
                        "public_sha256": public_hash(public), "snapshot": env.get_state()}
                if kind:
                    counts[kind] += 1
                    last_accepted[kind] = step
                    reservoir_add(pools[kind], item, counts[kind], rng)
                if step == 5000:
                    routine = item
            mask = env.action_masks()
            if step < 100 or step % 10000 == 0:
                if env._placement_cache != env.core.get_legal_placements():
                    raise RuntimeError(f"Fast/legal placement disagreement: {seed}:{step}")
            action = fast_action(model, obs, mask)
            if not mask[action]:
                raise RuntimeError(f"Illegal PPO action: {seed}:{step}")
            obs, _, terminated, truncated, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            recent_heights.append(board_max_height(env.core.board))
            if terminated or truncated:
                break
        selected = sorted((pools["rapid_stack"] + pools["high_control"] +
                           ([] if routine is None else [routine])), key=lambda item: item["step"])
        by_step = {}
        for item in selected:
            if item["step"] not in by_step or item["stratum"] != "routine_control":
                by_step[item["step"]] = item
        states = []
        for step, item in sorted(by_step.items()):
            snapshot = item.pop("snapshot")
            raw = gzip.compress(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=3)
            snap_path = snapshot_path(seed, step)
            atomic_bytes(snap_path, raw)
            states.append({**item, "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                           "inclusion_probability": (min(1.0, 2 / counts[item["stratum"]])
                                                     if item["stratum"] in counts else 1.0)})
        row = {"version": VERSION, "seed": seed, "cap": CAP, "ppo_sha256": PPO_SHA256,
               "pieces": env.pieces, "game_over": bool(terminated), "capped": bool(truncated),
               "score": env.core.score, "clears_0_to_4": clears,
               "eligible_counts": counts, "states": states, "completed_at": now()}
        atomic_json(path, row)
        return verify_episode(seed, row)
    finally:
        env.close()


def main():
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO checkpoint changed")
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "states").mkdir(exist_ok=True)
    (RUN / "episodes").mkdir(exist_ok=True)
    protocol = {"version": VERSION, "seeds": list(SEEDS), "cap": CAP,
                "max_states_per_seed": 5, "rapid_stack": "height>=15,rise_50>=5,step>=100,spacing>=200",
                "high_control": "height>=15,rise_50<=2,step>=100,spacing>=200",
                "routine_step": 5000, "per_high_stratum_reservoir": 2,
                "rng": "python.Random(seed xor 0x5A17C0DE)", "ppo_sha256": PPO_SHA256,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    frozen = RUN / "capture_protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Capture protocol changed")
    else:
        pattern = (r'"(?:seed|base_seed|run_seed|episode_seed)"\s*:\s*'
                   r'153(?:000[0-7]|10(?:0[0-9]|1[0-5]))\b|'
                   r'seed_153(?:000[0-7]|10(?:0[0-9]|1[0-5]))\b')
        historical_root = Path("/home/wyh/projects/tetris-ai-benchmark")
        audit = subprocess.run(["rg", "-n", "-P", pattern, "training", "tests", "reports", "runs",
                                "--glob", "*.json", "--glob", "*.md", "--glob", "*.py",
                                "--glob", "*.txt", "--glob", "*.log"],
                               cwd=historical_root, text=True, capture_output=True, check=False)
        if audit.returncode != 1:
            raise RuntimeError(f"Exact-seed audit failed: {audit.stdout[:300]} {audit.stderr[:300]}")
        atomic_json(frozen, protocol)
        atomic_json(RUN / "seed_audit.json", {"seeds": list(SEEDS),
                                                  "reserved_confirmation": list(range(1531000, 1531016)),
                                                  "exact_seed_collision_count": 0,
                                                  "searched_root": str(historical_root),
                                                  "pattern": pattern,
                                                  "audited_at": now()})
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(capture, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            atomic_json(RUN / "capture_status.json", {"stage": "running", "at": now(),
                                                       "completed": sum(f.done() and not f.exception()
                                                                        for f in futures),
                                                       "total": len(SEEDS), "last_seed": seed,
                                                       "last_states": len(result["states"])})
    rows = [verify_episode(seed, json.loads(episode_path(seed).read_text())) for seed in SEEDS]
    atomic_json(RUN / "capture_summary.json", {"episodes": len(rows),
                                                "deaths": sum(r["game_over"] for r in rows),
                                                "capped": sum(r["capped"] for r in rows),
                                                "states": sum(len(r["states"]) for r in rows),
                                                "rapid_seed_count": sum(bool(r["eligible_counts"]["rapid_stack"])
                                                                        for r in rows),
                                                "high_control_seed_count": sum(bool(r["eligible_counts"]["high_control"])
                                                                               for r in rows)})
    atomic_json(RUN / "capture_status.json", {"stage": "complete", "at": now(),
                                               "completed": len(rows), "total": len(rows)})


if __name__ == "__main__":
    main()
