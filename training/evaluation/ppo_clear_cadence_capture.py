"""Prospective clear-cadence sampling under the frozen PPO protocol."""

import gzip
import hashlib
import json
import pickle
from collections import deque

import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import board_features
from training.env.tetris_env import board_max_height
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.hybrid_queue import PPO, HASHES, digest
from training.evaluation.source_provenance import snapshot as source_snapshot, verify as verify_source
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now

RUN = ROOT / "runs/ppo_clear_cadence_counterfactual"
DESIGN = ROOT / "reports/experiments/ppo_clear_cadence_counterfactual_design.md"
SEEDS = tuple(range(1550000, 1550008))
CAP = 20000
VERSION = "clear_cadence_capture_v1"


def state_path(seed, step):
    return RUN / "states" / f"seed_{seed}_step_{step}.pkl.gz"


def episode_path(seed):
    return RUN / "episodes" / f"seed_{seed}.json"


def verify_episode(seed, row):
    if (row["version"] != VERSION or row["seed"] != seed or row["cap"] != CAP
            or row["ppo_sha256"] != HASHES["ppo"]
            or row["design_sha256"] != digest(DESIGN)):
        raise RuntimeError(f"Episode identity mismatch: {seed}")
    if "execution_provenance" in row:
        verify_source(ROOT, row["execution_provenance"])
    if len(row["states"]) > 4 or len({s["step"] for s in row["states"]}) != len(row["states"]):
        raise RuntimeError(f"Invalid state count: {seed}")
    for state in row["states"]:
        path = state_path(seed, state["step"])
        if not path.is_file() or digest(path) != state["snapshot_sha256"]:
            raise RuntimeError(f"Snapshot mismatch: {path}")
    return row


def capture(seed):
    if seed not in SEEDS:
        raise ValueError("Seed outside frozen development group")
    if digest(PPO) != HASHES["ppo"]:
        raise RuntimeError("Frozen PPO model changed")
    path = episode_path(seed)
    if path.exists():
        return verify_episode(seed, json.loads(path.read_text()))
    provenance = source_snapshot(ROOT)
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(PPO), device="cpu")
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    states = []
    counts = {name: {"eligible": 0, "spacing_excluded": 0, "quota_excluded": 0}
              for name in ("instability", "recovered_control")}
    recent_fours = deque(maxlen=200)
    recent_heights = deque(maxlen=100)
    clears = [0] * 5
    dead = capped = False
    try:
        env.reset(seed=seed)
        while env.pieces < CAP:
            step = env.pieces
            if step >= 250 and step % 25 == 0:
                public = env.core.get_public_observation()
                height = board_max_height(public["board"])
                holes = board_features(public["board"])[0]
                last_fours = sum(list(recent_fours)[-100:])
                prior_fours = sum(list(recent_fours)[:100])
                group = None
                if prior_fours >= 8 and last_fours <= 5 and (height >= 13 or holes >= 2):
                    group = "instability"
                elif max(recent_heights, default=0) >= 15 and height <= 10 and holes <= 1 and last_fours >= 7:
                    group = "recovered_control"
                if group:
                    counts[group]["eligible"] += 1
                    if sum(s["stratum"] == group for s in states) >= 2:
                        counts[group]["quota_excluded"] += 1
                    elif any(step - s["step"] < 500 for s in states):
                        counts[group]["spacing_excluded"] += 1
                    else:
                        snapshot = env.get_state()
                        payload = gzip.compress(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=3)
                        snapshot_file = state_path(seed, step)
                        if snapshot_file.exists():
                            if digest(snapshot_file) != hashlib.sha256(payload).hexdigest():
                                raise RuntimeError(f"Orphan snapshot mismatch: {snapshot_file}")
                        else:
                            atomic_bytes(snapshot_file, payload)
                        states.append({"step": step, "stratum": group, "height": height,
                                       "holes": holes, "prior_fours": prior_fours,
                                       "last_fours": last_fours,
                                       "recent_height_peak": max(recent_heights, default=0),
                                       "snapshot_sha256": digest(snapshot_file)})
            obs, mask = env.get_observation(), env.action_masks()
            action = fast_action(model, obs, mask)
            _, _, dead, capped, info = env.step(action)
            clear = info["cleared_lines"]
            clears[clear] += 1
            recent_fours.append(int(clear == 4))
            recent_heights.append(board_max_height(env.core.board))
            if dead or capped:
                break
        row = {"version": VERSION, "design_sha256": digest(DESIGN), "ppo_sha256": HASHES["ppo"],
               "execution_provenance": provenance,
               "seed": seed, "cap": CAP, "pieces": env.pieces, "game_over": bool(dead),
               "capped": bool(capped), "score": env.core.score, "clears": clears,
               "counts": counts, "states": states, "completed_at": now()}
        verify_source(ROOT, provenance)
        atomic_json(path, row)
        return verify_episode(seed, row)
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m training.evaluation.ppo_clear_cadence_capture SEED")
    print(json.dumps(capture(int(sys.argv[1])), ensure_ascii=False), flush=True)
