"""Replay frozen development episodes to audit every prospective selection decision."""

import json
import sys
from collections import deque

import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import board_features
from training.env.tetris_env import board_max_height
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.hybrid_queue import PPO, HASHES, digest
from training.evaluation.ppo_clear_cadence_capture import CAP, RUN, SEEDS, episode_path
from training.train_ppo import atomic_json


def verify_saved_audit(saved, seed, expected, episode_digest):
    groups = ("instability", "recovered_control")
    keys = ("eligible", "spacing_excluded", "quota_excluded")
    steps = [state["step"] for state in expected["states"]]
    if (set(saved) != {"seed", "episode_sha256", "ppo_sha256", "passed", "counts", "selected_steps"}
            or type(saved["seed"]) is not int or saved["seed"] != seed
            or saved["episode_sha256"] != episode_digest
            or saved["ppo_sha256"] != HASHES["ppo"]
            or saved["passed"] is not True
            or saved["selected_steps"] != steps
            or set(saved["counts"]) != set(groups)
            or saved["counts"] != expected["counts"]):
        raise RuntimeError(f"stale selection audit: {seed}")
    for group in groups:
        counts = saved["counts"][group]
        if (set(counts) != set(keys)
                or any(type(counts[key]) is not int or counts[key] < 0 for key in keys)
                or counts["eligible"] - counts["spacing_excluded"] - counts["quota_excluded"]
                != sum(state["stratum"] == group for state in expected["states"])):
            raise RuntimeError(f"stale selection audit: {seed}")
    return saved


def audit_seed(seed, model):
    if seed not in SEEDS:
        raise ValueError("seed outside frozen development group")
    expected = json.loads(episode_path(seed).read_text())
    output = RUN / "selection_audit" / f"seed_{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        return verify_saved_audit(saved, seed, expected, digest(episode_path(seed)))
    recent_fours, recent_heights = deque(maxlen=200), deque(maxlen=100)
    states = []
    counts = {group: {key: 0 for key in ("eligible", "spacing_excluded", "quota_excluded")}
              for group in ("instability", "recovered_control")}
    clears = [0] * 5
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=seed)
        dead = capped = False
        while env.pieces < CAP:
            step = env.pieces
            if step >= 250 and step % 25 == 0:
                public = env.core.get_public_observation()
                height = board_max_height(public["board"])
                holes = board_features(public["board"])[0]
                fours = list(recent_fours)
                last_fours, prior_fours = sum(fours[-100:]), sum(fours[:100])
                peak = max(recent_heights, default=0)
                group = None
                if prior_fours >= 8 and last_fours <= 5 and (height >= 13 or holes >= 2):
                    group = "instability"
                elif peak >= 15 and height <= 10 and holes <= 1 and last_fours >= 7:
                    group = "recovered_control"
                if group:
                    counts[group]["eligible"] += 1
                    if sum(s["stratum"] == group for s in states) >= 2:
                        counts[group]["quota_excluded"] += 1
                    elif any(step - s["step"] < 500 for s in states):
                        counts[group]["spacing_excluded"] += 1
                    else:
                        states.append({"step": step, "stratum": group, "height": height,
                                       "holes": holes, "prior_fours": prior_fours,
                                       "last_fours": last_fours, "recent_height_peak": peak})
            action = fast_action(model, env.get_observation(), env.action_masks())
            _, _, dead, capped, info = env.step(action)
            clear = info["cleared_lines"]
            clears[clear] += 1
            recent_fours.append(clear == 4)
            recent_heights.append(board_max_height(env.core.board))
            if dead or capped:
                break
        actual = {"pieces": env.pieces, "game_over": bool(dead), "capped": bool(capped),
                  "score": env.core.score, "clears": clears, "counts": counts}
        for key, value in actual.items():
            if value != expected[key]:
                raise AssertionError(f"seed {seed} {key}: replay {value} != archive {expected[key]}")
        archived_states = [{k: s[k] for k in row} for row in states for s in expected["states"]
                           if s["step"] == row["step"]]
        if states != archived_states or len(states) != len(expected["states"]):
            raise AssertionError(f"seed {seed}: selected states differ")
        row = {"seed": seed, "episode_sha256": digest(episode_path(seed)),
               "ppo_sha256": HASHES["ppo"], "passed": True, "counts": counts,
               "selected_steps": [s["step"] for s in states]}
        atomic_json(output, row)
        return row
    finally:
        env.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m training.evaluation.audit_clear_cadence_selection SEED")
    if digest(PPO) != HASHES["ppo"]:
        raise RuntimeError("frozen PPO model changed")
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(PPO), device="cpu")
    print(json.dumps(audit_seed(int(sys.argv[1]), model)), flush=True)
