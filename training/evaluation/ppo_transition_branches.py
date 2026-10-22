"""Bounded same-state alternatives for prospectively selected PPO transitions."""

import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from training.candidate_features import board_features
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import BC, HASHES, PPO, digest, load_models, verify_models
from training.evaluation.ppo_fresh_counterfactual import branch
from training.evaluation.ppo_multiaction_counterfactual import compare
from training.evaluation.ppo_transition_capture import (CAP, PPO_SHA256, RUN, SEEDS,
                                                          public_hash,
                                                          snapshot_path, verify_episode)
from training.evaluation.ppo_transition_audit import corrected_episode_path
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public
from training.train_ppo import atomic_json, now

VERSION = "transition_branch_v1"
MAX_SECONDS = 5400
MAX_BRANCHES = 200


def horizon(item):
    return 3000 if item["stratum"] == "rapid_stack" else 1000


def options(snapshot, models):
    env = FastCandidateTetrisEnv(max_pieces=snapshot["max_pieces"])
    try:
        env.reset(seed=snapshot["episode_seed"])
        env.set_state(snapshot)
        obs, mask = env.get_observation(), env.action_masks()
        public = env.core.get_public_observation()
        placements = {row["actionId"]: row for row in get_legal_placements(public)}
        if not placements or set(placements) != set(np.flatnonzero(mask)):
            raise RuntimeError("Candidate set disagrees with official action mask")
        tensor_obs = {key: torch.as_tensor(value, device="cpu").unsqueeze(0)
                      for key, value in obs.items()}
        with torch.no_grad():
            probs = models["ppo"].policy.get_distribution(
                tensor_obs, action_masks=torch.as_tensor(mask).unsqueeze(0)).distribution.probs[0].cpu().numpy()
        ranked = sorted(placements, key=lambda action: (-float(probs[action]), action))
        ppo = int(models["ppo"].predict(obs, deterministic=True, action_masks=mask)[0])
        if ranked[0] != ppo:
            raise RuntimeError("PPO top-1 mismatch")
        bc = int(models["bc"].predict(obs, deterministic=True, action_masks=mask)[0])
        safety, four = {}, {}
        for action, placement in placements.items():
            child, cleared = simulate_public(public, placement)
            holes, _, height, _, _ = board_features(child["board"])
            safety[action] = (child["phase"] == "over", height, holes, action)
            four[action] = int(cleared == 4)
        sources = [(ppo, "ppo")]
        if len(ranked) > 1:
            sources.append((ranked[1], "ppo_top2"))
        sources.extend(((bc, "bc"),
                        (min(placements, key=lambda action: safety[action]), "public_safe"),
                        (min(placements, key=lambda action: (-four[action], safety[action])), "immediate_four")))
        unique = {}
        for action, source in sources:
            if action not in unique:
                unique[action] = {"action": action, "sources": [],
                                  "ppo_rank": ranked.index(action) + 1,
                                  "ppo_probability": float(probs[action]),
                                  "hold": bool(placements[action]["hold"]),
                                  "immediate_four": four[action],
                                  "public_safety": list(safety[action][:3])}
            unique[action]["sources"].append(source)
        return list(unique.values())
    finally:
        env.close()


def grade(base, other):
    metric = compare(base, other)
    rescued = (base["game_over"] and other["pieces"] - base["pieces"] >= 256
               and other["pieces"] >= 1.5 * base["pieces"]
               and metric["score_ratio"] >= .90)
    harmed = (other["game_over"] and base["pieces"] - other["pieces"] >= 256
              and base["pieces"] >= 1.5 * other["pieces"])
    return {**metric, "rescued_in_window": bool(rescued), "harmed_in_window": bool(harmed),
            "ppo_right_censored": bool(base["censored"]),
            "alternative_right_censored": bool(other["censored"])}


def process(seed, item, episode, models):
    step = item["step"]
    path = snapshot_path(seed, step)
    if digest(path) != item["snapshot_sha256"]:
        raise RuntimeError("Selected snapshot checksum changed")
    snapshot = pickle.loads(gzip.decompress(path.read_bytes()))  # locally generated and SHA256 verified
    if snapshot["episode_seed"] != seed or snapshot["pieces"] != step or snapshot["max_pieces"] != CAP:
        raise RuntimeError("Snapshot state identity mismatch")
    public = snapshot["core"].get_public_observation()
    if public_hash(public) != item["public_sha256"]:
        raise RuntimeError("Snapshot public observation changed")
    choices = options(snapshot, models)
    choice_path = RUN / "candidates" / f"seed_{seed}_step_{step}.json"
    candidate = {"version": VERSION, "seed": seed, "step": step,
                 "snapshot_sha256": item["snapshot_sha256"], "model_sha256": HASHES,
                 "choices": choices}
    if choice_path.exists():
        if json.loads(choice_path.read_text()) != candidate:
            raise RuntimeError("Candidate choices changed on resume")
    else:
        atomic_json(choice_path, candidate)
    candidate_sha = digest(choice_path)
    results = {}
    window = horizon(item)
    for choice in choices:
        action = choice["action"]
        result_path = RUN / "branches" / f"seed_{seed}_step_{step}_action_{action}.json"
        identity = {"version": VERSION, "seed": seed, "step": step, "action": action,
                    "horizon": window, "snapshot_sha256": item["snapshot_sha256"],
                    "candidate_sha256": candidate_sha, "model_sha256": HASHES}
        if result_path.exists():
            row = json.loads(result_path.read_text())
            if any(row.get(key) != value for key, value in identity.items()):
                raise RuntimeError(f"Saved branch identity changed: {result_path}")
            outcome = row["outcome"]
            if (outcome["action"] != action or outcome["horizon"] != window
                    or not 0 < outcome["pieces"] <= window
                    or len(outcome["scores"]) != outcome["pieces"]
                    or len(outcome["fours"]) != outcome["pieces"]
                    or (outcome["censored"] and outcome["pieces"] != window)):
                raise RuntimeError(f"Saved branch outcome invalid: {result_path}")
        else:
            outcome = branch(snapshot, action, models, window)
            atomic_json(result_path, {**identity, "outcome": outcome, "completed_at": now()})
        results[action] = outcome
    base = results[choices[0]["action"]]
    if episode["game_over"] and episode["pieces"] - step <= window:
        if not base["game_over"] or base["pieces"] != episode["pieces"] - step:
            raise RuntimeError(f"Original PPO death did not replay: {seed}:{step}")
    comparison = {"version": VERSION, "seed": seed, "step": step,
                  "stratum": item["stratum"], "horizon": window,
                  "candidate_sha256": candidate_sha,
                  "ppo_continuation_pieces": base["pieces"],
                  "ppo_game_over": base["game_over"], "ppo_right_censored": base["censored"],
                  "comparisons": {str(choice["action"]): grade(base, results[choice["action"]])
                                  for choice in choices[1:]}}
    out = RUN / "comparisons" / f"seed_{seed}_step_{step}.json"
    if out.exists():
        if json.loads(out.read_text()) != comparison:
            raise RuntimeError("Comparison changed on resume")
    else:
        atomic_json(out, comparison)
    return comparison


def main():
    verify_models()
    if digest(PPO) != PPO_SHA256 or digest(BC) != HASHES["bc"]:
        raise RuntimeError("Frozen model identity changed")
    capture_status = json.loads((RUN / "capture_status.json").read_text())
    if capture_status["stage"] != "complete" or capture_status["completed"] != len(SEEDS):
        raise RuntimeError("All prospective episodes must be committed before branching")
    capture = json.loads((RUN / "capture_summary.json").read_text())
    if capture["episodes"] != len(SEEDS):
        raise RuntimeError("Prospective capture summary is incomplete")
    if capture["rapid_seed_count"] < 3:
        atomic_json(RUN / "branch_status.json", {"stage": "insufficient_rapid_seed_coverage",
                                                 "at": now(), "rapid_seeds": capture["rapid_seed_count"]})
        return
    protocol = {"version": VERSION, "seeds": list(SEEDS), "models": HASHES,
                "capture_protocol_sha256": digest(RUN / "capture_protocol.json"),
                "horizon": {"rapid_stack": 3000, "other": 1000},
                "max_branches": MAX_BRANCHES, "wall_seconds": MAX_SECONDS,
                "candidate_order": "PPO top1/top2, BC, public safety, immediate four; deduplicated",
                "rescued": "PPO dies; alternative survives >=256 longer and >=1.5x; common score >=90%",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    frozen = RUN / "branch_protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Branch protocol changed")
    else:
        atomic_json(frozen, protocol)
    for name in ("candidates", "branches", "comparisons"):
        (RUN / name).mkdir(exist_ok=True)
    models = load_models()
    integrity = json.loads((RUN / "capture_integrity.json").read_text())
    if integrity["episodes"] != len(SEEDS):
        raise RuntimeError("Capture integrity audit incomplete")
    started = time.monotonic()
    states = 0
    for seed in SEEDS:
        corrected = corrected_episode_path(seed)
        if hashlib.sha256(corrected.read_bytes()).hexdigest() != integrity["file_sha256"][str(corrected.relative_to(RUN))]:
            raise RuntimeError(f"Corrected episode checksum changed: {seed}")
        episode = verify_episode(seed, json.loads(corrected.read_text()))
        for item in episode["states"]:
            if time.monotonic() - started > MAX_SECONDS:
                atomic_json(RUN / "branch_status.json", {"stage": "budget_stop", "at": now(),
                                                         "completed_states": states})
                return
            process(seed, item, episode, models)
            states += 1
            branch_count = len(list((RUN / "branches").glob("*.json")))
            if branch_count > MAX_BRANCHES:
                raise RuntimeError("Branch budget exceeded")
            atomic_json(RUN / "branch_status.json", {"stage": "running", "at": now(),
                                                     "completed_states": states,
                                                     "completed_branches": branch_count,
                                                     "last_seed": seed, "last_step": item["step"]})
            print(f"state {states}: seed={seed} step={item['step']} branches={branch_count}", flush=True)
    files = {}
    for folder in ("episodes", "states", "candidates", "branches", "comparisons"):
        for path in sorted((RUN / folder).glob("*")):
            if path.is_file():
                files[str(path.relative_to(RUN))] = digest(path)
    atomic_json(RUN / "integrity.json", {"count": len(files), "sha256": files})
    rows = [json.loads(path.read_text()) for path in sorted((RUN / "comparisons").glob("*.json"))]
    atomic_json(RUN / "analysis.json", {"states": len(rows),
                                        "branches": len(list((RUN / "branches").glob("*.json"))),
                                        "comparisons": sum(len(row["comparisons"]) for row in rows),
                                        "rapid_states": sum(row["stratum"] == "rapid_stack" for row in rows),
                                        "rapid_seeds": len({row["seed"] for row in rows
                                                            if row["stratum"] == "rapid_stack"}),
                                        "rescue_states": sum(any(x["rescued_in_window"]
                                                                 for x in row["comparisons"].values())
                                                             for row in rows),
                                        "harm_states": sum(any(x["harmed_in_window"]
                                                               for x in row["comparisons"].values())
                                                           for row in rows),
                                        "ppo_censored_states": sum(row["ppo_right_censored"] for row in rows)})
    atomic_json(RUN / "branch_status.json", {"stage": "complete", "at": now(),
                                             "completed_states": states,
                                             "completed_branches": len(list((RUN / "branches").glob("*.json")))})


if __name__ == "__main__":
    main()
