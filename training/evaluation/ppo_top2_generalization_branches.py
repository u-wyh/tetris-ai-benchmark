"""Paired Top-1/Top-2 continuations on new PPO death and survival seeds."""

import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.hybrid_queue import PPO, digest, load_models, verify_models
from training.evaluation.ppo_fresh_counterfactual import branch
from training.evaluation.ppo_transition_branches import grade, options
from training.evaluation.ppo_transition_capture import PPO_SHA256, public_hash
from training.evaluation.ppo_top2_generalization_audit import audit_capture
from training.evaluation.ppo_top2_generalization_capture import (CAP, DEV_SEEDS, RUN, TRAIN_SEEDS,
                                                                   checkpoint_paths, episode_path,
                                                                   verify_checkpoint, verify_episode)
from training.train_ppo import atomic_bytes, atomic_json, now

VERSION = "top2_generalization_branches_v1"
HORIZON = 10000
MAX_BRANCHES = 48
MAX_SECONDS = 5400


def selected_step(episode):
    if episode["game_over"]:
        return episode["pieces"] - 500 if episode["pieces"] > 500 else None
    return 30000


def state_path(seed):
    return RUN / "selected_states" / f"seed_{seed}.pkl.gz"


def state_manifest_path(seed):
    return RUN / "selected_states" / f"seed_{seed}.json"


def restore_and_select(seed, episode, model):
    target = selected_step(episode)
    if target is None:
        return None
    checkpoint = max((item for item in episode["checkpoints"] if item["step"] <= target),
                     key=lambda item: item["step"])
    checkpoint_path, _ = checkpoint_paths(seed, checkpoint["step"])
    verify_checkpoint(seed, checkpoint["step"], checkpoint["public_sha256"])
    snapshot = pickle.loads(gzip.decompress(checkpoint_path.read_bytes()))
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    selected = None
    try:
        env.reset(seed=seed)
        env.set_state(snapshot)
        while env.pieces < episode["pieces"]:
            if env.pieces == target:
                selected = env.get_state()
            obs, mask = env.get_observation(), env.action_masks()
            action = fast_action(model, obs, mask)
            if not mask[action]:
                raise RuntimeError(f"Illegal replay action: {seed}:{env.pieces}")
            _, _, dead, capped, _ = env.step(action)
            if dead or capped:
                break
        if (selected is None or env.pieces != episode["pieces"]
                or bool(dead) != episode["game_over"] or bool(capped) != episode["capped"]
                or env.core.score != episode["score"]
                or public_hash(env.core.get_public_observation()) != episode["final_public_sha256"]):
            raise RuntimeError(f"Original episode did not replay exactly: {seed}")
        raw = gzip.compress(pickle.dumps(selected, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=3)
        manifest = {"version": VERSION, "seed": seed, "step": target,
                    "group": "candidate" if seed in TRAIN_SEEDS else "replication",
                    "original_game_over": episode["game_over"],
                    "original_episode_sha256": digest(episode_path(seed)),
                    "source_checkpoint_sha256": checkpoint["sha256"],
                    "public_sha256": public_hash(selected["core"].get_public_observation()),
                    "sha256": hashlib.sha256(raw).hexdigest()}
        path, manifest_path = state_path(seed), state_manifest_path(seed)
        if path.exists() and not manifest_path.exists():
            orphan = pickle.loads(gzip.decompress(path.read_bytes()))
            if (orphan["episode_seed"] != seed or orphan["pieces"] != target
                    or public_hash(orphan["core"].get_public_observation()) != manifest["public_sha256"]):
                raise RuntimeError(f"Orphan selected state disagrees with replay: {seed}")
            manifest["sha256"] = digest(path)
            atomic_json(manifest_path, manifest)
        if path.exists() or manifest_path.exists():
            if not path.is_file() or not manifest_path.is_file():
                raise RuntimeError(f"Selected state pair incomplete: {seed}")
            old = json.loads(manifest_path.read_text())
            if old != manifest or digest(path) != old["sha256"]:
                raise RuntimeError(f"Selected state changed: {seed}")
        else:
            atomic_bytes(path, raw)
            atomic_json(manifest_path, manifest)
        return manifest
    finally:
        env.close()


def run_state(seed, episode, model, models):
    completed_path = RUN / "comparisons" / f"seed_{seed}.json"
    if completed_path.exists():
        saved = json.loads(completed_path.read_text())
        manifest = json.loads(state_manifest_path(seed).read_text())
        candidate_path = RUN / "candidates" / f"seed_{seed}.json"
        candidate = json.loads(candidate_path.read_text())
        if (saved["version"] != VERSION or saved["seed"] != seed
                or saved["group"] != ("candidate" if seed in TRAIN_SEEDS else "replication")
                or saved["original_game_over"] != episode["game_over"]
                or manifest["original_episode_sha256"] != digest(episode_path(seed))
                or digest(state_path(seed)) != manifest["sha256"]
                or candidate["state_sha256"] != manifest["sha256"]
                or saved["step"] != manifest["step"]):
            raise RuntimeError(f"Completed state identity changed: {seed}")
        pair = candidate["choices"]
        if (len(pair) != 2 or pair[0]["action"] != saved["top1_action"]
                or pair[1]["action"] != saved["top2_action"]):
            raise RuntimeError(f"Completed candidate pair changed: {seed}")
        outcomes = {}
        for choice in pair:
            action = choice["action"]
            row = json.loads((RUN / "branches" / f"seed_{seed}_action_{action}.json").read_text())
            if (row["seed"] != seed or row["action"] != action or row["step"] != manifest["step"]
                    or row["state_sha256"] != manifest["sha256"]
                    or row["candidate_sha256"] != digest(candidate_path)
                    or row["horizon"] != HORIZON):
                raise RuntimeError(f"Completed branch changed: {seed}:{action}")
            outcomes[action] = row["outcome"]
        if saved["comparison"] != grade(outcomes[pair[0]["action"]], outcomes[pair[1]["action"]]):
            raise RuntimeError(f"Completed comparison changed: {seed}")
        return saved
    manifest = restore_and_select(seed, episode, model)
    if manifest is None:
        return None
    state_file = state_path(seed)
    if digest(state_file) != manifest["sha256"]:
        raise RuntimeError(f"Selected state changed: {seed}")
    snapshot = pickle.loads(gzip.decompress(state_file.read_bytes()))
    choices = options(snapshot, models)
    top2 = next((item for item in choices if "ppo_top2" in item["sources"]), None)
    if top2 is None:
        raise RuntimeError(f"No distinct PPO Top-2 legal action: {seed}")
    pair = [choices[0], top2]
    candidate_path = RUN / "candidates" / f"seed_{seed}.json"
    candidate = {"version": VERSION, "seed": seed, "state_sha256": manifest["sha256"], "choices": pair}
    if candidate_path.exists():
        if json.loads(candidate_path.read_text()) != candidate:
            raise RuntimeError(f"Candidate pair changed: {seed}")
    else:
        atomic_json(candidate_path, candidate)
    outcomes = {}
    for choice in pair:
        action = choice["action"]
        path = RUN / "branches" / f"seed_{seed}_action_{action}.json"
        branch_id = {"version": VERSION, "seed": seed, "step": manifest["step"],
                     "action": action, "horizon": HORIZON, "state_sha256": manifest["sha256"],
                     "candidate_sha256": digest(candidate_path), "ppo_sha256": PPO_SHA256}
        if path.exists():
            saved = json.loads(path.read_text())
            if any(saved.get(key) != value for key, value in branch_id.items()):
                raise RuntimeError(f"Saved branch changed: {seed}:{action}")
        else:
            saved = {**branch_id, "outcome": branch(snapshot, action, models, HORIZON),
                     "completed_at": now()}
            atomic_json(path, saved)
        outcome = saved["outcome"]
        if (outcome["action"] != action or not 0 < outcome["pieces"] <= HORIZON
                or len(outcome["scores"]) != outcome["pieces"]
                or len(outcome["fours"]) != outcome["pieces"]):
            raise RuntimeError(f"Invalid branch outcome: {seed}:{action}")
        outcomes[action] = outcome
    base = outcomes[pair[0]["action"]]
    if episode["game_over"]:
        if not base["game_over"] or base["pieces"] != 500:
            raise RuntimeError(f"Death branch did not match original: {seed}")
    elif base["game_over"] or base["pieces"] != HORIZON:
        raise RuntimeError(f"Survival control did not match original: {seed}")
    result = {"version": VERSION, "seed": seed, "group": manifest["group"],
              "original_game_over": episode["game_over"], "step": manifest["step"],
              "top1_action": pair[0]["action"], "top2_action": pair[1]["action"],
              "top1_probability": pair[0]["ppo_probability"],
              "top2_probability": pair[1]["ppo_probability"],
              "comparison": grade(base, outcomes[pair[1]["action"]]),
              "top2_pieces": outcomes[pair[1]["action"]]["pieces"],
              "top2_game_over": outcomes[pair[1]["action"]]["game_over"],
              "top2_censored": outcomes[pair[1]["action"]]["censored"]}
    path = RUN / "comparisons" / f"seed_{seed}.json"
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise RuntimeError(f"Comparison changed: {seed}")
    else:
        atomic_json(path, result)
    return result


def main():
    torch.set_num_threads(1)
    verify_models()
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO changed")
    capture_status = json.loads((RUN / "capture_status.json").read_text())
    if capture_status["stage"] == "insufficient_candidate_deaths":
        atomic_json(RUN / "branch_status.json", {"stage": "not_started_insufficient_deaths", "at": now()})
        return
    if capture_status["stage"] != "complete":
        raise RuntimeError("Capture incomplete")
    audit = audit_capture()
    if audit["candidate_deaths"] < 3:
        raise RuntimeError("Insufficient audited candidate deaths")
    for name in ("selected_states", "candidates", "branches", "comparisons"):
        (RUN / name).mkdir(exist_ok=True)
    protocol = {"version": VERSION, "seeds": list(TRAIN_SEEDS + DEV_SEEDS),
                "death_offset": 500, "survival_step": 30000,
                "horizon": HORIZON, "max_branches": MAX_BRANCHES, "max_seconds": MAX_SECONDS,
                "ppo_sha256": PPO_SHA256, "capture_protocol_sha256": digest(RUN / "capture_protocol.json"),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    frozen = RUN / "branch_protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Branch protocol changed")
    else:
        atomic_json(frozen, protocol)
    model = MaskablePPO.load(PPO, device="cpu")
    models = load_models()
    started = time.monotonic()
    completed = 0
    for seed in TRAIN_SEEDS + DEV_SEEDS:
        if time.monotonic() - started > MAX_SECONDS:
            atomic_json(RUN / "branch_status.json", {"stage": "budget_stop", "completed": completed, "at": now()})
            return
        episode = verify_episode(seed, json.loads(episode_path(seed).read_text()))
        result = run_state(seed, episode, model, models)
        completed += 1
        branch_count = len(list((RUN / "branches").glob("*.json")))
        if branch_count > MAX_BRANCHES:
            raise RuntimeError("Branch budget exceeded")
        atomic_json(RUN / "branch_status.json", {"stage": "running", "completed": completed,
                                                  "branches": branch_count, "last_seed": seed, "at": now()})
        print(f"seed={seed} group={result['group'] if result else 'excluded'} "
              f"rescued={result['comparison']['rescued_in_window'] if result else False}", flush=True)
    files = {str(path.relative_to(RUN)): digest(path)
             for folder in ("selected_states", "candidates", "branches", "comparisons")
             for path in sorted((RUN / folder).glob("*")) if path.is_file()}
    atomic_json(RUN / "branch_integrity.json", {"files": files, "count": len(files)})
    atomic_json(RUN / "branch_status.json", {"stage": "complete", "completed": completed,
                                             "branches": len(list((RUN / "branches").glob("*.json"))), "at": now()})


if __name__ == "__main__":
    main()
