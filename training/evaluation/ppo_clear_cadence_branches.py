"""Resume the frozen clear-cadence paired branches, one state at a time."""

import gzip
import json
import pickle
import sys

import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import board_features
from training.evaluation.fast_counterfactual import fast_branch
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.hybrid_queue import HASHES, PPO, digest
from training.evaluation.ppo_clear_cadence_capture import RUN, SEEDS, DESIGN, state_path, episode_path, verify_episode
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public
from training.train_ppo import atomic_json, now

VERSION = "clear_cadence_branches_v1"
HORIZON = 3000


def candidates(snapshot):
    env = LongCandidateTetrisEnv(max_pieces=snapshot["max_pieces"])
    try:
        env.reset(seed=snapshot["episode_seed"])
        env.set_state(snapshot)
        obs, mask = env.get_observation(), env.action_masks()
        public = env.core.get_public_observation()
        placements = {p["actionId"]: p for p in get_legal_placements(public)}
        model = candidates.model
        original = fast_action(model, obs, mask)
        if original not in placements or not mask[original]:
            raise RuntimeError("PPO action not legal")
        tensor = {k: torch.as_tensor(v, device="cpu").unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            probs = model.policy.get_distribution(tensor, action_masks=torch.as_tensor(mask).unsqueeze(0)).distribution.probs[0].cpu().numpy()
        ranked = []
        for action, placement in placements.items():
            if not mask[action]:
                raise RuntimeError("Placement and mask disagree")
            child, clear = simulate_public(public, placement)
            holes, _, height, _, _ = board_features(child["board"])
            key = [int(child["phase"] == "over"), -int(clear), int(height), int(holes), -float(probs[action]), int(action)]
            ranked.append({"action": int(action), "rank_key": key, "ppo_probability": float(probs[action])})
        ranked.sort(key=lambda r: r["rank_key"])
        alternatives = [r for r in ranked if r["action"] != original]
        return {"version": VERSION, "original_action": original,
                "alternative_action": alternatives[0]["action"] if alternatives else None,
                "ranked_legal_actions": ranked}
    finally:
        env.close()


def run_state(seed, item):
    path = state_path(seed, item["step"])
    if digest(path) != item["snapshot_sha256"]:
        raise RuntimeError("Snapshot checksum mismatch")
    snapshot = pickle.loads(gzip.decompress(path.read_bytes()))
    if snapshot["episode_seed"] != seed or snapshot["pieces"] != item["step"]:
        raise RuntimeError("Snapshot identity mismatch")
    choice_path = RUN / "candidates" / f"seed_{seed}_step_{item['step']}.json"
    choice = {"design_sha256": digest(DESIGN), "ppo_sha256": HASHES["ppo"],
              "snapshot_sha256": item["snapshot_sha256"], **candidates(snapshot)}
    if choice_path.exists():
        if json.loads(choice_path.read_text()) != choice:
            raise RuntimeError("Candidate drift")
    else:
        atomic_json(choice_path, choice)
    if choice["alternative_action"] is None:
        return {"seed": seed, "step": item["step"], "skipped": "no alternative"}
    results = {}
    for action in (choice["original_action"], choice["alternative_action"]):
        out = RUN / "branches" / f"seed_{seed}_step_{item['step']}_action_{action}.json"
        identity = {"version": VERSION, "candidate_sha256": digest(choice_path),
                    "snapshot_sha256": item["snapshot_sha256"], "ppo_sha256": HASHES["ppo"],
                    "action": action, "horizon": HORIZON}
        if out.exists():
            row = json.loads(out.read_text())
            if any(row.get(k) != v for k, v in identity.items()):
                raise RuntimeError("Branch identity mismatch")
        else:
            row = {**identity, "outcome": fast_branch(snapshot, action, {"ppo": candidates.model}, HORIZON),
                   "completed_at": now()}
            atomic_json(out, row)
        results[action] = row["outcome"]
    base, alt = (results[choice[k]] for k in ("original_action", "alternative_action"))
    common = min(base["pieces"], alt["pieces"])
    summary = {"seed": seed, "step": item["step"], "stratum": item["stratum"],
               "original_action": choice["original_action"], "alternative_action": choice["alternative_action"],
               "original_pieces": base["pieces"], "alternative_pieces": alt["pieces"],
               "original_game_over": base["game_over"], "alternative_game_over": alt["game_over"],
               "common_pieces": common,
               "original_score": base["scores"][common - 1], "alternative_score": alt["scores"][common - 1],
               "original_fours": base["fours"][common - 1], "alternative_fours": alt["fours"][common - 1],
               "rescue": base["game_over"] and alt["pieces"] - base["pieces"] >= 256,
               "harm": alt["game_over"] and base["pieces"] - alt["pieces"] >= 256}
    output = RUN / "comparisons" / f"seed_{seed}_step_{item['step']}.json"
    if output.exists():
        if json.loads(output.read_text()) != summary:
            raise RuntimeError("Comparison drift")
    else:
        atomic_json(output, summary)
    return summary


def main(seed):
    if seed not in SEEDS or digest(PPO) != HASHES["ppo"]:
        raise RuntimeError("Seed or model identity mismatch")
    episode = verify_episode(seed, json.loads(episode_path(seed).read_text()))
    torch.set_num_threads(1)
    candidates.model = MaskablePPO.load(str(PPO), device="cpu")
    for item in episode["states"]:
        print(json.dumps(run_state(seed, item)), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
