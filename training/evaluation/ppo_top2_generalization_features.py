"""Public, pre-action features for Top-1/Top-2 comparison states."""

import gzip
import hashlib
import json
import pickle
from pathlib import Path

from training.candidate_features import board_features
from training.evaluation.hybrid_queue import digest
from training.evaluation.ppo_top2_generalization_capture import DEV_SEEDS, RUN, TRAIN_SEEDS
from training.evaluation.ppo_top2_generalization_branches import state_manifest_path, state_path
from training.evaluation.ppo_transition_capture import public_hash
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public
from training.train_ppo import atomic_json, now


def board_values(board):
    holes, aggregate, height, bumpiness, wells = board_features(board)
    return {"holes": holes, "aggregate_height": aggregate, "max_height": height,
            "bumpiness": bumpiness, "wells": wells}


def main():
    if json.loads((RUN / "branch_status.json").read_text())["stage"] != "complete":
        raise RuntimeError("Paired branch study incomplete")
    if not (RUN / "analysis.json").is_file():
        raise RuntimeError("Seed-level analysis incomplete")
    rows = []
    source_hashes = {}
    for seed in TRAIN_SEEDS + DEV_SEEDS:
        manifest_path = state_manifest_path(seed)
        if not manifest_path.exists():
            continue  # death before the frozen 500-block offset
        manifest = json.loads(manifest_path.read_text())
        path = state_path(seed)
        if digest(path) != manifest["sha256"]:
            raise RuntimeError(f"Selected state changed: {seed}")
        state = pickle.loads(gzip.decompress(path.read_bytes()))  # local, hashed snapshot
        public = state["core"].get_public_observation()
        if public_hash(public) != manifest["public_sha256"]:
            raise RuntimeError(f"Public state changed: {seed}")
        choices_path = RUN / "candidates" / f"seed_{seed}.json"
        comparison_path = RUN / "comparisons" / f"seed_{seed}.json"
        choices = json.loads(choices_path.read_text())["choices"]
        outcome = json.loads(comparison_path.read_text())
        placements = {row["actionId"]: row for row in get_legal_placements(public)}
        action_features = []
        for choice in choices:
            action = choice["action"]
            if action not in placements:
                raise RuntimeError(f"Official legal action disappeared: {seed}:{action}")
            child, cleared = simulate_public(public, placements[action])
            action_features.append({"action": action, "ppo_probability": choice["ppo_probability"],
                                    "hold": choice["hold"], "cleared_lines": cleared,
                                    "game_over_after": child["phase"] == "over",
                                    "board_after": board_values(child["board"])})
        if len(action_features) != 2 or action_features[0]["action"] != outcome["top1_action"]:
            raise RuntimeError(f"Candidate pair mismatch: {seed}")
        rows.append({"seed": seed, "group": outcome["group"], "original_game_over": outcome["original_game_over"],
                     "step": manifest["step"], "current": public["currentPiece"]["type"],
                     "hold_piece": public["hold"], "next3": list(public["next"])[:3],
                     "board_before": board_values(public["board"]),
                     "top1": action_features[0], "top2": action_features[1],
                     "top2_rescued": outcome["comparison"]["rescued_in_window"],
                     "top2_harmed": outcome["comparison"]["harmed_in_window"],
                     "top2_true_death": outcome["top2_game_over"],
                     "score_ratio_common": outcome["comparison"]["score_ratio"],
                     "four_delta_common": (outcome["comparison"]["candidate_fours"]
                                           - outcome["comparison"]["ppo_fours"])})
        source_hashes[f"seed_{seed}_state"] = digest(path)
        source_hashes[f"seed_{seed}_comparison"] = digest(comparison_path)
    atomic_json(RUN / "public_feature_table.json", {"version": "top2_public_features_v1",
                                                  "created_at": now(), "rows": rows,
                                                  "source_sha256": source_hashes,
                                                  "feature_source_sha256": hashlib.sha256(
                                                      Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
