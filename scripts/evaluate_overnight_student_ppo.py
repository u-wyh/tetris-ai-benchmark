"""Measure paired development performance for an overnight PPO checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import FEATURE_NAMES
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.train_ppo import atomic_json


HEIGHT_COLUMN = FEATURE_NAMES.index("max_height_before")


def game(model, seed, cap):
    env = LongCandidateTetrisEnv(max_pieces=cap)
    env.reset(seed=seed)
    scores = [0]
    high = 0
    try:
        for _ in range(cap):
            observation, legal = env.get_observation(), env.action_masks()
            action = fast_action(model, observation, legal)
            if not legal[action]:
                raise RuntimeError("Evaluation actor chose an illegal action")
            high += int(observation["candidate"][action, HEIGHT_COLUMN] >= 12)
            _, _, over, truncated, info = env.step(int(action))
            scores.append(int(info["score"]))
            if over or truncated:
                break
        return scores, {"pieces": len(scores) - 1, "over": bool(over), "height_ge_12": high}
    finally:
        env.close()


def evaluate(reference, candidate, seeds, cap):
    torch.set_num_threads(1)
    student = MaskablePPO.load(reference, device="cpu")
    trained = MaskablePPO.load(candidate, device="cpu")
    pairs = []
    for seed in seeds:
        old_scores, old = game(student, seed, cap)
        new_scores, new = game(trained, seed, cap)
        shared = min(old["pieces"], new["pieces"])
        pairs.append({
            "seed": seed, "shared_pieces": shared,
            "reference": {**old, "common_score": old_scores[shared]},
            "candidate": {**new, "common_score": new_scores[shared]},
        })
    refs, candidates = ([row[role] for row in pairs] for role in ("reference", "candidate"))
    reference_score = sum(r["common_score"] for r in refs)
    return {
        "cap": cap, "pairs": pairs,
        "summary": {
            "common_score_ratio": sum(r["common_score"] for r in candidates) / reference_score
            if reference_score else None,
            "reference_deaths": sum(r["over"] for r in refs),
            "candidate_deaths": sum(r["over"] for r in candidates),
            "reference_pieces": sum(r["pieces"] for r in refs),
            "candidate_pieces": sum(r["pieces"] for r in candidates),
            "reference_high_fraction": sum(r["height_ge_12"] for r in refs) / sum(r["pieces"] for r in refs),
            "candidate_high_fraction": sum(r["height_ge_12"] for r in candidates) / sum(r["pieces"] for r in candidates),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cap", type=int, default=4096)
    args = parser.parse_args()
    if args.cap < 1:
        parser.error("--cap must be positive")
    result = evaluate(args.reference, args.candidate, args.seeds, args.cap)
    atomic_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
