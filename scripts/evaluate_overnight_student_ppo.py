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


def game(model, seed, cap, fallback=None):
    env = LongCandidateTetrisEnv(max_pieces=cap)
    env.reset(seed=seed)
    scores = [0]
    high = 0
    fallback_moves = 0
    try:
        for _ in range(cap):
            observation, legal = env.get_observation(), env.action_masks()
            actor = model
            if fallback is not None and observation["candidate"][legal, HEIGHT_COLUMN][0] >= fallback[1]:
                actor = fallback[0]
                fallback_moves += 1
            action = fast_action(actor, observation, legal)
            if not legal[action]:
                raise RuntimeError("Evaluation actor chose an illegal action")
            high += int(observation["candidate"][action, HEIGHT_COLUMN] >= 12)
            _, _, over, truncated, info = env.step(int(action))
            scores.append(int(info["score"]))
            if over or truncated:
                break
        outcome = {"pieces": len(scores) - 1, "over": bool(over), "height_ge_12": high}
        if fallback is not None:
            outcome["fallback_moves"] = fallback_moves
        return scores, outcome
    finally:
        env.close()


def evaluate(reference, candidate, seeds, cap, fallback_height=None):
    torch.set_num_threads(1)
    student = MaskablePPO.load(reference, device="cpu")
    trained = MaskablePPO.load(candidate, device="cpu")
    fallback = (student, fallback_height) if fallback_height is not None else None
    pairs = []
    for seed in seeds:
        old_scores, old = game(student, seed, cap)
        new_scores, new = game(trained, seed, cap, fallback)
        shared = min(old["pieces"], new["pieces"])
        pairs.append({
            "seed": seed, "shared_pieces": shared,
            "reference": {**old, "common_score": old_scores[shared]},
            "candidate": {**new, "common_score": new_scores[shared]},
        })
    return {
        "cap": cap, "pairs": pairs,
        "summary": summarize_pairs(pairs),
    }


def summarize_pairs(pairs):
    refs, candidates = ([row[role] for row in pairs] for role in ("reference", "candidate"))
    reference_score = sum(r["common_score"] for r in refs)
    return {
        "common_score_ratio": sum(r["common_score"] for r in candidates) / reference_score
        if reference_score else None,
        "reference_deaths": sum(r["over"] for r in refs),
        "candidate_deaths": sum(r["over"] for r in candidates),
        "reference_pieces": sum(r["pieces"] for r in refs),
        "candidate_pieces": sum(r["pieces"] for r in candidates),
        "reference_high_fraction": sum(r["height_ge_12"] for r in refs) / sum(r["pieces"] for r in refs),
        "candidate_high_fraction": sum(r["height_ge_12"] for r in candidates) / sum(r["pieces"] for r in candidates),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cap", type=int, default=4096)
    parser.add_argument("--fallback-height", type=int)
    args = parser.parse_args()
    if args.cap < 1:
        parser.error("--cap must be positive")
    if args.fallback_height is not None and args.fallback_height < 1:
        parser.error("--fallback-height must be positive")
    result = evaluate(args.reference, args.candidate, args.seeds, args.cap, args.fallback_height)
    if args.fallback_height is not None:
        result["fallback_height"] = args.fallback_height
    atomic_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
