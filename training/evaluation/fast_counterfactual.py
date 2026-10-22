"""Optional accelerated frozen-PPO continuation for new counterfactual studies.

Uses the existing evaluation-only bitboard placement/feature path. Formal Core,
action IDs, checkpoints, and historical branch records are never modified.
"""

from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action


def fast_branch(snapshot, action, models, horizon):
    """Change one legal action, then run frozen PPO; match legacy branch schema."""
    cap = max(snapshot["max_pieces"], snapshot["pieces"] + horizon + 1)
    state = dict(snapshot, max_pieces=cap)
    env = LongCandidateTetrisEnv(max_pieces=cap)
    try:
        env.reset(seed=state["episode_seed"])
        env.set_state(state)
        start, initial_score = env.pieces, env.core.score
        if not env.action_masks()[action]:
            raise RuntimeError("Illegal first action")
        scores, fours = [], []
        done = capped = False
        while env.pieces - start < horizon:
            observation, mask = env.get_observation(), env.action_masks()
            chosen = action if env.pieces == start else fast_action(models["ppo"], observation, mask)
            if not mask[chosen]:
                raise RuntimeError("Illegal continuation action")
            _, _, done, capped, info = env.step(chosen)
            scores.append(info["score"] - initial_score)
            fours.append((fours[-1] if fours else 0) + int(info["cleared_lines"] == 4))
            if done or capped:
                break
        return {"action": action, "pieces": len(scores), "game_over": bool(done),
                "censored": not (done or capped), "horizon": horizon,
                "scores": scores, "fours": fours}
    finally:
        env.close()
