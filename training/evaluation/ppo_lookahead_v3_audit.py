"""Read-only scoring audit of the frozen S1 failure trajectory."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.candidate_features import board_features
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.ppo_lookahead_queue import CONFIGS, load_models
from training.ppo_lookahead_policy import LookaheadController, board_value
from training.train_ppo import ROOT, atomic_json
from training.traditional.v2 import simulate_public

RUN = ROOT / "runs/ppo30m_lookahead_v3"


def terms(child, before, cleared):
    holes, _, height, bump, wells = board_features(child["board"])
    old_holes = board_features(before["board"])[0]
    central = sum(child["board"][y][x] is not None for y in range(4) for x in range(3, 7))
    gain = (child["score"] - before["score"]) / max(800, 800 * before["level"])
    result = {"score_gain_raw": gain, "score_gain_capped": min(1.2, gain),
              "extra_tetris": 2.0 * (cleared == 4), "cleared": 0.10 * cleared,
              "new_holes": -0.32 * max(0, holes-old_holes), "holes": -0.06 * holes,
              "height": -0.12 * max(0, height-12), "bumpiness": -0.025 * bump,
              "birth_zone": -0.10 * central, "wells": 0.08 * min(4, wells),
              "game_over": -100.0 if child["phase"] == "over" else 0.0}
    result["sum"] = sum(v for k, v in result.items() if k not in ("sum", "score_gain_raw"))
    assert abs(result["sum"] - board_value(child, before, cleared)) < 1e-8
    return result


def policy_probs(model, observation, mask):
    with torch.no_grad():
        tensor = {key: torch.as_tensor(value).unsqueeze(0).to(model.device)
                  for key, value in observation.items()}
        distribution = model.policy.get_distribution(tensor, action_masks=mask[None, :])
        return distribution.distribution.probs[0].detach().cpu().numpy()


def audit_failure(seed=990109, cap=20000):
    model = load_models()["ppo"]
    controller = LookaheadController(model, CONFIGS["S1"])
    env = FastCandidateTetrisEnv(max_pieces=cap)
    events = []
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            public = env.core.get_public_observation()
            selected, decision = controller.decide(observation, mask, public, env._placement_cache)
            if selected != decision["ppo_action"]:
                probabilities = policy_probs(model, observation, mask)
                ranked = sorted(np.flatnonzero(mask).tolist(), key=lambda a: (-float(probabilities[a]), a))
                by_id = {p["actionId"]: p for p in env._placement_cache}
                roots = {}
                for label, action in (("ppo", decision["ppo_action"]), ("search", selected)):
                    child, cleared = simulate_public(public, by_id[action])
                    roots[label] = {"action": action, "probability": float(probabilities[action]),
                                    "rank": ranked.index(action)+1, "hold": bool(by_id[action]["hold"]),
                                    "cleared_lines": cleared, "official_score_gain": child["score"]-public["score"],
                                    "phase": child["phase"], "next": child["next"],
                                    "terms": terms(child, public, cleared)}
                p, s = roots["ppo"], roots["search"]
                log_ratio = float(np.log(max(s["probability"], 1e-12) / p["probability"]))
                score_gap = CONFIGS["S1"].ppo_weight*log_ratio + CONFIGS["S1"].search_weight*(s["terms"]["sum"]-p["terms"]["sum"])
                events.append({"seed": seed, "step": env.pieces, "public_hash": hashlib.sha256(
                    json.dumps(public, sort_keys=True).encode()).hexdigest(), "public": public,
                    "ppo": p, "search": s, "hold_changed": p["hold"] != s["hold"],
                    "ppo_log_prob_difference": log_ratio, "search_total_gap": score_gap,
                    "switch_margin": CONFIGS["S1"].switch_margin,
                    "passes_margin": score_gap >= CONFIGS["S1"].switch_margin,
                    "no_tetris_bonus_gap": score_gap-CONFIGS["S1"].search_weight*(s["terms"]["extra_tetris"]-p["terms"]["extra_tetris"]),
                    "root_state_equal": False})
            observation, _, over, capped, info = env.step(selected)
            if over or capped:
                break
        out = {"seed": seed, "pieces": info["pieces"], "game_over": bool(over),
               "events": events, "count": len(events)}
        atomic_json(RUN / "diagnostics" / f"S1_score_audit_seed_{seed}.json", out)
        return out
    finally:
        env.close()


if __name__ == "__main__":
    result = audit_failure()
    print(result["seed"], result["pieces"], result["count"], flush=True)
