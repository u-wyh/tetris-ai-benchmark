"""Local same-snapshot branches for new early deaths; do not use to tune frozen V3."""
import sys

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.ppo_lookahead_queue import load_models
from training.evaluation.ppo_lookahead_v3_audit import RUN
from training.ppo_lookahead_v3 import V3Config, V3Controller
from training.train_ppo import atomic_json


def branch(seed=991104, step=1592, variant="V3A", horizon=1000):
    model = load_models()["ppo"]
    controller = V3Controller(model, V3Config(variant, hold_extra_margin=0.45 if variant == "V3B" else 0))
    env = FastCandidateTetrisEnv(max_pieces=20000)
    try:
        obs, _ = env.reset(seed=seed)
        while env.pieces < step:
            chosen, _ = controller.decide(obs, env.action_masks(), env.core.get_public_observation(), env._placement_cache)
            obs, _, over, capped, _ = env.step(chosen)
            if over or capped:
                raise RuntimeError("Died before requested snapshot")
        selected, decision = controller.decide(obs, env.action_masks(), env.core.get_public_observation(), env._placement_cache)
        if selected == decision["ppo_action"]:
            raise RuntimeError("Requested step is not an action replacement")
        saved = env.get_state()
        starting_score = env.core.score
        branches = {}
        for continuation in ("ppo", variant):
            for label, first in (("ppo", decision["ppo_action"]), ("search", selected)):
                copy = FastCandidateTetrisEnv(max_pieces=20000)
                try:
                    copy.reset(seed=seed)
                    copy.set_state(saved)
                    current = copy.get_observation()
                    fours = 0
                    for offset in range(horizon):
                        if offset == 0:
                            action = first
                        elif continuation == "ppo":
                            action = int(model.predict(current, deterministic=True,
                                          action_masks=copy.action_masks())[0])
                        else:
                            action, _ = controller.decide(current, copy.action_masks(),
                                copy.core.get_public_observation(), copy._placement_cache)
                        current, _, over, capped, info = copy.step(action)
                        fours += info["cleared_lines"] == 4
                        if over or capped:
                            break
                    branches[f"{continuation}|{label}"] = {"pieces": copy.pieces-step,
                        "game_over": bool(over), "censored": not over,
                        "score_gain": info["score"]-starting_score, "four_clears": fours}
                finally:
                    copy.close()
        result = {"seed": seed, "step": step, "variant": variant,
                  "first_actions": {"ppo": decision["ppo_action"], "search": selected},
                  "continuation": "fixed per paired branch", "branches": branches}
        atomic_json(RUN / "diagnostics" / f"new_failure_{variant}_seed_{seed}_step_{step}.json", result)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    print(branch(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv)>3 else "V3A"), flush=True)
