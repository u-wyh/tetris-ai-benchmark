"""Deterministic public-state replay of a newly early-dead lookahead game."""
from training.candidate_features import board_features
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.ppo_lookahead_queue import CONFIGS, RUN, load_models
from training.ppo_lookahead_policy import LookaheadController, board_value
from training.traditional.v2 import simulate_public
from training.train_ppo import atomic_json


def replay(seed=990109, variant="S1", cap=20000):
    models = load_models()
    controller = LookaheadController(models["ppo"], CONFIGS[variant])
    env = FastCandidateTetrisEnv(max_pieces=cap)
    changes = []
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            public = env.core.get_public_observation()
            action, decision = controller.decide(obs, mask, public, env._placement_cache)
            if not decision["top1_retained"]:
                by_id = {p["actionId"]: p for p in env._placement_cache}
                alternatives = {}
                for label, chosen in (("ppo", decision["ppo_action"]), ("search", action)):
                    state, cleared = simulate_public(public, by_id[chosen])
                    holes, _, height, bump, wells = board_features(state["board"])
                    alternatives[label] = {"action": chosen, "cleared": cleared,
                        "height": height, "holes": holes, "bumpiness": bump, "wells": wells,
                        "phase": state["phase"], "score_delta": state["score"]-public["score"],
                        "one_step_value": board_value(state, public, cleared)}
                changes.append({"step": env.pieces, "searched": decision["searched"],
                                "alternatives": alternatives})
            obs, _, over, capped, info = env.step(action)
            if over or capped: break
        result = {"seed": seed, "variant": variant, "pieces": info["pieces"],
                  "game_over": bool(over), "changes": changes}
        atomic_json(RUN / "diagnostics" / f"{variant}_seed_{seed}.json", result)
        return result
    finally:
        env.close()


def branch_last_change(seed=990109, step=2849, horizon=1000):
    """Fork the exact Core snapshot before a changed move; continue both with S1."""
    models = load_models()
    controller = LookaheadController(models["ppo"], CONFIGS["S1"])
    env = FastCandidateTetrisEnv(max_pieces=20000)
    try:
        obs, _ = env.reset(seed=seed)
        while env.pieces < step:
            mask = env.action_masks()
            action, _ = controller.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            obs, _, over, capped, _ = env.step(action)
            if over or capped: raise RuntimeError("Died before target branch")
        mask = env.action_masks()
        selected, decision = controller.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
        if selected == decision["ppo_action"]:
            raise RuntimeError("Target step does not contain a search replacement")
        snapshot = env.get_state()
        initial_score = env.core.score
        branches = {}
        for label, first in (("ppo", decision["ppo_action"]), ("search", selected)):
            branch = FastCandidateTetrisEnv(max_pieces=20000)
            try:
                branch.reset(seed=seed)
                branch.set_state(snapshot)
                current = branch.get_observation()
                cleared_four = 0
                for offset in range(horizon):
                    if offset == 0:
                        action = first
                    else:
                        legal = branch.action_masks()
                        action, _ = controller.decide(current, legal,
                                      branch.core.get_public_observation(), branch._placement_cache)
                    current, _, over, capped, info = branch.step(action)
                    cleared_four += info["cleared_lines"] == 4
                    if over or capped: break
                branches[label] = {"pieces": branch.pieces-step, "game_over": bool(over),
                                   "score_gain": info["score"]-initial_score,
                                   "four_clears": cleared_four}
            finally:
                branch.close()
        result = {"seed": seed, "step": step, "horizon": horizon,
                  "first_actions": {"ppo": decision["ppo_action"], "search": selected},
                  "continuation": "S1_after_first_action", "branches": branches}
        atomic_json(RUN / "diagnostics" / f"S1_branch_seed_{seed}_step_{step}.json", result)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    result = replay()
    print(result["seed"], result["variant"], result["pieces"], len(result["changes"]))
