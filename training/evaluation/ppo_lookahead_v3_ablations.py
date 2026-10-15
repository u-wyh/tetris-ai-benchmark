"""Same-Core-snapshot interventions at the known S1 failure, with fixed continuations."""
from dataclasses import replace

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.ppo_lookahead_queue import CONFIGS, load_models
from training.ppo_lookahead_policy import LookaheadController
from training.ppo_lookahead_v3 import V3Config, V3Controller
from training.train_ppo import atomic_json
from training.evaluation.ppo_lookahead_v3_audit import RUN


def run(seed=990109, step=2849, horizon=1000):
    models = load_models()
    ppo = models["ppo"]
    original = LookaheadController(ppo, CONFIGS["S1"])
    controllers = {"S1": original,
                   "A_no_bonus": V3Controller(ppo, V3Config("V3A")),
                   "B_hold_margin": V3Controller(ppo, V3Config("V3B", hold_extra_margin=0.45)),
                   "C_large_margin": LookaheadController(ppo, replace(CONFIGS["S1"], switch_margin=1.0))}
    env = FastCandidateTetrisEnv(max_pieces=20000)
    try:
        observation, _ = env.reset(seed=seed)
        while env.pieces < step:
            mask = env.action_masks()
            action, _ = original.decide(observation, mask, env.core.get_public_observation(), env._placement_cache)
            observation, _, over, capped, _ = env.step(action)
            if over or capped:
                raise RuntimeError("S1 ended before snapshot")
        saved = env.get_state()
        original_score = env.core.score
        mask, public = env.action_masks(), env.core.get_public_observation()
        first = {}
        for name, controller in controllers.items():
            action, decision = controller.decide(observation, mask, public, env._placement_cache)
            first[name] = {"action": action, "ppo_action": decision["ppo_action"],
                           "replaced": action != decision["ppo_action"],
                           "reason": decision.get("reason", "original_s1_rule")}
        ppo_action = first["S1"]["ppo_action"]
        if first["S1"]["action"] != 249 or ppo_action != 383:
            raise RuntimeError("Failure snapshot differs from frozen branch")
        branches = {}
        for name, controller in controllers.items():
            for continuation in ("ppo", name):
                # PPO continuation uses the official deterministic masked policy.
                for initial_label, action in (("ppo", ppo_action), ("variant", first[name]["action"])):
                    key = f"{name}|{continuation}|{initial_label}"
                    branch = FastCandidateTetrisEnv(max_pieces=20000)
                    try:
                        current, _ = branch.reset(seed=seed)
                        branch.set_state(saved)
                        current = branch.get_observation()
                        four = 0
                        for offset in range(horizon):
                            if offset == 0:
                                selected = action
                            elif continuation == "ppo":
                                selected = int(ppo.predict(current, deterministic=True,
                                        action_masks=branch.action_masks())[0])
                            else:
                                selected, _ = controller.decide(current, branch.action_masks(),
                                    branch.core.get_public_observation(), branch._placement_cache)
                            current, _, over, capped, info = branch.step(selected)
                            four += info["cleared_lines"] == 4
                            if over or capped:
                                break
                        branches[key] = {"pieces": branch.pieces-step, "game_over": bool(over),
                                         "censored": not over, "score_gain": info["score"]-original_score,
                                         "four_clears": four}
                    finally:
                        branch.close()
        result = {"seed": seed, "step": step, "horizon": horizon,
                  "first_actions": first, "branches": branches,
                  "interpretation": "Compare first actions only within the same continuation"}
        atomic_json(RUN / "diagnostics" / f"snapshot_ablations_seed_{seed}_step_{step}.json", result)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    print(run(), flush=True)
