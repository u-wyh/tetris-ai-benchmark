"""Deterministic, public-state fusion of frozen Candidate PPO and BC policies."""
from dataclasses import dataclass

from training.candidate_features import FEATURE_NAMES, candidate_board

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class HybridConfig:
    mode: str
    height: int = 14
    recovery: int = 10
    min_dwell: int = 8

    def __post_init__(self):
        if self.mode not in ("A", "B") or not 8 <= self.height <= 18:
            raise ValueError("Invalid hybrid mode or height")
        if not 0 <= self.recovery < self.height or self.min_dwell < 0:
            raise ValueError("Invalid recovery or dwell")


def risk(public, features, placement):
    """Score visible one-step danger; lower is safer. No future bag access."""
    board, _, lock_out = candidate_board(public["board"], placement["occupiedCells"])
    central = sum(board[y][x] is not None for y in range(4) for x in range(3, 7))
    height = int(features[F["max_height_after"]])
    growth = max(0, height - int(features[F["max_height_before"]]))
    holes = int(features[F["new_holes"]])
    bump = int(features[F["bumpiness_after"]])
    return (1000000 * int(lock_out) + 1000 * central + 100 * max(0, height - 11)
            + 35 * holes + 12 * growth + 2 * bump), {
        "lock_out": bool(lock_out), "central_top4": central, "height": height,
        "height_growth": growth, "new_holes": holes, "bumpiness": bump}


class HybridController:
    def __init__(self, ppo, bc, config):
        self.ppo, self.bc, self.config = ppo, bc, config
        self.active = "ppo"
        self.last_switch = -config.min_dwell
        self.step = 0

    def decide(self, observation, mask, public, placements):
        choices = {}
        for name, model in (("ppo", self.ppo), ("bc", self.bc)):
            action = int(model.predict(observation, deterministic=True, action_masks=mask)[0])
            if not mask[action]:
                raise RuntimeError(f"{name} proposed an illegal action: {action}")
            choices[name] = action
        by_id = {p["actionId"]: p for p in placements}
        if any(a not in by_id for a in choices.values()):
            raise RuntimeError("Model action and official placements disagree")
        candidates = {name: risk(public, observation["candidate"][action], by_id[action])
                      for name, action in choices.items()}
        p_score, p_risk = candidates["ppo"]
        b_score, b_risk = candidates["bc"]
        before_height = int(observation["candidate"][choices["ppo"]][F["max_height_before"]])
        if self.config.mode == "A":
            if self.active == "ppo" and (before_height >= self.config.height or
                    p_risk["central_top4"] or p_risk["lock_out"]):
                self.active, self.last_switch = "bc", self.step
            elif (self.active == "bc" and before_height <= self.config.recovery and
                  self.step - self.last_switch >= self.config.min_dwell):
                self.active, self.last_switch = "ppo", self.step
            selected = self.active
            reason = "state_switch" if self.last_switch == self.step else "state_hold"
            if selected == "bc" and (b_risk["lock_out"] and not p_risk["lock_out"] or
                                     b_score > p_score + 100):
                selected, reason = "ppo", "no_safe_bc"
        else:
            triggered = (p_risk["lock_out"] or p_risk["central_top4"] or
                         (p_risk["height"] >= self.config.height and
                          (p_risk["height_growth"] or p_risk["new_holes"])))
            safer = (not b_risk["lock_out"] and b_score + 100 <= p_score and
                     b_risk["height"] <= p_risk["height"] and
                     b_risk["central_top4"] <= p_risk["central_top4"])
            selected = "bc" if triggered and safer else "ppo"
            reason = "intercept" if selected == "bc" else ("no_safe_bc" if triggered else "low_risk")
        self.step += 1
        return choices[selected], {"source": selected, "reason": reason,
            "ppo_action": choices["ppo"], "bc_action": choices["bc"],
            "ppo_risk": p_risk, "bc_risk": b_risk,
            "selected_risk": candidates[selected][1]}
