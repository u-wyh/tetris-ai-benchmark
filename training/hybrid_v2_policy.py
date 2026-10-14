"""Frozen C1/C2 safety interceptors for PPO30M with BC assistance."""
from training.hybrid_policy import F, risk
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public

VARIANTS = ("C1", "C2")


def obvious_danger(p):
    return (p["lock_out"] or p["central_top4"] >= 2 or
            (p["height"] >= 16 and (p["height_growth"] >= 2 or p["new_holes"] >= 2)) or
            (p["height"] >= 18 and p["height_growth"] >= 1))


def materially_safer(p, b):
    if b["lock_out"]:
        return False
    if p["lock_out"]:
        return True
    return (b["new_holes"] <= p["new_holes"] and
            b["central_top4"] <= p["central_top4"] and
            b["height"] <= p["height"] and
            (b["height"] <= p["height"] - 2 or
             b["central_top4"] <= p["central_top4"] - 2))


def next_visible_legal(public, placement):
    """Count official next actions using only current public Next and Hold."""
    state, _ = simulate_public(public, placement)
    if state["phase"] != "playing":
        return state["phase"], 0
    return state["phase"], len(get_legal_placements(state))


class V2Controller:
    def __init__(self, ppo, bc, variant):
        if variant not in VARIANTS:
            raise ValueError("Unknown Hybrid V2 variant")
        self.ppo, self.bc, self.variant = ppo, bc, variant

    def decide(self, observation, mask, public, placements):
        choices = {}
        for label, model in (("ppo", self.ppo), ("bc", self.bc)):
            action = int(model.predict(observation, deterministic=True, action_masks=mask)[0])
            if not mask[action]:
                raise RuntimeError(f"Frozen {label} chose an illegal action")
            choices[label] = action
        by_id = {p["actionId"]: p for p in placements}
        if any(action not in by_id for action in choices.values()):
            raise RuntimeError("Candidate action differs from official legal placements")
        _, p = risk(public, observation["candidate"][choices["ppo"]], by_id[choices["ppo"]])
        _, b = risk(public, observation["candidate"][choices["bc"]], by_id[choices["bc"]])
        selected, reason = "ppo", "no_obvious_danger"
        next_counts = None
        if obvious_danger(p):
            reason = "bc_not_materially_safer"
            if materially_safer(p, b):
                selected, reason = "bc", "c1_intercept"
                if self.variant == "C2" and not p["lock_out"]:
                    features = observation["candidate"]
                    holds_differ = (features[choices["ppo"], F["hold"]] !=
                                    features[choices["bc"], F["hold"]])
                    if p["height"] < 18 and p["central_top4"] == 0 and holds_differ:
                        selected, reason = "ppo", "c2_hold_continuity"
                    else:
                        p_phase, p_count = next_visible_legal(public, by_id[choices["ppo"]])
                        b_phase, b_count = next_visible_legal(public, by_id[choices["bc"]])
                        next_counts = {"ppo": p_count, "bc": b_count,
                                       "ppo_phase": p_phase, "bc_phase": b_phase}
                        if (b_phase == "over" and p_phase != "over") or (
                                p_phase == "playing" and b_phase == "playing" and
                                b_count <= p_count - 12):
                            selected, reason = "ppo", "c2_next_space"
        return choices[selected], {"source": selected, "reason": reason,
            "ppo_action": choices["ppo"], "bc_action": choices["bc"],
            "ppo_risk": p, "bc_risk": b, "selected_risk": b if selected == "bc" else p,
            "next_visible_legal": next_counts}
