"""V3-A: preserve B12, veto structurally fragile BC interventions only."""

from training.candidate_features import board_features, candidate_board
from training.hybrid_policy import HybridConfig, HybridController
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public

VERSION = "b12-v3a-structural-continuity-v1"
ROW_TRANSITION_EXCESS = 2
SAFE_NEXT_GAIN_FOR_HOLD_CHANGE = 3
SAFE_HEIGHT_FLOOR = 16


def row_transitions(board):
    total = 0
    for row in board:
        previous = True
        for cell in row:
            current = cell is not None
            total += previous != current
            previous = current
        total += not previous
    return total


def next_safe_count(public, placement, height_limit):
    """Known next piece only; no bag, RNG, or hidden preview is read."""
    state, _ = simulate_public(public, placement)
    if state["phase"] != "playing":
        return 0, 0
    baseline_holes = board_features(state["board"])[0]
    legal = get_legal_placements(state)
    safe = 0
    for next_placement in legal:
        child, _, lock_out = candidate_board(state["board"], next_placement["occupiedCells"])
        holes, _, height, _, _ = board_features(child)
        safe += not lock_out and height <= height_limit and holes <= baseline_holes + 1
    return len(legal), safe


class V3Controller:
    def __init__(self, ppo, bc):
        self.base = HybridController(ppo, bc, HybridConfig("B", 12, 8))

    def decide(self, observation, mask, public, placements):
        actual, decision = self.base.decide(observation, mask, public, placements)
        decision = dict(decision)
        decision.update({"base_source": decision["source"], "base_reason": decision["reason"],
                         "v3_rule": VERSION, "v3_rejected": False,
                         "v3_features": None})
        if decision["source"] != "bc" or decision["ppo_action"] == decision["bc_action"]:
            return actual, decision
        p_risk, b_risk = decision["ppo_risk"], decision["bc_risk"]
        if p_risk["lock_out"] and not b_risk["lock_out"]:
            decision["reason"] = "v3_emergency_bc"
            return actual, decision
        by_id = {p["actionId"]: p for p in placements}
        p_action, b_action = decision["ppo_action"], decision["bc_action"]
        if p_action not in by_id or b_action not in by_id:
            raise RuntimeError("V3 candidates differ from legal placements")
        p_state, _ = simulate_public(public, by_id[p_action])
        b_state, _ = simulate_public(public, by_id[b_action])
        if b_state["phase"] != "playing" and p_state["phase"] == "playing":
            return self._reject(p_action, decision, "v3_bc_terminal", None)
        p_rows = row_transitions(p_state["board"])
        b_rows = row_transitions(b_state["board"])
        features = {"ppo_row_transitions": p_rows, "bc_row_transitions": b_rows,
                    "ppo_hold_after": p_state["hold"], "bc_hold_after": b_state["hold"],
                    "ppo_uses_hold": bool(by_id[p_action]["hold"]),
                    "bc_uses_hold": bool(by_id[b_action]["hold"])}
        if b_rows >= p_rows + ROW_TRANSITION_EXCESS:
            return self._reject(p_action, decision, "v3_row_fragmentation", features)
        if features["ppo_uses_hold"] != features["bc_uses_hold"]:
            limit = max(SAFE_HEIGHT_FLOOR, p_risk["height"], b_risk["height"])
            p_total, p_safe = next_safe_count(public, by_id[p_action], limit)
            b_total, b_safe = next_safe_count(public, by_id[b_action], limit)
            features.update({"ppo_next_legal": p_total, "bc_next_legal": b_total,
                             "ppo_next_safe": p_safe, "bc_next_safe": b_safe})
            if b_safe < p_safe + SAFE_NEXT_GAIN_FOR_HOLD_CHANGE:
                return self._reject(p_action, decision, "v3_hold_without_safe_mobility_gain", features)
        decision["v3_features"] = features
        return actual, decision

    @staticmethod
    def _reject(ppo_action, decision, reason, features):
        decision["source"] = "ppo"
        decision["reason"] = reason
        decision["selected_risk"] = decision["ppo_risk"]
        decision["v3_rejected"] = True
        decision["v3_features"] = features
        return ppo_action, decision
