"""One-step, public-state V2 Hold rescue of the frozen B12 controller."""

from time import perf_counter

from training.candidate_features import board_features, candidate_board
from training.hybrid_policy import HybridConfig, HybridController
from training.traditional.v2 import V2, V2Config

VERSION = "b12-v2-rescue-v1"
MIN_HEIGHT = 17
MIN_BUMPINESS = 25


class RescueController:
    def __init__(self, ppo, bc):
        self.base = HybridController(ppo, bc, HybridConfig("B", 12, 8))
        self.expert = V2(V2Config(mode="hold"))
        self.v2_calls = 0
        self.v2_seconds = 0.0

    def decide(self, observation, mask, public, placements):
        action, original = self.base.decide(observation, mask, public, placements)
        decision = dict(original)
        decision.update({"base_source": original["source"], "v2_called": False,
                         "v2_changed": False, "v2_action": None, "v2_seconds": 0.0})
        holes, total_height, height, bump, covered = board_features(public["board"])
        decision["rescue_features"] = {"height": height, "bumpiness": bump}
        if height < MIN_HEIGHT or bump < MIN_BUMPINESS:
            return action, decision
        started = perf_counter()
        v2_action = self.expert.choose(public, placements)
        elapsed = perf_counter() - started
        self.v2_calls += 1
        self.v2_seconds += elapsed
        decision.update({"v2_called": True, "v2_action": v2_action, "v2_seconds": elapsed})
        by_id = {p["actionId"]: p for p in placements}
        if v2_action not in by_id or not mask[v2_action]:
            raise RuntimeError("V2 Hold proposed an illegal action")
        if v2_action in (decision["ppo_action"], decision["bc_action"]):
            decision["reason"] = "v2_same_as_neural_expert"
            return action, decision
        _, _, lock_out = candidate_board(public["board"], by_id[v2_action]["occupiedCells"])
        if lock_out:
            decision["reason"] = "v2_immediate_lock_out"
            return action, decision
        decision["source"] = "v2"
        decision["reason"] = "v2_single_step_rescue"
        decision["v2_changed"] = v2_action != action
        return v2_action, decision
