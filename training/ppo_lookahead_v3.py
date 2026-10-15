"""Frozen-PPO-first, public-only conservative reranking of its legal Top-5."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from training.candidate_features import board_features
from training.evaluation.ppo_lookahead_v3_audit import policy_probs, terms
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public


@dataclass(frozen=True)
class V3Config:
    name: str = "V3A"
    top_k: int = 5
    switch_margin: float = 0.45
    hold_extra_margin: float = 0.0
    search_weight: float = 0.45

    def __post_init__(self):
        if self.name not in ("V3A", "V3B") or self.top_k != 5 or self.switch_margin < 0:
            raise ValueError("Invalid V3 configuration")


def value(state, before, cleared):
    breakdown = terms(state, before, cleared)
    return breakdown["sum"] - breakdown["extra_tetris"]


def hold_guard(top, candidate):
    """Require a visible safety gain and preserve next-step options on Hold changes."""
    left, right = top["state"], candidate["state"]
    if left["phase"] != "playing" or right["phase"] != "playing":
        return False, "visible_next_unavailable"
    top_holes, _, top_height, _, top_wells = board_features(left["board"])
    holes, _, height, _, wells = board_features(right["board"])
    if not (holes < top_holes or height <= top_height-2):
        return False, "no_visible_safety_gain"
    if wells < top_wells-2 and height > top_height-2:
        return False, "four_clear_channel_reduced"
    if len(get_legal_placements(right)) < max(1, len(get_legal_placements(left))//2):
        return False, "next_legal_options_reduced"
    return True, "visible_hold_transition_checked"


class V3Controller:
    def __init__(self, ppo, config):
        self.ppo, self.config = ppo, config

    def decide(self, observation, mask, public, placements=None):
        started = perf_counter()
        placements = get_legal_placements(public) if placements is None else placements
        by_id = {p["actionId"]: p for p in placements}
        legal = np.flatnonzero(mask).tolist()
        if set(legal) != set(by_id):
            raise RuntimeError("Official action mask and BFS disagree")
        probs = policy_probs(self.ppo, observation, mask)
        ranked = sorted(legal, key=lambda a: (-float(probs[a]), a))[:self.config.top_k]
        top1 = ranked[0]
        if probs[top1] <= 0:
            raise RuntimeError("PPO returned zero probability for Top-1")
        roots = []
        for rank, action in enumerate(ranked, 1):
            state, cleared = simulate_public(public, by_id[action])
            breakdown = terms(state, public, cleared)
            roots.append({"action": action, "rank": rank, "probability": float(probs[action]),
                          "hold": bool(by_id[action]["hold"]), "cleared": cleared,
                          "phase": state["phase"], "state": state, "terms": breakdown,
                          "value": breakdown["sum"]-breakdown["extra_tetris"]})
        top = roots[0]
        best = top
        best_gap = 0.0
        rejected = []
        for root in roots[1:]:
            gap = (np.log(max(root["probability"], 1e-12)/top["probability"])
                   + self.config.search_weight*(root["value"]-top["value"]))
            margin = self.config.switch_margin
            if root["hold"] != top["hold"]:
                margin += self.config.hold_extra_margin
                if self.config.name == "V3B" and gap >= margin:
                    allowed, reason = hold_guard(top, root)
                    if not allowed:
                        rejected.append({"action": root["action"], "reason": reason})
                        continue
            if gap >= margin and gap-margin > best_gap + 1e-9:
                best, best_gap = root, gap-margin
        selected = best["action"]
        if not mask[selected]:
            raise RuntimeError("V3 selected illegal action")
        return selected, {"ppo_action": top1, "selected_rank": best["rank"],
                          "top1_retained": selected == top1, "changed_hold": best["hold"] != top["hold"],
                          "immediate_four": best["cleared"] == 4,
                          "reason": "visible_value_exceeds_margin" if selected != top1 else "retain_ppo",
                          "margin": self.config.switch_margin + (self.config.hold_extra_margin if best["hold"] != top["hold"] else 0),
                          "score_gap": float(np.log(max(best["probability"], 1e-12)/top["probability"])
                                              + self.config.search_weight*(best["value"]-top["value"])),
                          "hold_transition": {"ppo": {"current": top["state"]["currentPiece"]["type"] if top["state"]["currentPiece"] else None,
                                                       "hold": top["state"]["hold"], "next": top["state"]["next"]},
                                              "selected": {"current": best["state"]["currentPiece"]["type"] if best["state"]["currentPiece"] else None,
                                                           "hold": best["state"]["hold"], "next": best["state"]["next"]}} if best["hold"] != top["hold"] else None,
                          "rejected_hold_candidates": rejected,
                          "roots": [{k: r[k] for k in ("action", "rank", "probability", "hold", "cleared", "phase", "terms")}
                                    for r in roots] if selected != top1 else None,
                          "elapsed_ms": (perf_counter()-started)*1000}
