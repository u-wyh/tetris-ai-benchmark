"""Conservative, public-only lookahead among frozen PPO's legal Top-5."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from training.candidate_features import board_features
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public


@dataclass(frozen=True)
class LookaheadConfig:
    mode: str = "S1"
    top_k: int = 5
    beam_width: int = 8
    ppo_weight: float = 1.0
    search_weight: float = 0.45
    switch_margin: float = 0.45
    trigger_height: int = 12

    def __post_init__(self):
        if self.mode not in ("S1", "S2") or not 1 <= self.top_k <= 10 or self.beam_width < 1:
            raise ValueError("Invalid lookahead configuration")


def public_key(state):
    return (tuple(tuple(bool(c) for c in row) for row in state["board"]),
            state["currentPiece"]["type"] if state["currentPiece"] else None,
            state["hold"], tuple(state["next"]), state["holdAvailable"], state["phase"],
            state["score"], state["lines"])


def board_value(state, before, cleared):
    """Fixed units: per-piece score normalized by pre-action level."""
    holes, _, height, bump, wells = board_features(state["board"])
    old_holes = board_features(before["board"])[0]
    central = sum(state["board"][y][x] is not None for y in range(4) for x in range(3, 7))
    score_gain = (state["score"] - before["score"]) / max(800, 800 * before["level"])
    result = (min(1.2, score_gain) + 2.0 * (cleared == 4) + 0.10 * cleared
              - 0.32 * max(0, holes - old_holes) - 0.06 * holes
              - 0.12 * max(0, height - 12) - 0.025 * bump - 0.10 * central
              + 0.08 * min(4, wells))
    if state["phase"] == "over":
        result -= 100.0
    return result


def trigger(top_state, public, height_threshold):
    before = board_features(public["board"])
    after = board_features(top_state["board"])
    return (top_state["phase"] == "over" or after[2] >= height_threshold and
            (after[2] > before[2] or after[0] > before[0] or
             any(top_state["board"][y][x] is not None for y in range(4) for x in range(3, 7))))


class LookaheadController:
    def __init__(self, ppo, config):
        self.ppo = ppo
        self.config = config

    def decide(self, observation, mask, public, placements=None):
        began = perf_counter()
        placements = get_legal_placements(public) if placements is None else placements
        by_id = {p["actionId"]: p for p in placements}
        legal = np.flatnonzero(mask)
        if set(legal.tolist()) != set(by_id):
            raise RuntimeError("Official action mask and BFS disagree")
        with torch.no_grad():
            tensor = {key: torch.as_tensor(value).unsqueeze(0).to(self.ppo.device)
                      for key, value in observation.items()}
            distribution = self.ppo.policy.get_distribution(tensor, action_masks=mask[None, :])
            probs = distribution.distribution.probs[0].detach().cpu().numpy()
        ranked = sorted(legal.tolist(), key=lambda a: (-float(probs[a]), a))[:self.config.top_k]
        top1 = ranked[0]
        if probs[top1] <= 0:
            raise RuntimeError("PPO returned zero probability for Top-1")
        roots, cache = [], {}
        for rank, action in enumerate(ranked, 1):
            state, cleared = simulate_public(public, by_id[action])
            key = public_key(state)
            if key not in cache:
                cache[key] = {"value": board_value(state, public, cleared), "state": state,
                              "cleared": cleared}
            roots.append({"action": action, "rank": rank, "probability": float(probs[action]),
                          "key": key, **cache[key]})
        searched = self.config.mode == "S2" and trigger(roots[0]["state"], public, self.config.trigger_height)
        future_cache = {}
        nodes = len(cache)
        if searched:
            for root in roots:
                key, state = root["key"], root["state"]
                if key in future_cache:
                    continue
                if state["phase"] != "playing":
                    future_cache[key] = 0.0
                    continue
                future = get_legal_placements(state)
                nodes += len(future)
                if not future:
                    future_cache[key] = -100.0
                    continue
                # Cheap first-pass ordering, then exact simulation on a fixed beam.
                ranked_future = sorted(future, key=lambda p: (p["y"], -abs(p["x"] - 4), -p["actionId"]), reverse=True)
                best = -float("inf")
                for placement in ranked_future[:self.config.beam_width]:
                    leaf, cleared = simulate_public(state, placement)
                    best = max(best, board_value(leaf, state, cleared))
                future_cache[key] = best
        best = roots[0]
        top_value = roots[0]["value"] + (0.35 * future_cache.get(roots[0]["key"], 0.0) if searched else 0.0)
        top_score = self.config.search_weight * top_value
        best_score = top_score
        for root in roots[1:]:
            value = root["value"] + (0.35 * future_cache.get(root["key"], 0.0) if searched else 0.0)
            score = (self.config.ppo_weight * np.log(max(root["probability"], 1e-12) / probs[top1])
                     + self.config.search_weight * value)
            if score > best_score + 1e-9:
                best, best_score = root, score
        selected = best["action"] if best_score >= top_score + self.config.switch_margin else top1
        if not mask[selected]:
            raise RuntimeError("Search selected illegal action")
        elapsed = (perf_counter() - began) * 1000
        return selected, {"ppo_action": top1, "selected_rank": ranked.index(selected) + 1,
                          "top1_retained": selected == top1, "searched": searched,
                          "nodes": nodes, "elapsed_ms": elapsed,
                          "root_phase": roots[0]["state"]["phase"],
                          "selected_phase": next(r["state"]["phase"] for r in roots if r["action"] == selected),
                          "visible_next_lengths": [len(r["state"]["next"]) for r in roots]}
