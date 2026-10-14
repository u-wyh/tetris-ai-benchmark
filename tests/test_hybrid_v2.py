import copy
import numpy as np

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_v2_queue import audit_seeds
from training.hybrid_policy import F
from training.hybrid_v2_policy import V2Controller, next_visible_legal


class FixedModel:
    def __init__(self, action):
        self.action = action

    def predict(self, observation, deterministic, action_masks):
        return self.action, None


def synthetic(before=14, p_after=16, b_after=13, p_holes=0, b_holes=0):
    features = np.zeros((1840, 16), dtype=np.uint8)
    for action, after, holes in ((1, p_after, p_holes), (2, b_after, b_holes)):
        features[action, F["max_height_before"]] = before
        features[action, F["max_height_after"]] = after
        features[action, F["new_holes"]] = holes
    mask = np.zeros(1840, dtype=bool)
    mask[[1, 2]] = True
    public = {"board": [[None] * 10 for _ in range(20)]}
    placements = [{"actionId": a, "occupiedCells": [{"x": a, "y": 19}]}
                  for a in (1, 2)]
    return {"candidate": features}, mask, public, placements


def test_c1_danger_and_pareto_boundaries():
    c = V2Controller(FixedModel(1), FixedModel(2), "C1")
    assert c.decide(*synthetic(p_after=15))[1]["reason"] == "no_obvious_danger"
    assert c.decide(*synthetic())[1]["reason"] == "c1_intercept"
    assert c.decide(*synthetic(b_holes=1))[1]["reason"] == "bc_not_materially_safer"
    assert c.decide(*synthetic(b_after=15))[1]["reason"] == "bc_not_materially_safer"


def test_c2_hold_and_next_space_veto(monkeypatch):
    import training.hybrid_v2_policy as policy

    obs, mask, public, placements = synthetic()
    obs["candidate"][2, F["hold"]] = 1
    c = V2Controller(FixedModel(1), FixedModel(2), "C2")
    assert c.decide(obs, mask, public, placements)[1]["reason"] == "c2_hold_continuity"
    obs["candidate"][2, F["hold"]] = 0
    monkeypatch.setattr(policy, "next_visible_legal",
                        lambda public, placement: ("playing", 68 if placement["actionId"] == 1 else 51))
    assert c.decide(obs, mask, public, placements)[1]["reason"] == "c2_next_space"
    monkeypatch.setattr(policy, "next_visible_legal",
                        lambda public, placement: ("playing", 68 if placement["actionId"] == 1 else 60))
    assert c.decide(obs, mask, public, placements)[1]["reason"] == "c1_intercept"


def test_new_seed_sets_are_unoccupied():
    audit_seeds()


def test_full_core_snapshot_preserves_hold_next_and_sequence():
    first = FastCandidateTetrisEnv(max_pieces=100)
    second = FastCandidateTetrisEnv(max_pieces=100)
    try:
        first.reset(seed=710200)
        second.reset(seed=710200)
        first.action_masks()
        hold = next(p["actionId"] for p in first._placement_cache if p["hold"])
        snapshot = copy.deepcopy(first.get_state())
        second.set_state(snapshot)
        for index in range(30):
            action = hold if index == 0 else int(first.action_masks().nonzero()[0][0])
            if index == 0:
                assert first.core.get_public_observation()["next"] == second.core.get_public_observation()["next"]
            first.step(action)
            second.step(action)
            assert first.core.get_public_observation() == second.core.get_public_observation()
            if first._finished:
                break
    finally:
        first.close()
        second.close()


def test_frozen_model_loads_reproduce_actions():
    from training.evaluation.hybrid_queue import load_models, verify_models

    verify_models()
    left, right = load_models(), load_models()
    env = FastCandidateTetrisEnv(max_pieces=10)
    try:
        obs, _ = env.reset(seed=710201)
        mask = env.action_masks()
        for label in ("ppo", "bc"):
            a = int(left[label].predict(obs, deterministic=True, action_masks=mask)[0])
            b = int(right[label].predict(obs, deterministic=True, action_masks=mask)[0])
            assert a == b and mask[a]
    finally:
        env.close()


def test_counterfactual_branch_restarts_from_identical_core():
    from training.evaluation.hybrid_failure import branch
    from training.evaluation.hybrid_queue import load_models
    from training.hybrid_policy import HybridConfig, HybridController

    models = load_models()
    env = FastCandidateTetrisEnv(max_pieces=100000)
    try:
        obs, _ = env.reset(seed=700026)
        control = HybridController(models["ppo"], models["bc"], HybridConfig("B", 12, 8))
        for index in range(269):
            mask = env.action_masks()
            action, decision = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            if index == 268:
                saved = env.get_state()
                before = env.core.get_public_observation()
                first = branch(models, saved, decision["ppo_action"], index, horizon=20)
                second = branch(models, saved, decision["ppo_action"], index, horizon=20)
                assert first == second
                assert env.core.get_public_observation() == before
                break
            obs, _, _, _, _ = env.step(action)
    finally:
        env.close()
