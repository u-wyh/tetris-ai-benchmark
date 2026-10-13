import numpy as np

from training.hybrid_policy import HybridConfig, HybridController, F


class Model:
    def __init__(self, action):
        self.action = action

    def predict(self, observation, deterministic, action_masks):
        assert deterministic
        return self.action, None


def scene(height, p_height, b_height, p_holes=0):
    features = np.zeros((1840, 16), dtype=np.uint8)
    for action, after in ((1, p_height), (2, b_height)):
        features[action, F["max_height_before"]] = height
        features[action, F["max_height_after"]] = after
    features[1, F["new_holes"]] = p_holes
    mask = np.zeros(1840, dtype=bool)
    mask[[1, 2]] = True
    public = {"board": [[None] * 10 for _ in range(20)]}
    placements = [{"actionId": a, "occupiedCells": [{"x": a, "y": 19}]}
                  for a in (1, 2)]
    return {"candidate": features}, mask, public, placements


def test_state_switch_hysteresis():
    controller = HybridController(Model(1), Model(2), HybridConfig("A", 14, 10, 2))
    assert controller.decide(*scene(14, 15, 10))[1]["source"] == "bc"
    assert controller.decide(*scene(10, 11, 10))[1]["source"] == "bc"
    assert controller.decide(*scene(10, 11, 10))[1]["source"] == "ppo"


def test_intercept_requires_safer_bc():
    controller = HybridController(Model(1), Model(2), HybridConfig("B", 14, 10))
    assert controller.decide(*scene(14, 16, 11, 3))[1]["reason"] == "intercept"
    assert controller.decide(*scene(14, 16, 16))[1]["reason"] == "no_safe_bc"


def test_low_risk_keeps_ppo_and_reproducible():
    inputs = scene(2, 3, 2)
    results = [HybridController(Model(1), Model(2), HybridConfig("B")).decide(*inputs)
               for _ in range(2)]
    assert results[0] == results[1]
    assert results[0][0] == 1


def test_illegal_action_rejected():
    controller = HybridController(Model(3), Model(2), HybridConfig("B"))
    try:
        controller.decide(*scene(2, 3, 2))
    except RuntimeError as error:
        assert "illegal" in str(error)
    else:
        raise AssertionError("Illegal action accepted")


def test_frozen_models_preserve_public_sequence_for_identical_actions():
    from training.evaluation.hybrid_queue import load_models, verify_models
    from training.env.fast_candidate_env import FastCandidateTetrisEnv

    verify_models()
    models = load_models()
    first, second = FastCandidateTetrisEnv(max_pieces=20), FastCandidateTetrisEnv(max_pieces=20)
    try:
        observation, _ = first.reset(seed=700100)
        second.reset(seed=700100)
        controller = HybridController(models["ppo"], models["bc"], HybridConfig("A", 14, 10))
        for _ in range(20):
            mask = first.action_masks()
            public = first.core.get_public_observation()
            action, decision = controller.decide(observation, mask, public, first._placement_cache)
            again, _ = models[decision["source"]].predict(observation, deterministic=True,
                                                            action_masks=mask)
            assert action == int(again) and mask[action]
            observation, _, _, _, _ = first.step(action)
            second.step(action)
            assert first.core.get_public_observation() == second.core.get_public_observation()
    finally:
        first.close()
        second.close()
