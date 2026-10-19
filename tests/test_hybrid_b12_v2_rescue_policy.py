import numpy as np

from training.hybrid_b12_v2_rescue_policy import RescueController
from training.hybrid_policy import HybridController
from training.traditional.v2 import V2


def inputs():
    board = [[None] * 10 for _ in range(20)]
    mask = np.zeros(1840, dtype=bool)
    mask[[1, 2, 3]] = True
    placements = [{"actionId": i, "occupiedCells": [{"x": i, "y": 17}], "hold": False}
                  for i in (1, 2, 3)]
    return {}, mask, {"board": board}, placements


def base(*args):
    return 1, {"source": "ppo", "reason": "low_risk", "ppo_action": 1, "bc_action": 2,
               "ppo_risk": {}, "bc_risk": {}, "selected_risk": {}}


def test_low_risk_is_identical_to_b12_without_v2_call(monkeypatch):
    monkeypatch.setattr(HybridController, "decide", base)
    monkeypatch.setattr(V2, "choose", lambda *args: (_ for _ in ()).throw(AssertionError("V2 called")))
    controller = RescueController(None, None)
    action, decision = controller.decide(*inputs())
    assert action == 1 and decision["source"] == "ppo"
    assert not decision["v2_called"] and controller.v2_calls == 0


def test_high_risk_single_step_then_b12_resumes(monkeypatch):
    import training.hybrid_b12_v2_rescue_policy as policy
    monkeypatch.setattr(HybridController, "decide", base)
    monkeypatch.setattr(policy, "board_features", lambda board: (0, 0, 17, 27, 0))
    monkeypatch.setattr(V2, "choose", lambda *args: 3)
    monkeypatch.setattr(policy, "candidate_board", lambda *args: ([], 0, False))
    controller = RescueController(None, None)
    action, decision = controller.decide(*inputs())
    assert action == 3 and decision["source"] == "v2" and decision["v2_changed"]
    monkeypatch.setattr(policy, "board_features", lambda board: (0, 0, 16, 27, 0))
    action, decision = controller.decide(*inputs())
    assert action == 1 and decision["source"] == "ppo" and controller.base.step == 0


def test_duplicate_or_lock_out_action_is_not_used(monkeypatch):
    import training.hybrid_b12_v2_rescue_policy as policy
    monkeypatch.setattr(HybridController, "decide", base)
    monkeypatch.setattr(policy, "board_features", lambda board: (0, 0, 17, 27, 0))
    controller = RescueController(None, None)
    monkeypatch.setattr(V2, "choose", lambda *args: 2)
    assert controller.decide(*inputs())[0] == 1
    monkeypatch.setattr(V2, "choose", lambda *args: 3)
    monkeypatch.setattr(policy, "candidate_board", lambda *args: ([], 0, True))
    assert controller.decide(*inputs())[0] == 1


def test_illegal_v2_action_fails(monkeypatch):
    import pytest
    import training.hybrid_b12_v2_rescue_policy as policy
    monkeypatch.setattr(HybridController, "decide", base)
    monkeypatch.setattr(policy, "board_features", lambda board: (0, 0, 17, 27, 0))
    monkeypatch.setattr(V2, "choose", lambda *args: 9)
    with pytest.raises(RuntimeError, match="illegal"):
        RescueController(None, None).decide(*inputs())
