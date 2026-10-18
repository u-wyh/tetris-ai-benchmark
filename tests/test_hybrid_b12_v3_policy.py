import numpy as np

from training.hybrid_b12_v3_policy import V3Controller
from training.hybrid_policy import HybridController


def inputs(hold_diff=False):
    mask = np.zeros(1840, dtype=bool)
    mask[[1, 2]] = True
    public = {"board": [[None]*10 for _ in range(20)]}
    placements = [{"actionId": 1, "hold": False, "occupiedCells": []},
                  {"actionId": 2, "hold": hold_diff, "occupiedCells": []}]
    return {"candidate": np.zeros((1840, 16), dtype=np.uint8)}, mask, public, placements


def decision(lock_out=False):
    p = {"lock_out": lock_out, "height": 16, "central_top4": 0}
    b = {"lock_out": False, "height": 14, "central_top4": 0}
    return {"source": "bc", "reason": "intercept", "ppo_action": 1, "bc_action": 2,
            "ppo_risk": p, "bc_risk": b, "selected_risk": b}


def test_row_fragmentation_veto_keeps_ppo(monkeypatch):
    import training.hybrid_b12_v3_policy as policy
    monkeypatch.setattr(HybridController, "decide", lambda self, *args: (2, decision()))
    monkeypatch.setattr(policy, "simulate_public",
                        lambda public, placement: ({"phase": "playing", "board": [[placement["actionId"]]],
                                                    "hold": None}, None))
    monkeypatch.setattr(policy, "row_transitions", lambda board: 60 if board[0][0] == 1 else 62)
    action, result = V3Controller(None, None).decide(*inputs())
    assert action == 1 and result["reason"] == "v3_row_fragmentation"
    assert result["base_source"] == "bc" and result["v3_rejected"]


def test_hold_veto_requires_safe_mobility_gain(monkeypatch):
    import training.hybrid_b12_v3_policy as policy
    monkeypatch.setattr(HybridController, "decide", lambda self, *args: (2, decision()))
    monkeypatch.setattr(policy, "simulate_public",
                        lambda public, placement: ({"phase": "playing", "board": [[placement["actionId"]]],
                                                    "hold": "I"}, None))
    monkeypatch.setattr(policy, "row_transitions", lambda board: 60)
    monkeypatch.setattr(policy, "next_safe_count", lambda public, placement, limit:
                        (68, 10 if placement["actionId"] == 1 else 11))
    action, result = V3Controller(None, None).decide(*inputs(hold_diff=True))
    assert action == 1 and result["reason"] == "v3_hold_without_safe_mobility_gain"
    monkeypatch.setattr(policy, "next_safe_count", lambda public, placement, limit:
                        (68, 10 if placement["actionId"] == 1 else 13))
    action, result = V3Controller(None, None).decide(*inputs(hold_diff=True))
    assert action == 2 and not result["v3_rejected"]


def test_lock_out_rescue_bypasses_local_veto(monkeypatch):
    monkeypatch.setattr(HybridController, "decide", lambda self, *args: (2, decision(True)))
    action, result = V3Controller(None, None).decide(*inputs(hold_diff=True))
    assert action == 2 and result["reason"] == "v3_emergency_bc"
