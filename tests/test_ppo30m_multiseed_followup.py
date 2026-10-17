"""Multi-seed follow-up keeps frozen models and traditional rules separate."""

from pathlib import Path

from training.evaluation import ppo30m_multiseed_followup as followup


def test_training_seeds_and_evaluation_seeds_are_distinct():
    assert len(followup.TRAIN_SEEDS) == 3
    assert len(followup.EVAL_SEEDS) == 16
    assert set(followup.TRAIN_SEEDS).isdisjoint(followup.EVAL_SEEDS)
    assert followup.TASKS * 4096 == 503808
    assert followup.VARIANTS[0:3] == ("ppo", "bc", "B12")
    assert followup.VARIANTS[-1] == "traditional_v2_hold"


def test_neural_model_routing_keeps_frozen_references():
    path = Path("/tmp/frozen-independent-candidate.zip")
    models = followup.neural_models("normal_994101", {"normal_994101": {"path": str(path)}})
    assert models == {"ppo": followup.PPO, "normal": path}
    assert followup.neural_models("B12", {}) == {"ppo": followup.PPO, "bc": followup.BC}


def test_traditional_smoke_uses_same_cap_and_score(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "RUN", tmp_path)
    monkeypatch.setattr(followup, "CAP", 50)
    (tmp_path / "evaluation/traditional_v2_hold").mkdir(parents=True)
    code_hash = followup.sha256(followup.ROOT / "training/traditional/v2.py")
    result = followup.traditional_game(1420000, code_hash)
    assert result["pieces"] == 50 and result["capped"] and not result["game_over"]
    assert result["score"] > 0 and sum(result["clears_0_to_4"]) == 50
