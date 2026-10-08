import pytest
from training.env.tetris_env import height_risk
from training.train_vector_transaction import vector_config

def test_height_risk_threshold_and_square():
    assert height_risk(12) == 0
    assert height_risk(15) == 9
    assert height_risk(19) == 49

def test_v3_v4_reward_configuration():
    v3 = vector_config(target_steps=2_000_000, hole_penalty_coef=.02, height_penalty_coef=.02)
    v4 = vector_config(target_steps=2_000_000, height_penalty_coef=.02)
    assert v3["reward_version"] != v4["reward_version"]
    assert v3["reward"]["new_holes_penalty_coef"] == .02
    assert v3["reward"]["height_risk_penalty_coef"] == .02
    assert "new_holes_penalty_coef" not in v4["reward"]
    assert v4["reward"]["height_risk_penalty_coef"] == .02

def test_zero_coefficients_are_raw_compatible():
    raw = vector_config()
    assert raw["reward_version"] == "placement-reward-v1"
    assert "height_penalty_coef" not in raw
