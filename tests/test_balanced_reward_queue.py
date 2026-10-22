from training.evaluation.balanced_reward_queue import long_gate, short_gate


def test_survival_gate_does_not_reward_four_clear_drop_alone():
    base = {"mean_pieces": 18000, "p10_pieces": 12000, "early_under_10000": 1,
            "mean_score": 100, "four_fraction": .93}
    candidate = {"mean_pieces": 17000, "p10_pieces": 11000, "early_under_10000": 1,
                 "mean_score": 95, "four_fraction": .70,
                 "common_score_ratio_to_ppo": .95}
    assert not short_gate({"ppo": base, "candidate": candidate})
    assert not long_gate({"ppo": base, "candidate": candidate})
    candidate.update(mean_pieces=19000, p10_pieces=13000)
    assert short_gate({"ppo": base, "candidate": candidate})
    assert long_gate({"ppo": base, "candidate": candidate})
