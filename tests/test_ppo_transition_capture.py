import random

from training.evaluation.ppo_transition_capture import prospective_stratum, reservoir_add


def test_prospective_risk_and_recovery_control_are_distinct():
    last = {"rapid_stack": -10000, "high_control": -10000}
    assert prospective_stratum(15, 5, 100, last) == "rapid_stack"
    assert prospective_stratum(15, 2, 100, last) == "high_control"
    assert prospective_stratum(15, 3, 100, last) is None
    assert prospective_stratum(14, 8, 100, last) is None
    assert prospective_stratum(15, 5, 99, last) is None
    last["rapid_stack"] = 100
    assert prospective_stratum(19, 8, 299, last) is None
    assert prospective_stratum(19, 8, 300, last) == "rapid_stack"


def test_reservoir_is_seed_reproducible_and_bounded():
    def sample():
        rng = random.Random(99)
        pool = []
        for count in range(1, 1001):
            reservoir_add(pool, count, count, rng)
        return pool

    assert sample() == sample()
    assert len(sample()) == 2
    assert all(1 <= value <= 1000 for value in sample())
