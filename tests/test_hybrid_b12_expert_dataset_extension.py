from training.evaluation import hybrid_b12_expert_dataset_extension as study


def test_new_seed_groups_and_budget_are_separate():
    assert len(study.TRAIN) == 32 and len(study.VALID) == 16
    assert not set(study.TRAIN + study.VALID) & set(study.RESERVED)
    assert study.SAMPLER["A"]["quota"] == 4
    assert study.SAMPLER["B"]["quota"] == 2
    assert sum(rule["quota"] for rule in study.SAMPLER.values()) == 8


def test_new_sampler_is_deterministic():
    value = study.sample_uniform(study.TRAIN[0], 1234, "A")
    assert value == study.sample_uniform(study.TRAIN[0], 1234, "A")
    assert 0 <= value < 1
    assert value != study.sample_uniform(study.TRAIN[0], 1234, "B")


def test_censoring_classification_is_unchanged():
    dead = {"pieces_after": 200, "game_over": True}
    capped = {"pieces_after": 5000, "game_over": False}
    assert study.classify_pair(dead, capped) == "beneficial"
    assert study.classify_pair(capped, dead) == "harmful"
    assert study.classify_pair(capped, capped) == "uncertain"
