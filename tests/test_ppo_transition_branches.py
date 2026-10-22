from training.evaluation.ppo_transition_branches import grade, horizon


def outcome(pieces, game_over, score):
    return {"pieces": pieces, "game_over": game_over,
            "censored": not game_over, "scores": [score] * pieces,
            "fours": [0] * pieces}


def test_hazard_branch_gets_longer_bounded_window():
    assert horizon({"stratum": "rapid_stack"}) == 3000
    assert horizon({"stratum": "high_control"}) == 1000


def test_grade_requires_observed_death_and_preserved_score():
    base = outcome(300, True, 100)
    alternative = outcome(1000, False, 95)
    assert grade(base, alternative)["rescued_in_window"]
    alternative["scores"] = [80] * 1000
    assert not grade(base, alternative)["rescued_in_window"]
    base["game_over"] = False
    assert not grade(base, outcome(1000, False, 95))["rescued_in_window"]
