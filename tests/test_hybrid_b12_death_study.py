from training.evaluation.hybrid_b12_death_study import classify, windows


def test_counterfactual_classification_respects_censoring():
    actual = {"action": 1, "pieces_after": 100, "game_over": True, "censored": False}
    alternate = {"action": 2, "pieces_after": 5000, "game_over": False, "censored": True}
    assert classify(actual, alternate) == "bc_harmful"
    assert classify(alternate, actual) == "bc_helpful"
    assert classify({**actual, "pieces_after": 5000, "censored": True, "game_over": False},
                    alternate) == "insufficient"
    assert classify(actual, {**alternate, "action": 1}) == "no_action_difference"


def test_death_windows_use_episode_tail_only():
    rows = [{"piece": i, "height": i, "holes": i//2, "bumpiness": i,
             "central_top4": i//3, "legal_next": 70-i,
             "cleared": 4 if i == 29 else 0,
             "source": "bc" if i%2 else "ppo", "hold": "I" if i<20 else "T"}
            for i in range(30)]
    result = windows(rows, 30)
    assert result["20"]["observed"] == 20
    assert result["20"]["height_start"] == 10
    assert result["20"]["fours"] == 1
    assert result["20"]["hold_changes"] == 1
    assert result["100"]["observed"] == 30
