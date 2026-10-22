from training.evaluation.balanced_reward_failure import windows


def test_risk_window_uses_recent_placements_and_marks_observed_length():
    data = [(2, 0, 0)] * 4900 + [(17, 3, 1)] * 100
    result = windows(data)
    assert result["100"] == {"observed": 100, "height_start": 17, "height_end": 17,
                             "height_max": 17, "height_mean": 17,
                             "holes_start": 3, "holes_end": 3, "four_clears": 100}
    assert result["5000"]["height_start"] == 2
    assert result["5000"]["four_clears"] == 100
