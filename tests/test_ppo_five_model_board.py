import pytest

from analysis.ppo_five_model_board import bootstrap_mean, death_row, paired


def test_death_metrics_and_height_15_remaining_life():
    board = [[None] * 10 for _ in range(20)]
    board[1][4] = "T"
    board[2][5] = "T"
    row = death_row(100000, 40, 3, 500, 4, board, 35, True)
    assert row["max_height_at_death"] == 19
    assert row["aggregate_height_at_death"] == 37
    assert row["holes_at_death"] == 35
    assert row["central_tallest"] is True
    assert row["remaining_after_15"] == 5


def test_paired_bootstrap_resamples_whole_seed_differences():
    raw = [{"seed": seed, "pieces_survived": value}
           for seed, value in ((1, 10), (2, 20), (3, 30))]
    other = [{"seed": seed, "pieces_survived": value}
             for seed, value in ((1, 12), (2, 22), (3, 32))]
    result = paired(other, raw)
    assert result["mean_difference"] == 2
    assert result["bootstrap_95ci"] == [2, 2]
    assert bootstrap_mean([1, 3]) == bootstrap_mean([1, 3])
    with pytest.raises(AssertionError, match="seed order"):
        paired(list(reversed(other)), raw)
