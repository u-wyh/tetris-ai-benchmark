"""Parallel milestone evaluation preserves the formal seed protocol."""

from training.evaluation.evaluate import evaluate_model
from training.evaluation.bc_generalization import CHAMPION
from training.evaluation.next_stage_models import choose_workers


def test_parallel_evaluation_matches_serial_seed_rows():
    seeds = [1234000, 1234001, 1234002]
    serial = evaluate_model(CHAMPION, seeds, 80, "speed_parity", 0, workers=1)
    parallel = evaluate_model(CHAMPION, seeds, 80, "speed_parity", 0, workers=3)
    assert [row["seed"] for row in parallel["per_seed"]] == seeds
    assert parallel["per_seed"] == serial["per_seed"]
    assert parallel["aggregate"] == serial["aggregate"]


def test_long_worker_selection_uses_measured_batch_time():
    assert choose_workers({6: 12.0, 8: 9.0, 10: 9.0, 12: 10.0}) == 8
