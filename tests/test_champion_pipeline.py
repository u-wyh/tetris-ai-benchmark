"""Checkpoint selection and resumable evaluator preserve the frozen protocol."""

import json

from sb3_contrib import MaskablePPO

import training.evaluation.champion_pipeline as pipeline
from training.evaluation.evaluate import evaluate_model
from training.train_ppo import atomic_json
from training.train_vector_transaction import sha256


def test_seed_evaluator_matches_existing_deterministic_protocol():
    model_path = (pipeline.SOURCE / "milestones/step_007000064/model.zip")
    model = MaskablePPO.load(str(model_path), device="cpu")
    actual = pipeline.evaluate_seed(model, 100000, 30)
    expected = evaluate_model(model_path, [100000], 30, "smoke", 7_000_064)["per_seed"][0]
    for field in ("seed", "pieces_survived", "lines", "score", "episode_reward",
                  "game_over", "survived_cap"):
        assert actual[field] == expected[field]
    pipeline.validate_row(actual, 100000, 30)


def test_per_seed_result_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RUN", tmp_path)
    model_path = pipeline.SOURCE / "milestones/step_007000064/model.zip"
    digest = sha256(model_path)
    first = pipeline.rows_for("validation_7000064", model_path, digest, 7_000_064,
                              [100000], 30)
    monkeypatch.setattr(pipeline, "evaluate_seed",
                        lambda *args: (_ for _ in ()).throw(AssertionError("reran seed")))
    second = pipeline.rows_for("validation_7000064", model_path, digest, 7_000_064,
                               [100000], 30)
    assert first == second


def test_champion_selection_uses_validation_before_final(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RUN", tmp_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "SELECTION_REPORT", tmp_path / "selection.md")
    monkeypatch.setattr(pipeline, "FINAL_REPORT", tmp_path / "final.md")
    seeds = pipeline.protocol()["validation_seeds"]
    for step, survived in zip(pipeline.STEPS, (15000, 18000, 12000, 14000)):
        rows = [{"seed": seed, "pieces_survived": survived,
                 "game_over": True, "survived_cap": False, "lines": 0,
                 "score": 0, "episode_reward": 0.0, "new_holes_events": 0,
                 "new_holes_total": 0,
                 "holes_at_end": 1, "height_at_end": 20, "wall_seconds": 1.0}
                for seed in seeds]
        atomic_json(tmp_path / f"validation_{step}.json",
                    {"seeds": seeds, "max_pieces": 20000, "per_seed": rows,
                     "aggregate": pipeline.summarize(rows, 20000)})
    atomic_json(tmp_path / "protocol.json", pipeline.protocol())
    # A tempting final-test file cannot enter the selection calculation.
    atomic_json(tmp_path / "final_100.json", {"best_step": pipeline.STEPS[-1]})
    decision = pipeline.comparison()
    assert decision["selected_step"] == pipeline.STEPS[1]
    assert decision["highest_mean_step"] == pipeline.STEPS[1]
    assert decision["near_candidates"] == [pipeline.STEPS[1]]
    assert decision["paired"][f"{pipeline.STEPS[1]}_minus_{pipeline.STEPS[0]}"]["mean_difference"] == 3000
    rows = [{"seed": seed, "pieces_survived": 1000, "lines": 100, "score": 1000,
             "episode_reward": 0.0, "game_over": True, "survived_cap": False,
             "new_holes_events": 0, "new_holes_total": 0, "holes_at_end": 1,
             "height_at_end": 20, "wall_seconds": 1.0}
            for seed in range(200000, 200100)]
    result = {"checkpoint": {"selected_step": decision["selected_step"],
                             "model_sha256": "test-sha"},
              "per_seed": rows, "aggregate": pipeline.summarize(rows, 50000)}
    atomic_json(tmp_path / "final_100.json", result)
    atomic_json(tmp_path / "raw_comparison.json",
                {"raw_aggregate": result["aggregate"],
                 "paired": {"mean_difference": 0, "bootstrap_95ci": [0, 0]}})
    atomic_json(tmp_path / "traditional_plan.json",
                {"sequential_cost_estimates": {"Legacy V1": {
                    "100_seeds_x_50000_cap_hours_upper_bound": 80}}})
    (tmp_path / "CHANGELOG.md").write_text("# 修改日志\n\n## 2026-10-10\n")
    pipeline.reports()
    assert (tmp_path / "selection.md").exists()
    assert "1000.0" in (tmp_path / "final.md").read_text()
