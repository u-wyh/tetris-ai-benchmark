import gzip
import json

from training.evaluation import ppo_death_capture as capture
from training.evaluation import ppo_death_replay as replay
from training.evaluation import ppo_death_counterfactual as cf
from training.evaluation.ppo_death_report import compare
from training.evaluation.hybrid_queue import load_models
from training.evaluation.ppo_death_analysis import match_survivor, windows


def test_atomic_short_capture_and_rng_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "RUN", tmp_path)
    monkeypatch.setattr(capture, "CAP", 24)
    monkeypatch.setattr(capture, "CHECKPOINT_EVERY", 8)
    monkeypatch.setattr(replay, "CAP", 24)
    model = capture.load_ppo()
    row = capture.capture(992099, model)
    assert row["capped"] and not row["game_over"] and row["pieces"] == 24
    trace, state, meta = capture.trace_paths(992099)
    assert trace.exists() and state.exists() and meta.exists()
    with gzip.open(trace, "rt") as file:
        steps = [json.loads(line) for line in file]
    assert len(steps) == 24
    assert steps[0]["four_counts"] == {"100": 0, "500": 0, "1000": 0}
    assert steps[0]["action"] in [candidate["action"] for candidate in steps[0]["top5"]]
    assert replay.validate(992099, model, all_segments=True)["valid"]
    assert capture.capture(992099, model) == row  # Verified resume, no duplicate episode.
    assert not list(tmp_path.rglob("*.tmp"))


def test_death_windows_and_episode_matched_control():
    def row(step, height, holes):
        return {"step": step, "max_height": height, "holes": holes, "covered_holes": holes,
                "deepest_well_depth": 3, "birth_zone": 0, "cleared": 4 if step%5==0 else 0,
                "used_hold": step%3==0, "action_probability": .6,
                "four_counts": {"1000": 2}, "total_height": 20, "row_transitions": 30,
                "column_transitions": 30, "bumpiness": 4, "deepest_well_column": 9,
                "four_rate_100": .1, "hold_rate_100": .2, "score": step*100,
                "heights": [height]*10}
    rows = [row(i, 10+i//100, i//200) for i in range(600)]
    result = windows(rows)
    assert result["20"]["start_step"] == 580
    assert result["500"]["actual_width"] == 500
    match = match_survivor(rows[300], {999: [row(300, 12, 1)],
                                        998: [row(300, 17, 8)]}, 123, 300)
    assert match["seed"] == 999


def test_counterfactual_branch_snapshot_isolation_and_censoring(tmp_path, monkeypatch):
    for module in (capture, replay, cf):
        monkeypatch.setattr(module, "RUN", tmp_path)
        monkeypatch.setattr(module, "CAP", 24)
    monkeypatch.setattr(capture, "CHECKPOINT_EVERY", 8)
    models = load_models()
    capture.capture(992098, models["ppo"])
    meta, rows, checkpoints = replay.load_episode(992098)
    item = {"seed": 992098, "step": 18, "death_distance": 6,
            "tier": "1-100", "source_trace_sha256": meta["trace_sha256"]}
    snapshot = cf.materialize_state(item, rows, checkpoints)
    assert snapshot["public_hash"] == rows[18]["public_hash"]
    choices = cf.candidates(item, rows, models)["choices"]
    assert all(choice["ppo_rank"] >= 1 for choice in choices)
    first = cf.branch(992098, 18, choices[0], 12, models)
    assert first["pieces"] == 12 and first["censored"]
    assert first["initial_public_hash"] == snapshot["public_hash"]
    assert cf.branch(992098, 18, choices[0], 12, models) == first
    assert cf.materialize_state(item, rows, checkpoints) == snapshot


def test_common_survival_score_and_right_censor():
    base = {"pieces": 2, "game_over": True, "censored": False,
            "initial_score": 0, "scores": [100,200], "four_counts": [0,1]}
    alt = {"pieces": 300, "game_over": False, "censored": True,
           "initial_score": 0, "scores": [110,210]+[210]*298,
           "four_counts": [0,1]+[1]*298}
    result = compare(base,alt)
    assert result["common_pieces"] == 2 and result["strong_local_correction"]
    base["censored"] = True
    base["game_over"] = False
    assert not compare(base,alt)["strong_local_correction"]
