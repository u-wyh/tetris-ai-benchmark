import gzip
import json

from training.evaluation import ppo_death_capture as capture
from training.evaluation import ppo_death_replay as replay


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
