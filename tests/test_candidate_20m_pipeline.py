"""Selection remains frozen before the independent test and retries are safe."""

import json

import pytest

from training.evaluation import candidate_20m_pipeline as pipeline
from training.train_ppo import atomic_json


def test_independent_seeds_are_new_and_disjoint():
    proto = pipeline.frozen_protocol()
    assert proto["test_seeds"] == list(range(400000, 400100))
    assert not set(proto["test_seeds"]) & set(proto["validation_seeds"])
    assert not set(proto["test_seeds"]) & set(range(300000, 300100))


def test_selection_is_frozen_and_recoverable(tmp_path, monkeypatch):
    # The real 400000-series independent results now exist. Isolate this unit
    # test's seed-audit root so it exercises pre-evaluation selection behavior.
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "RUN", tmp_path)
    proto = pipeline.frozen_protocol()
    for step, pieces in zip(pipeline.STEPS, (10000, 16000, 8000)):
        rows = [{"seed": seed, "pieces_survived": pieces}
                for seed in proto["validation_seeds"]]
        atomic_json(tmp_path / f"validation_{step}.json",
                    {"per_seed": rows, "aggregate": {"mean_pieces": pieces,
                     "capped_count": 0, "p10_pieces": pieces}})
    first = pipeline.select({}, proto)
    assert first["selected_step"] == pipeline.STEPS[1]
    assert pipeline.select({}, proto) == first
    assert first["selected_before_independent_test"]
    changed = json.loads((tmp_path / f"validation_{pipeline.STEPS[1]}.json").read_text())
    changed["aggregate"]["mean_pieces"] = 0
    atomic_json(tmp_path / f"validation_{pipeline.STEPS[1]}.json", changed)
    with pytest.raises(RuntimeError, match="Frozen selection differs"):
        pipeline.select({}, proto)
