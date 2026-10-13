"""DAgger labels match the original teacher and BC data schema."""
import gzip
import json

import pytest
from sb3_contrib import MaskablePPO

from training.dagger_dataset import CHAMPION, capture, collect
from training.imitation_bc import batch
from training.imitation_dataset import TEACHER_HASH, label_public


def test_dagger_real_public_states_and_resume(tmp_path):
    if not CHAMPION.exists():
        pytest.skip("Frozen BC checkpoint unavailable")
    model = MaskablePPO.load(str(CHAMPION), device="cpu")
    meta = capture(1249999, tmp_path, model, limit=4)
    assert meta["samples"] == 4
    assert meta == capture(1249999, tmp_path, model, limit=4)
    with gzip.open(tmp_path / "seed_1249999.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 4
    for row in rows:
        assert row["teacher_hash"] == TEACHER_HASH
        assert row["action"] == label_public(row["public"])
        assert row["action"] in row["legal_ids"]
        assert row["learner_action"] in row["legal_ids"]
        assert all("rng" not in key.lower() and "bag" not in key.lower() for key in row["public"])
        obs, mask, actions = batch([row], "cpu")
        assert bool(mask[0, actions[0]])
    with pytest.raises(RuntimeError, match="integrity"):
        capture(1249999, tmp_path, model, limit=5)


def test_collection_resume_keeps_original_commit_provenance(tmp_path, monkeypatch):
    if not CHAMPION.exists():
        pytest.skip("Frozen BC checkpoint unavailable")
    import training.dagger_dataset as dagger
    output = tmp_path / "data"
    assert collect(output, [1248888], limit=4, target=4) == 4
    original = json.loads((output / "config.json").read_text())["git_commit"]
    monkeypatch.setattr(dagger.subprocess, "check_output", lambda *args, **kwargs: "new-doc-commit\n")
    assert collect(output, [1248888], limit=4, target=4) == 4
    assert json.loads((output / "config.json").read_text())["git_commit"] == original
