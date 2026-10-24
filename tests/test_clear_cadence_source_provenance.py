"""The accelerated branch's executed modules must be part of provenance."""

import pytest

from training.evaluation import source_provenance


@pytest.mark.parametrize("path", [
    "training/evaluation/fast_candidate_features.py",
    "training/evaluation/fast_legal.py",
])
def test_fast_branch_source_change_is_detected(tmp_path, monkeypatch, path):
    for name in source_provenance.SOURCE_PATHS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"original")
    monkeypatch.setattr(source_provenance.importlib.metadata, "version", lambda _: "test")
    recorded = source_provenance.snapshot(tmp_path)
    assert path in recorded["source_sha256"]
    (tmp_path / path).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source or package versions changed"):
        source_provenance.verify(tmp_path, recorded)
