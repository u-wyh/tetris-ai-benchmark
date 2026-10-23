import gzip
import pickle

import pytest

from training.evaluation.ppo_clear_cadence_capture import save_snapshot


def test_snapshot_resume_accepts_legacy_gzip_timestamp(tmp_path):
    path = tmp_path / "snapshot.pkl.gz"
    state = {"pieces": 25, "board": [1, 2, 3]}
    raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(gzip.compress(raw, compresslevel=3, mtime=12345))
    before = path.read_bytes()
    assert save_snapshot(path, state)
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="Orphan snapshot mismatch"):
        save_snapshot(path, {**state, "pieces": 26})


def test_new_snapshot_is_deterministic(tmp_path):
    state = {"pieces": 25}
    paths = [tmp_path / "a.pkl.gz", tmp_path / "b.pkl.gz"]
    assert save_snapshot(paths[0], state) == save_snapshot(paths[1], state)
    assert paths[0].read_bytes() == paths[1].read_bytes()
