import gzip
import json

import pytest

from training.imitation_dataset import capture, digest, label_public
from training.imitation_bc import batch, records
from training.tetris_core import TetrisCore


def test_teacher_shard_is_public_legal_and_resumable(tmp_path):
    meta = capture(900001, tmp_path, limit=8)
    assert meta["samples"] == 8
    assert capture(900001, tmp_path, limit=8) == meta
    assert len(list(tmp_path.glob("seed_*.jsonl.gz"))) == 1
    core = TetrisCore(900001)
    with gzip.open(tmp_path / "seed_900001.jsonl.gz", "rt") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            public = core.get_public_observation()
            assert row["public"] == public
            assert row["step"] == index
            assert row["action"] == label_public(public, core.get_legal_placements())
            assert row["action"] in row["legal_ids"]
            core.step(row["action"])
    obs, mask, actions = batch(list(records(tmp_path, [900001])), "cpu")
    assert bool(mask.gather(1, actions[:, None]).all())
    with pytest.raises(RuntimeError):
        capture(900001, tmp_path, limit=9)
