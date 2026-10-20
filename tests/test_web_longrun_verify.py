import json

import pytest

from training.evaluation.web_longrun_verify import atomic, resume, sealed, verify_chunk
from training.tetris_core.core import TetrisCore


def test_atomic_result_rejects_corrupt_content(tmp_path):
    target = tmp_path / "result.json"
    atomic(target, b'{"ok":true}')
    assert sealed(target) == b'{"ok":true}'
    target.write_bytes(b'{"ok":false}')
    with pytest.raises(RuntimeError, match="invalid atomic result"):
        sealed(target)


def test_core_browser_chunk_resume_and_legality(tmp_path):
    seed = 12345
    core = TetrisCore(seed)
    before = {**core.get_public_observation(), "placed": 0}
    legal = [p["actionId"] for p in core.get_legal_placements()]
    action = legal[0]
    core.step(action)
    after = {**core.get_public_observation(), "placed": 1}
    row = {"before": before, "after": after, "action": action, "legal_ids": legal}
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps({"chunk": 1, "start_piece": 0, "end_piece": 1,
                                   "entries": [row], "browser_save": "saved",
                                   "game_over": core.phase == "over", "metrics": {}}))
    result = verify_chunk(tmp_path, seed, pending)
    assert result["end_piece"] == 1
    assert resume(tmp_path, seed)["browser_save"] == "saved"
    row["before"] = after
    row["legal_ids"] = [p["actionId"] for p in core.get_legal_placements()]
    row["action"] = 1840
    pending.write_text(json.dumps({"chunk": 2, "start_piece": 1, "end_piece": 2,
                                   "entries": [row], "browser_save": "saved",
                                   "game_over": False, "metrics": {}}))
    with pytest.raises(AssertionError, match="Illegal"):
        verify_chunk(tmp_path, seed, pending)
