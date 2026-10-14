"""V2 active selection and resumed branch invariants on frozen V1 data."""
import gzip
import hashlib
import json

from training.evaluation.local_correction_active_v2 import (MIN_SPACING, RANDOM_QUOTA,
                                                             SEEDS, V2, _load_seed, choose)
from training.evaluation.local_correction_v2_report import _reasons
from training.evaluation.local_correction_v2_materialize import _canonical, main as materialize, normalize
from training.imitation_dataset import digest


def test_active_selection_is_isolated_diverse_and_deterministic():
    selected = [choose(seed) for seed in SEEDS]
    assert sum(len(row["steps"]) for row in selected) >= 500
    assert sum(row["random_count"] for row in selected) >= 80
    for row in selected:
        _, _, old = _load_seed(row["seed"])
        assert not set(row["steps"]) & set(old["branch_steps"])
        assert all(b - a >= MIN_SPACING for a, b in zip(row["steps"], row["steps"][1:]))
        assert row == choose(row["seed"])
        assert row["random_count"] <= RANDOM_QUOTA


def test_completed_branch_identity_and_original_label_version():
    seed = 730015
    selection = choose(seed)
    for step in selection["steps"]:
        path = V2 / "branches" / f"seed_{seed}" / f"step_{step}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        assert row["seed"] == seed and row["step"] == step
        assert row["source_sha256"] == selection["source_sha256"]
        assert row["evidence"]["category"] in {"A", "B"}
        assert 0 <= row["ppo_action_probability"] <= 1


def test_hold_audit_public_next_and_rejection_reason_coverage():
    hold = json.loads((V2 / "hold_audit.json").read_text())["records"]
    assert len(hold) == 21
    assert sum(row["hold_flag_differs"] for row in hold) == 12
    assert all(not row["future_piece_state_differs"] for row in hold
               if not row["hold_flag_differs"])
    diagnosis = json.loads((V2 / "diagnosis.json").read_text())
    assert len(diagnosis["rejected"]) == 202
    assert all(_reasons(row) for row in diagnosis["rejected"])


def test_materialized_record_identity_and_offline_only_field():
    path = next((V2 / "branches").glob("seed_*/step_*.json"))
    branch = json.loads(path.read_text())
    row = normalize(branch)
    assert row == normalize(branch)
    assert row["episode_id"] == f"seed_{branch['seed']}_episode_0"
    assert row["step"] == branch["step"]
    assert row["ppo_action"] in row["legal_action_ids"]
    assert row["bc_action"] in row["legal_action_ids"]
    assert len(row["record_sha256"]) == 64
    assert 0 <= row["ppo_action_probability"] <= 1


def test_complete_dataset_manifest_and_resume_are_immutable():
    manifest = json.loads((V2 / "dataset_manifest.json").read_text())
    assert manifest["branch_count"] == 556
    assert all(digest(V2 / path) == sha for path, sha in manifest["branch_hashes"].items())
    before = digest(V2 / "dataset_rows.jsonl.gz")
    meta = materialize()
    assert meta["dataset_sha256"] == before == digest(V2 / "dataset_rows.jsonl.gz")
    assert meta["source_manifest_sha256"] == digest(V2 / "dataset_manifest.json")
    with gzip.open(V2 / "dataset_rows.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 556
    for record in records:
        expected = record.pop("record_sha256")
        assert hashlib.sha256(_canonical(record).encode()).hexdigest() == expected
