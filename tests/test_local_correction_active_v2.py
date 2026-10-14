"""V2 active selection and resumed branch invariants on frozen V1 data."""
import json

from training.evaluation.local_correction_active_v2 import (MIN_SPACING, RANDOM_QUOTA,
                                                             SEEDS, V2, _load_seed, choose)


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
