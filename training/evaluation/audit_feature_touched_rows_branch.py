"""Read-only integrity audit for the archived touched-row branch benchmark."""

import hashlib
import json
import math
import statistics

from training.evaluation.hybrid_queue import PPO, digest
from training.train_ppo import ROOT


def audit():
    run = ROOT / "runs/ppo_fast_feature_touched_rows_branch_20261024"
    data = ROOT / "runs/ppo_clear_cadence_counterfactual"
    protocol_path = run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["protocol_sha256"] == digest(protocol_path)
    assert digest(PPO) == protocol["model_sha256"]
    for key, path in (
        ("old_source_sha256", ROOT / "training/evaluation/fast_candidate_features.py"),
        ("new_source_sha256", ROOT / "training/evaluation/fast_candidate_features_touched_rows.py"),
        ("branch_source_sha256", ROOT / "training/evaluation/fast_counterfactual.py"),
    ):
        assert digest(path) == protocol[key]
    assert protocol["version"] == 1 and protocol["horizon"] == 500
    assert protocol["repeats"] == 2 and protocol["cpu_threads"] == 1
    assert protocol["order"] == "old,new,new,old"

    old, new, hashes = [], [], {}
    for seed, step in protocol["cases"]:
        name = f"seed_{seed}_step_{step}.json"
        path = run / name
        row = json.loads(path.read_text())
        hashes[name] = digest(path)
        assert (row["seed"], row["step"]) == (seed, step)
        candidate_path = data / "candidates" / name
        candidate = json.loads(candidate_path.read_text())
        assert row["candidate_sha256"] == digest(candidate_path)
        assert row["action"] == candidate["original_action"]
        assert row["snapshot_sha256"] == candidate["snapshot_sha256"]
        assert row["snapshot_sha256"] == digest(data / "states" / name.replace(".json", ".pkl.gz"))
        assert len(row["outcome_sha256"]) == hashlib.sha256().digest_size * 2
        assert row["pieces"] == 500 and row["game_over"] is False
        timings = row["timings"]
        assert [(t["repeat"], t["variant"]) for t in timings] == [
            (0, "old"), (0, "new"), (1, "new"), (1, "old")
        ]
        assert all(math.isfinite(t["seconds"]) and t["seconds"] > 0 for t in timings)
        old.extend(t["seconds"] for t in timings if t["variant"] == "old")
        new.extend(t["seconds"] for t in timings if t["variant"] == "new")

    assert summary["result_sha256"] == hashes
    assert summary["cases"] == len(protocol["cases"])
    assert summary["paired_runs"] == len(old) == len(new)
    assert summary["outputs_identical"] is True
    assert math.isclose(summary["old_total_seconds"], sum(old))
    assert math.isclose(summary["new_total_seconds"], sum(new))
    assert math.isclose(summary["speedup_total"], sum(old) / sum(new))
    assert math.isclose(summary["old_median_seconds"], statistics.median(old))
    assert math.isclose(summary["new_median_seconds"], statistics.median(new))
    assert len(summary["paired_speedups"]) == len(old)
    assert all(math.isclose(a / b, ratio) for a, b, ratio in zip(old, new, summary["paired_speedups"]))
    return {"cases": len(protocol["cases"]), "paired_runs": len(old),
            "speedup": summary["speedup_total"], "row_sha256": hashes}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
