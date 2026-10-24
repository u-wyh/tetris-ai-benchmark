"""Read-only integrity audit for the archived feature-limit branch benchmark."""

import hashlib
import json
import math
import statistics
import subprocess

from training.evaluation.hybrid_queue import PPO, digest
from training.train_ppo import ROOT


def audit():
    run = ROOT / "runs/ppo_fast_feature_limits_branch_20261024"
    data = ROOT / "runs/ppo_clear_cadence_counterfactual"
    protocol_path = run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["protocol_sha256"] == digest(protocol_path)
    assert digest(PPO) == protocol["model_sha256"]
    old = subprocess.check_output(
        ["git", "show", "65e5c0c:training/evaluation/fast_candidate_features.py"], cwd=ROOT
    )
    assert hashlib.sha256(old).hexdigest() == protocol["old_source_sha256"]
    for key, path in (
        ("new_source_sha256", "training/evaluation/fast_candidate_features.py"),
        ("branch_source_sha256", "training/evaluation/fast_counterfactual.py"),
    ):
        assert digest(ROOT / path) == protocol[key]
    assert protocol["order"] == "old,new,new,old" and protocol["repeats"] == 2
    expected_files = {f"seed_{seed}_step_{step}.json" for seed, step in protocol["cases"]}
    assert set(summary["result_sha256"]) == expected_files
    assert summary["cases"] == len(expected_files)
    old_times, new_times = [], []
    for seed, step in protocol["cases"]:
        name = f"seed_{seed}_step_{step}.json"
        path = run / name
        assert digest(path) == summary["result_sha256"][name]
        row = json.loads(path.read_text())
        assert (row["seed"], row["step"]) == (seed, step)
        candidate_path = data / "candidates" / name
        candidate = json.loads(candidate_path.read_text())
        assert digest(candidate_path) == row["candidate_sha256"]
        assert row["action"] == candidate["original_action"]
        assert row["snapshot_sha256"] == candidate["snapshot_sha256"]
        snapshot = data / "states" / name.replace(".json", ".pkl.gz")
        assert digest(snapshot) == row["snapshot_sha256"]
        assert len(row["outcome_sha256"]) == 64 and row["pieces"] <= protocol["horizon"]
        timings = row["timings"]
        assert [(x["repeat"], x["variant"]) for x in timings] == [
            (0, "old"), (0, "new"), (1, "new"), (1, "old")
        ]
        assert all(math.isfinite(x["seconds"]) and x["seconds"] > 0 for x in timings)
        old_times.extend(x["seconds"] for x in timings if x["variant"] == "old")
        new_times.extend(x["seconds"] for x in timings if x["variant"] == "new")
    assert summary["paired_runs"] == len(old_times) == len(new_times)
    assert summary["outputs_identical"] is True
    assert math.isclose(summary["old_total_seconds"], sum(old_times))
    assert math.isclose(summary["new_total_seconds"], sum(new_times))
    assert math.isclose(summary["speedup_total"], sum(old_times) / sum(new_times))
    assert all(math.isclose(a, b) for a, b in zip(
        summary["paired_speedups"], [a / b for a, b in zip(old_times, new_times)]
    ))
    assert math.isclose(summary["old_median_seconds"], statistics.median(old_times))
    assert math.isclose(summary["new_median_seconds"], statistics.median(new_times))
    return {"cases": len(expected_files), "paired_runs": len(old_times),
            "speedup_total": summary["speedup_total"]}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
