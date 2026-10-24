"""Read-only consistency audit of the archived landing-sort branch benchmark."""

import hashlib
import json
import math
import subprocess

from training.evaluation.hybrid_queue import PPO, digest
from training.train_ppo import ROOT


def audit(run_name="ppo_fast_legal_landing_sort_20261024", baseline_ref="ceed260"):
    run = ROOT / "runs" / run_name
    data = ROOT / "runs/ppo_clear_cadence_counterfactual"
    protocol_path = run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["protocol_sha256"] == digest(protocol_path)
    assert digest(PPO) == protocol["model_sha256"]
    baseline = subprocess.check_output(
        ["git", "show", f"{baseline_ref}:training/evaluation/fast_legal.py"], cwd=ROOT
    )
    assert hashlib.sha256(baseline).hexdigest() == protocol["old_source_sha256"]
    assert digest(run / "candidate_fast_legal.py") == protocol["new_source_sha256"]
    assert digest(ROOT / "training/evaluation/fast_counterfactual.py") == protocol["branch_source_sha256"]
    assert protocol["order"] == "old,new,new,old"
    assert protocol["repeats"] == 2 and protocol["horizon"] == 500
    assert protocol["cpu_threads"] == 1

    old_times, new_times, row_hashes = [], [], {}
    for seed, step in protocol["cases"]:
        name = f"seed_{seed}_step_{step}.json"
        row_path = run / name
        row_hashes[name] = digest(row_path)
        row = json.loads(row_path.read_text())
        assert (row["seed"], row["step"]) == (seed, step)
        candidate_path = data / "candidates" / name
        candidate = json.loads(candidate_path.read_text())
        assert digest(candidate_path) == row["candidate_sha256"]
        assert candidate["original_action"] == row["action"]
        assert candidate["snapshot_sha256"] == row["snapshot_sha256"]
        assert digest(data / "states" / name.replace(".json", ".pkl.gz")) == row["snapshot_sha256"]
        assert len(row["outcome_sha256"]) == 64
        timings = row["timings"]
        assert [(x["repeat"], x["variant"]) for x in timings] == [
            (0, "old"), (0, "new"), (1, "new"), (1, "old")
        ]
        assert all(math.isfinite(x["seconds"]) and x["seconds"] > 0 for x in timings)
        old_times.extend(x["seconds"] for x in timings if x["variant"] == "old")
        new_times.extend(x["seconds"] for x in timings if x["variant"] == "new")

    assert summary["cases"] == len(protocol["cases"])
    assert summary["paired_runs"] == len(old_times) == len(new_times)
    assert summary["outputs_identical"] is True
    assert math.isclose(summary["old_seconds"], sum(old_times))
    assert math.isclose(summary["new_seconds"], sum(new_times))
    assert math.isclose(summary["speedup"], sum(old_times) / sum(new_times))
    assert len(summary["pair_speedups"]) == len(old_times)
    assert all(math.isclose(a / b, reported) for a, b, reported in zip(
        old_times, new_times, summary["pair_speedups"]
    ))
    return {"cases": len(protocol["cases"]), "paired_runs": len(old_times),
            "speedup": summary["speedup"], "row_sha256": row_hashes}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
