"""Read-only integrity check for the archived combined placement benchmark."""

import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/ppo_fast_legal_combined_20261024"
DATA = ROOT / "runs/ppo_clear_cadence_counterfactual"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: {actual!r} != {expected!r}")


def main():
    protocol = json.loads((RUN / "protocol.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    check(summary["protocol_sha256"], digest(RUN / "protocol.json"), "protocol hash")
    check(protocol["model_sha256"], digest(ROOT / "runs/ppo_candidate_raw_30m_seed42/final/model.zip"), "model hash")
    old = subprocess.check_output(["git", "show", "5cf0c74:training/evaluation/fast_legal.py"], cwd=ROOT)
    check(protocol["old_source_sha256"], hashlib.sha256(old).hexdigest(), "old source")
    for key, name in (("new_source_sha256", "fast_legal.py"),
                      ("branch_source_sha256", "fast_counterfactual.py")):
        check(protocol[key], digest(ROOT / "training/evaluation" / name), name)
    check(protocol["order"], "old,new,new,old", "order")
    check(protocol["repeats"], 2, "repeats")
    check(protocol["horizon"], 500, "horizon")
    check(summary["cases"], len(protocol["cases"]), "cases")
    old_times, new_times = [], []
    for seed, step in protocol["cases"]:
        row = json.loads((RUN / f"seed_{seed}_step_{step}.json").read_text())
        check((row["seed"], row["step"]), (seed, step), "case identity")
        stem = f"seed_{seed}_step_{step}"
        snapshot = DATA / "states" / f"{stem}.pkl.gz"
        candidate = DATA / "candidates" / f"{stem}.json"
        check(row["snapshot_sha256"], digest(snapshot), "snapshot")
        check(row["candidate_sha256"], digest(candidate), "candidate")
        candidate_data = json.loads(candidate.read_text())
        check(candidate_data["snapshot_sha256"], digest(snapshot), "candidate snapshot")
        check(row["action"], candidate_data["original_action"], "original action")
        check([(x["variant"], x["repeat"]) for x in row["timings"]],
              [("old", 0), ("new", 0), ("new", 1), ("old", 1)], "timing order")
        if len(row["outcome_sha256"]) != 64:
            raise ValueError("invalid outcome digest")
        for entry in row["timings"]:
            seconds = entry["seconds"]
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid timing")
            (old_times if entry["variant"] == "old" else new_times).append(seconds)
    check(summary["paired_runs"], len(old_times), "pairs")
    for key, actual in (("old_seconds", sum(old_times)),
                        ("new_seconds", sum(new_times)),
                        ("speedup", sum(old_times) / sum(new_times))):
        if not math.isclose(summary[key], actual, rel_tol=1e-12):
            raise ValueError(f"{key} mismatch")
    for actual, expected in zip(summary["pair_speedups"],
                                (a / b for a, b in zip(old_times, new_times))):
        if not math.isclose(actual, expected, rel_tol=1e-12):
            raise ValueError("pair speedup mismatch")
    check(len(summary["pair_speedups"]), len(old_times), "pair count")
    print(f"PASS: {len(protocol['cases'])} cases, {len(old_times)} pairs, {summary['speedup']:.6f}x")


if __name__ == "__main__":
    main()
