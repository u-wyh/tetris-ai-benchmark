"""Read-only integrity audit of the archived PPO branch throughput benchmark."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    protocol = json.loads((run / "protocol.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert protocol["version"] == summary["version"]
    assert summary["all_outputs_identical"] is True
    assert summary["cases"] == len(protocol["cases"])
    assert summary["horizon"] == protocol["horizon"]
    assert len({case["seed"] for case in protocol["cases"]}) == len(protocol["cases"])
    names = {f"seed_{case['seed']}.json" for case in protocol["cases"]}
    assert set(summary["file_sha256"]) == names
    durations = {"legacy": [], "fast": []}
    for case in protocol["cases"]:
        name = f"seed_{case['seed']}.json"
        assert sha256(run / name) == summary["file_sha256"][name], name
        row = json.loads((run / name).read_text())
        assert row["case"] == case, name
        assert row["outcome"]["action"] == case["action"], name
        assert row["outcome"]["horizon"] == protocol["horizon"], name
        assert len(row["timings"]) == 2 * protocol["repeats"], name
        assert {(t["repeat"], t["variant"]) for t in row["timings"]} == {
            (rep, variant) for rep in range(protocol["repeats"])
            for variant in durations}, name
        for timing in row["timings"]:
            seconds = timing["seconds"]
            assert isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0
            durations[timing["variant"]].append(seconds)
    count = len(protocol["cases"]) * protocol["repeats"]
    assert summary["timed_branches_per_variant"] == count
    for variant in durations:
        assert len(durations[variant]) == count
        assert math.isclose(sum(durations[variant]), summary[f"{variant}_seconds_total"], rel_tol=1e-12)
        assert math.isclose(statistics.median(durations[variant]), summary[f"{variant}_seconds_median"], rel_tol=1e-12)
    assert math.isclose(summary["speedup_total"],
                        sum(durations["legacy"]) / sum(durations["fast"]), rel_tol=1e-12)
    return {"cases": len(names), "timings": 2 * count,
            "speedup_total": summary["speedup_total"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), sort_keys=True))
