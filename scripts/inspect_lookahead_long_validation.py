"""Reproduce the frozen Lookahead report from a local research checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def read_record(raw):
    record = json.loads(raw)
    canonical = json.dumps(
        record["data"], sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != record["sha256"]:
        raise ValueError("Archived JSON checksum mismatch")
    return record["data"]


def verify_sources(checkout, sources):
    for relative, expected in sources.items():
        source = (checkout / relative).resolve()
        if not source.is_relative_to(checkout):
            raise ValueError(f"Frozen source path is outside the checkout: {relative}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source differs from protocol: {relative}")


def with_inventory(group, evaluation):
    completed = 0
    missing = {}
    checkpoints = {}
    for variant, seeds in group["completed_seed_ids"].items():
        completed += len(seeds)
        absent = sorted(set(group["planned_seeds"]) - set(seeds))
        missing[variant] = absent
        checkpoints[variant] = [
            seed for seed in absent
            if (evaluation / group["label"] / variant / f"seed_{seed}.checkpoint.pkl").is_file()
        ]
    return {
        **group,
        "completed_games": completed,
        "planned_games": len(group["planned_seeds"]) * len(missing),
        "missing_seed_ids": missing,
        "checkpoint_files_present": checkpoints,
    }


def inspect(checkout):
    run = checkout / "runs/lookahead_long_validation"
    raw_protocol = (run / "protocol.json").read_bytes()
    protocol = read_record(raw_protocol)
    verify_sources(checkout, protocol["sources"])

    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation_report as report

    evaluation = run / "evaluation"
    stop_path = run / "stop.json"
    stop = read_record(stop_path.read_bytes()) if stop_path.is_file() else None
    previous_run = report.study.RUN
    # The frozen summarizer emits two intermediate summaries. Isolate those
    # writes while retaining its original artifact checks and statistics.
    with TemporaryDirectory(prefix="lookahead-report-") as directory:
        scratch = Path(directory)
        (scratch / "protocol.json").write_bytes(raw_protocol)
        (scratch / "evaluation").symlink_to(evaluation, target_is_directory=True)
        report.study.RUN = scratch
        try:
            main = report.summarize(
                "100k", protocol["groups"]["main"], ["A", "B", "C", "E", "BC"],
                protocol["main_cap"],
            )
            extension = report.summarize(
                "200k", protocol["groups"]["extension"], ["BC", "C", "E"],
                protocol["extension_cap"],
            )
        finally:
            report.study.RUN = previous_run

    return {
        "archive_protocol_sha256": hashlib.sha256(raw_protocol).hexdigest(),
        "frozen_protocol_git": protocol["git"],
        "verified_frozen_sources": len(protocol["sources"]),
        "stop": stop,
        "main": with_inventory(main, evaluation),
        "extension": with_inventory(extension, evaluation),
        "decision": report.decision(main, extension),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Research checkout containing the frozen run")
    args = parser.parse_args()
    print(json.dumps(inspect(args.checkout.resolve()), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
