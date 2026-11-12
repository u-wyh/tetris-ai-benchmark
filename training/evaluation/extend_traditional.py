"""Prepare a larger seed set by importing completed same-cap benchmark seeds."""
import argparse
import json
from pathlib import Path

from training.evaluation.traditional import atomic, now
from training.evaluation.longlife import read_run


def prepare(source, output, seeds):
    source, output = Path(source), Path(output)
    config = json.loads((source / "config.json").read_text())
    mode = "v1" if config["strategy"] == "v1-adapter-1" else config["policy"]["mode"]
    original, rows = read_run(source, mode)
    if len(seeds) != len(set(seeds)) or not set(original["seeds"]).issubset(seeds):
        raise ValueError("New seed set must be unique and contain every completed source seed")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers")
    output.mkdir(parents=True, exist_ok=False)
    extended = dict(original, seeds=list(seeds), created_at=now(),
                    extended_from=str(source), imported_completed_seeds=list(original["seeds"]))
    atomic(output / "config.json", extended)
    for seed, row in rows.items():
        atomic(output / f"seed_{seed}.json", row)
    atomic(output / "import_manifest.json", dict(source=str(source),
          source_git_commit=original["git_commit"], source_code_sha256=original["code_sha256"],
          imported_seeds=list(original["seeds"]), created_at=now()))
    atomic(output / "status.json", dict(state="prepared", completed_seeds=len(rows),
          total_seeds=len(seeds), updated_at=now()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.seeds))


if __name__ == "__main__":
    main()
