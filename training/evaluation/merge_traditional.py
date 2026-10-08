"""Import completed, compatible seed records into a traditional benchmark run.

Use while the target is stopped, then resume the target to build its aggregate.
"""
import argparse
import json
import os
from pathlib import Path

from training.evaluation.traditional import atomic, now

COMPATIBLE_V1 = ("max_pieces", "strategy", "weights", "lookahead", "jitter", "hold", "code_sha256")
COMPATIBLE_V2 = ("max_pieces", "strategy", "policy", "code_sha256")


def merge(target, sources):
    target = Path(target)
    destination_config = json.loads((target / "config.json").read_text())
    compatible = COMPATIBLE_V2 if destination_config["strategy"] == "v2-1" else COMPATIBLE_V1
    manifest = target / "merge_manifest.json"
    if manifest.exists():
        raise FileExistsError("Merge manifest already exists")
    pending = []
    merged_seeds = set(destination_config["seeds"])
    for source in map(Path, sources):
        config = json.loads((source / "config.json").read_text())
        if any(config[key] != destination_config[key] for key in compatible):
            raise ValueError(f"Incompatible benchmark configuration: {source}")
        merged_seeds.update(config["seeds"])
        if json.loads((source / "status.json").read_text())["state"] != "complete":
            raise ValueError(f"Source is not complete: {source}")
        for seed in config["seeds"]:
            path = source / f"seed_{seed}.json"
            row = json.loads(path.read_text())
            if (row["seed"] != seed or len(row["actions"]) != row["pieces_survived"]
                    or len(row["decision_ms"]) != row["pieces_survived"]
                    or row["game_over"] == row["truncated"]):
                raise ValueError(f"Invalid seed record: {path}")
            if (target / path.name).exists() or any(p.name == path.name for p, _ in pending):
                raise FileExistsError(f"Target already has seed {seed}; refusing overwrite")
            pending.append((path, config["git_commit"]))
    imported = []
    for path, git_commit in pending:
        try:
            # Same filesystem, atomic and refuses to replace any target seed.
            os.link(path, target / path.name)
        except FileExistsError as error:
            raise FileExistsError(f"Target already has {path.name}; refusing overwrite") from error
        imported.append(dict(seed=int(path.stem.split("_")[1]), source=str(path.parent),
                             source_git_commit=git_commit))
    atomic(manifest, dict(imported=imported, merged_at=now(),
                          note="Independent CPU processes ran concurrently; wall-clock decision timings include system load."))
    if sorted(merged_seeds) != destination_config["seeds"]:
        destination_config["seeds"] = sorted(merged_seeds)
        atomic(target / "config.json", destination_config)
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.target, args.sources), indent=2))


if __name__ == "__main__":
    main()
