"""Explicit-only 100-seed final test; the trainer never imports or invokes this CLI."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from training.long_run import SEEDS_FILE
from training.train_transaction import fsync_dir, fsync_tree
from training.train_vector_transaction import sha256
from training.train_ppo import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    if sha256(SEEDS_FILE) != config["evaluation_seed_sha256"]:
        raise RuntimeError("Fixed evaluation seed file changed")
    model = run_dir / "best" / "model.zip"
    if not model.is_file():
        raise FileNotFoundError("No committed best model is available")
    checksum = json.loads((run_dir / "best" / "checksum.json").read_text())["model_sha256"]
    if sha256(model) != checksum:
        raise RuntimeError("Best model checksum failed")
    step = json.loads((run_dir / "best" / "metadata.json").read_text())["committed_steps"]
    final_dir = run_dir / "evaluations" / "final"
    working_dir = run_dir / "evaluations" / "working"
    final_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    destination = final_dir / f"step_{step:09d}.json"
    if destination.exists():
        raise FileExistsError("Final test already exists for this best checkpoint")
    pending = working_dir / f"final_step_{step:09d}"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    command = [sys.executable, "-m", "training.evaluation.evaluate",
               "--model", str(model), "--seeds-file", str(SEEDS_FILE),
               "--seed-set", "final_test", "--seed-count", str(config["final_test_seeds"]),
               "--max-pieces", str(config["final_test_max_pieces"]),
               "--protocol", "final_test", "--committed-steps", str(step),
               "--output", str(pending / "result.json")]
    subprocess.run(command, cwd=ROOT, check=True)
    result = json.loads((pending / "result.json").read_text())
    if (result["num_seeds"] != 100 or result["max_pieces"] != 50000
            or len(result["per_seed"]) != 100):
        raise RuntimeError("Incomplete final test result")
    fsync_tree(pending)
    os.replace(pending / "result.json", destination)
    fsync_dir(final_dir)
    shutil.rmtree(pending)
    print(destination)


if __name__ == "__main__":
    main()
