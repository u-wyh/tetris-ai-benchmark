"""Create an independent, auditable continuation from a completed Candidate run."""

import argparse
import fcntl
import json
import os
import pickle
import shutil
import copy
from pathlib import Path

from training.long_run import evaluate_atomic
from training.env.tetris_env import RAW_REWARD_VERSION
from training.train_ppo import atomic_json, now
from training.train_transaction import fsync_dir, fsync_tree, task_name
from training.train_vector_transaction import sha256, verify_task


def verify_continuation(run_dir: Path, config: dict, state: dict, metadata: dict):
    """Reject any extension that changes training settings or its source checkpoint."""
    record = json.loads((run_dir / "continuation.json").read_text())
    expected = copy.deepcopy(record["source_config"])
    expected.update(target_total_steps=record["actual_target_steps"],
                    requested_target_steps=record["requested_target_steps"],
                    actual_target_committed_steps=record["actual_target_steps"],
                    pinned_tasks=[record["source_task"]],
                    longlife_validation={"seeds": 16, "max_pieces": 20000})
    if (config != expected or metadata.get("continuation") != record
            or state["target_steps"] != record["actual_target_steps"]
            or state["committed_steps"] < record["source_committed_steps"]):
        raise RuntimeError("Continuation settings or target differ from approved source")
    checkpoint = run_dir / "committed" / task_name(record["source_task"])
    verify_task(checkpoint, record["source_task"], config)
    if (sha256(checkpoint / "manifest.json") != record["source_sha256"]["checkpoint_manifest"]
            or sha256(checkpoint / "model.zip") != record["source_sha256"]["checkpoint_model"]):
        raise RuntimeError("Pinned source checkpoint changed")


def prepare(source: Path, destination: Path, requested_target: int = 10_000_000,
            evaluate_baseline: bool = True):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Continuation destination must be new and separate")
    if requested_target <= 0:
        raise ValueError("Target must be positive")
    with (source / ".train.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((source / "config.json").read_text())
        state = json.loads((source / "transaction_state.json").read_text())
        metadata = json.loads((source / "metadata.json").read_text())
        if (state["status"] != "completed" or state["working_task"] is not None
                or state["committed_steps"] != state["target_steps"]
                or state["committed_steps"] != config["target_total_steps"]
                or state["committed_task"] * config["task_steps"] != state["committed_steps"]):
            raise RuntimeError("Source is not a fully committed, completed run")
        if (config.get("policy") != "CandidateScoringPolicy"
                or config.get("reward_version") != RAW_REWARD_VERSION
                or config["device"] != "cuda" or config["ppo"]["n_envs"] != 8
                or config["run_seed"] != 42 or metadata["device"] != "cuda"):
            raise RuntimeError("Source differs from the approved Candidate CUDA configuration")
        target = ((requested_target + config["task_steps"] - 1) // config["task_steps"]) * config["task_steps"]
        if target <= state["committed_steps"]:
            raise ValueError("New target must exceed the completed target")
        number = state["committed_task"]
        checkpoint = source / "committed" / task_name(number)
        verify_task(checkpoint, number, config)
        trainer = pickle.loads((checkpoint / "trainer_state.pkl").read_bytes())
        workers = pickle.loads((checkpoint / "vector_env_state.pkl").read_bytes())
        if (trainer["num_timesteps"] != state["committed_steps"]
                or trainer.get("torch_cuda_random") is None or len(workers) != 8):
            raise RuntimeError("Committed trainer RNG or worker state is incomplete")
        original = {name: sha256(source / name) for name in
                    ("config.json", "metadata.json", "transaction_state.json")}
        original["checkpoint_manifest"] = sha256(checkpoint / "manifest.json")
        original["checkpoint_model"] = sha256(checkpoint / "model.zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name(destination.name + ".preparing")
        if pending.exists():
            raise FileExistsError(f"Incomplete preparation requires inspection: {pending}")
        shutil.copytree(source, pending, symlinks=True)
        copied = pending / "committed" / task_name(number)
        verify_task(copied, number, config)
        if sha256(copied / "manifest.json") != original["checkpoint_manifest"]:
            raise RuntimeError("Copied checkpoint differs from source")
        provenance = {"source_run": str(source), "source_config": copy.deepcopy(config),
                      "source_committed_steps": state["committed_steps"],
                      "source_task": number, "source_sha256": original,
                      "requested_target_steps": requested_target, "actual_target_steps": target,
                      "prepared_at": now()}
        config.update(target_total_steps=target, requested_target_steps=requested_target,
                      actual_target_committed_steps=target, pinned_tasks=[number],
                      longlife_validation={"seeds": 16, "max_pieces": 20000})
        state.update(status="awaiting_resume", target_steps=target, pid=None)
        metadata["continuation"] = provenance
        atomic_json(pending / "continuation.json", provenance)
        atomic_json(pending / "config.json", config)
        atomic_json(pending / "metadata.json", metadata)
        atomic_json(pending / "transaction_state.json", state)
        fsync_tree(pending)
        os.replace(pending, destination)
        fsync_dir(destination.parent)
    if evaluate_baseline:
        evaluate_atomic(destination, destination / "committed" / task_name(number) / "model.zip",
                        state["committed_steps"], "longlife", 16, 20000, config,
                        extra={"requested_milestone": 2_000_000,
                               "actual_committed_steps": state["committed_steps"]})
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=10_000_000)
    parser.add_argument("--evaluate-baseline", action="store_true")
    args = parser.parse_args()
    if args.evaluate_baseline:
        run = args.destination
        config = json.loads((run / "config.json").read_text())
        state = json.loads((run / "transaction_state.json").read_text())
        metadata = json.loads((run / "metadata.json").read_text())
        verify_continuation(run, config, state, metadata)
        record = metadata["continuation"]
        step = record["source_committed_steps"]
        model = run / "committed" / task_name(record["source_task"]) / "model.zip"
        evaluate_atomic(run, model, step, "longlife", 16, 20000, config,
                        extra={"requested_milestone": 2_000_000,
                               "actual_committed_steps": step})
    else:
        if args.source is None:
            parser.error("--source is required when preparing a continuation")
        print(prepare(args.source, args.destination, args.target_steps,
                      evaluate_baseline=False), flush=True)


if __name__ == "__main__":
    main()
