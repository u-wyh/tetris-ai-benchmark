"""Post-commit validation, durable milestones, best selection and safe retention."""

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from training.evaluation.evaluate import best_key
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now
from training.train_transaction import fsync_dir, fsync_tree, log_event, task_name

SEEDS_FILE = ROOT / "training" / "evaluation" / "seeds.json"
VALIDATION_FIELDS = ("committed_steps", "requested_validation_step", "model_checkpoint", "num_seeds", "max_pieces",
                     "mean_pieces", "median_pieces", "mean_lines", "mean_score",
                     "mean_reward", "survival_cap_rate", "game_over_rate", "is_best",
                     "wall_time")


def crossings(previous, current, interval):
    if interval < 1 or current < previous:
        raise ValueError("Invalid threshold interval or step range")
    return range((previous // interval + 1) * interval, current + 1, interval)


def evaluation_path(run_dir, step, protocol):
    return run_dir / "evaluations" / "validation" / f"step_{step:09d}_{protocol}.json"


def validate_evaluation(result, step, protocol, count, cap):
    expected_seeds = json.loads(SEEDS_FILE.read_text())["validation"][:count]
    required_metrics = ("mean_pieces", "median_pieces", "mean_lines", "median_lines",
                        "mean_score", "median_score", "mean_reward", "p90_pieces",
                        "best_pieces", "game_over_rate", "survival_cap_rate",
                        "evaluation_saturated", "metric_censored_by_cap")
    required_row = ("seed", "pieces_survived", "lines", "score", "episode_reward",
                    "game_over", "survived_cap")
    if (result.get("committed_steps") != step or result.get("protocol") != protocol
            or result.get("num_seeds") != count or result.get("max_pieces") != cap
            or result.get("seeds") != expected_seeds
            or [row.get("seed") for row in result.get("per_seed", [])] != expected_seeds
            or not all(all(key in row for key in required_row)
                       and not (row["game_over"] and row["survived_cap"])
                       for row in result.get("per_seed", []))
            or not all(key in result.get("aggregate", {}) for key in required_metrics)
            or "wall_seconds" not in result):
        raise RuntimeError("Incomplete or mismatched validation result")


def evaluate_atomic(run_dir, source_model, step, protocol, count, cap, config, extra=None):
    destination = evaluation_path(run_dir, step, protocol)
    if destination.exists():
        result = json.loads(destination.read_text())
        validate_evaluation(result, step, protocol, count, cap)
        if extra:
            if any(key in result and result[key] != value for key, value in extra.items()):
                raise RuntimeError("Validation metadata differs from run schedule")
            if any(key not in result for key in extra):
                result.update(extra)
                atomic_json(destination, result)
                fsync_dir(destination.parent)
        return result
    working = run_dir / "evaluations" / "working"
    working.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:09d}_{protocol}"
    pending = working / name
    if pending.exists():
        shutil.rmtree(pending)  # Incomplete evaluation is never a formal result.
    pending.mkdir()
    fsync_dir(working)
    began = time.monotonic()
    command = [sys.executable, "-m", "training.evaluation.evaluate",
               "--model", str(source_model), "--seeds-file", str(SEEDS_FILE),
               "--seed-set", "validation", "--seed-count", str(count),
               "--max-pieces", str(cap), "--protocol", protocol,
               "--committed-steps", str(step), "--output", str(pending / "result.json")]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads((pending / "result.json").read_text())
        validate_evaluation(result, step, protocol, count, cap)
        if extra:
            result.update(extra)
        result["inference_seconds"] = result["wall_seconds"]
        result["wall_seconds"] = time.monotonic() - began
        atomic_json(pending / "result.json", result)
        fsync_tree(pending)
        os.replace(pending / "result.json", destination)
        fsync_dir(destination.parent)
        shutil.rmtree(pending)
        log_event(run_dir, "VALIDATED", step=step, protocol=protocol,
                  seconds=round(result["wall_seconds"], 3))
        return result
    except Exception:
        # Preserve the committed Task; resume retries from its immutable model.
        raise


def create_milestone(run_dir, source, threshold, step, config, run_metadata, result):
    destination = run_dir / "milestones" / f"step_{step:09d}"
    if destination.exists():
        return destination
    destination.parent.mkdir(exist_ok=True)
    pending = destination.with_name(destination.name + ".tmp")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    shutil.copy2(source / "model.zip", pending / "model.zip")
    meta = {"requested_milestone": threshold, "actual_committed_steps": step,
            "task_id": step // config["task_steps"], "git_commit": run_metadata["git_commit"],
            "training_config": config, "device": config["device"],
            "n_envs": config["ppo"]["n_envs"], "n_steps": config["ppo"]["n_steps"],
            "base_seed": config["run_seed"], "model_path": "model.zip",
            "validation_result": "evaluation.json", "created_at": now()}
    atomic_json(pending / "metadata.json", meta)
    atomic_json(pending / "evaluation.json", result)
    from training.train_vector_transaction import sha256
    atomic_json(pending / "checksum.json", {"model_sha256": sha256(pending / "model.zip")})
    fsync_tree(pending)
    os.replace(pending, destination)
    fsync_dir(destination.parent)
    log_event(run_dir, "MILESTONE", requested=threshold, actual=step)
    return destination


def sync_best(run_dir):
    """Compare one fixed protocol; switch a complete best version atomically."""
    validation = run_dir / "evaluations" / "validation"
    if not validation.exists():
        return None
    # Milestone's 32-seed/10k cap is not comparable with periodic's 16-seed/5k cap.
    # Formal 1M milestones also cross a 250k periodic threshold, so each model
    # remains eligible for best under the same fixed periodic protocol.
    candidates = [(json.loads(path.read_text()), path)
                  for path in sorted(validation.glob("*_periodic.json"))]
    if not candidates:
        return None
    winner, result_path = max(candidates, key=lambda item: best_key(item[0]))
    step = winner["committed_steps"]
    current = run_dir / "best"
    if current.is_symlink():
        current_meta = json.loads((current / "metadata.json").read_text())
        if current_meta["evaluation_path"] == str(result_path.relative_to(run_dir)):
            return winner
    versions = run_dir / "best_versions"
    versions.mkdir(exist_ok=True)
    version = versions / f"step_{step:09d}_{winner['protocol']}"
    if not version.exists():
        source = run_dir / "committed" / task_name(
            step // json.loads((run_dir / "config.json").read_text())["task_steps"])
        if not source.exists():
            source = run_dir / "milestones" / f"step_{step:09d}"
        if step == 0 and not source.exists():
            source = run_dir / "baseline"
        if not source.exists():
            raise FileNotFoundError(f"Best source for step {step} was pruned")
        pending = versions / (version.name + ".tmp")
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        shutil.copy2(source / "model.zip", pending / "model.zip")
        atomic_json(pending / "evaluation.json", winner)
        atomic_json(pending / "metadata.json", {"committed_steps": step,
                    "evaluation_path": str(result_path.relative_to(run_dir)),
                    "selection_key": best_key(winner), "created_at": now()})
        from training.train_vector_transaction import sha256
        atomic_json(pending / "checksum.json", {"model_sha256": sha256(pending / "model.zip")})
        fsync_tree(pending)
        os.replace(pending, version)
        fsync_dir(versions)
    link = run_dir / f".best.tmp.{os.getpid()}"
    link.unlink(missing_ok=True)
    link.symlink_to(version.relative_to(run_dir), target_is_directory=True)
    os.replace(link, current)
    fsync_dir(run_dir)
    for old in versions.iterdir():
        if old != version and old.is_dir():
            shutil.rmtree(old)
    log_event(run_dir, "BEST_UPDATED", step=step, key=best_key(winner))
    return winner


def rebuild_validation_summary(run_dir):
    directory = run_dir / "evaluations" / "validation"
    if not directory.exists():
        return
    best_path = None
    if (run_dir / "best").is_symlink():
        best_path = json.loads((run_dir / "best" / "metadata.json").read_text())["evaluation_path"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=VALIDATION_FIELDS)
    writer.writeheader()
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text())
        agg = result["aggregate"]
        writer.writerow({"committed_steps": result["committed_steps"],
                         "requested_validation_step": result.get("requested_validation_step", ""),
                         "model_checkpoint": result["model_checkpoint"],
                         "num_seeds": result["num_seeds"], "max_pieces": result["max_pieces"],
                         "mean_pieces": agg["mean_pieces"], "median_pieces": agg["median_pieces"],
                         "mean_lines": agg["mean_lines"], "mean_score": agg["mean_score"],
                         "mean_reward": agg["mean_reward"],
                         "survival_cap_rate": agg["survival_cap_rate"],
                         "game_over_rate": agg["game_over_rate"],
                         "is_best": str(path.relative_to(run_dir)) == best_path,
                         "wall_time": result["wall_seconds"]})
    atomic_bytes(run_dir / "evaluations" / "validation_summary.csv", buffer.getvalue().encode())


def prune_transactions(run_dir, state, config, verify):
    keep = config.get("transaction_retention", 3)
    latest = state["committed_task"]
    if latest < keep:
        return 0.0
    # Never delete before the pointer and all retained checkpoints are valid.
    for number in range(latest - keep + 1, latest + 1):
        verify(run_dir / "committed" / task_name(number), number, config)
    began = time.monotonic()
    for source in sorted((run_dir / "committed").glob("task_*")):
        if (source.is_dir() and int(source.name.split("_")[1]) <= latest - keep
                and int(source.name.split("_")[1]) not in config.get("pinned_tasks", [])):
            shutil.rmtree(source)
    fsync_dir(run_dir / "committed")
    elapsed = time.monotonic() - began
    if elapsed:
        log_event(run_dir, "RETENTION", latest=latest, kept=keep,
                  cleanup_seconds=round(elapsed, 3))
    return elapsed


def post_commit(run_dir, state, config, run_metadata, verify):
    number = state["committed_task"]
    if not number:
        return
    step = state["committed_steps"]
    previous = step - config["task_steps"]
    source = run_dir / "committed" / task_name(number)
    verify(source, number, config)
    for threshold in crossings(previous, step, config.get("validation_interval", 250000)):
        evaluate_atomic(run_dir, source / "model.zip", step, "periodic",
                        config.get("periodic_validation_seeds", 16),
                        config.get("periodic_validation_max_pieces", 5000), config,
                        extra={"requested_validation_step": threshold,
                               "actual_committed_steps": step})
    for threshold in crossings(previous, step, config.get("milestone_interval", 1000000)):
        result = evaluate_atomic(run_dir, source / "model.zip", step, "milestone",
                                 config.get("milestone_validation_seeds", 32),
                                 config.get("milestone_validation_max_pieces", 10000), config,
                                 extra={"requested_milestone": threshold,
                                        "actual_committed_steps": step})
        create_milestone(run_dir, source, threshold, step, config, run_metadata, result)
        if config.get("longlife_validation"):
            evaluate_atomic(run_dir, source / "model.zip", step, "longlife",
                            config["longlife_validation"]["seeds"],
                            config["longlife_validation"]["max_pieces"], config,
                            extra={"requested_milestone": threshold,
                                   "actual_committed_steps": step})
    sync_best(run_dir)
    rebuild_validation_summary(run_dir)
    prune_transactions(run_dir, state, config, verify)
