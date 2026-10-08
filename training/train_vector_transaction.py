"""Task-atomic MaskablePPO training with recoverable SubprocVecEnv workers."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import pickle
import subprocess
import time
from pathlib import Path

# Required before CUDA libraries initialize when deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from training.train_ppo import ROOT, atomic_bytes, atomic_json, default_config, metadata, now
from training.train_transaction import (
    FIELDS, discard_uncommitted, fsync_dir, fsync_tree, global_rng_state, initial_state,
    log_event, metrics_for_task, read_state, restore_global_rng, state_path, task_name,
    write_state,
)
from training.env import TetrisEnv
from training.env.tetris_env import (HEIGHT_REWARD_VERSION, HOLE_HEIGHT_REWARD_VERSION,
                                     HOLE_REWARD_VERSION, RAW_REWARD_VERSION)
from training.long_run import SEEDS_FILE, post_commit
from training.vector_env import make_vector_env, worker_seeds

DEFAULT_RUN = ROOT / "runs" / "ppo_transaction_vec8_cuda_test_seed42"
REQUIRED_FILES = {"model.zip", "trainer_state.pkl", "vector_env_state.pkl",
                  "metrics.json", "task_meta.json"}
HOLE_METRIC_FIELDS = ("mean_raw_reward", "mean_shaped_reward", "mean_new_holes",
                      "new_holes_per_100_pieces", "mean_hole_penalty")


def reward_settings(config):
    return (config.get("reward_version", RAW_REWARD_VERSION),
            config.get("hole_penalty_coef", 0.0), config.get("height_penalty_coef", 0.0))


def hole_metrics_for_task(callback, count):
    return {"mean_raw_reward": callback.raw_reward_sum / count,
            "mean_shaped_reward": callback.shaped_reward_sum / count,
            "mean_new_holes": callback.new_holes_sum / count,
            "new_holes_per_100_pieces": 100 * callback.new_holes_sum / count,
            "mean_hole_penalty": callback.hole_penalty_sum / count}


class TransactionTetrisEnv(TetrisEnv):
    def initial_state_digest(self):
        """Inspect an unstarted worker without instantiating its lazy RNG."""
        numpy_rng = self._np_random
        action_rng = self.action_space._np_random
        state = (self.core, self.episode_seed, self.episode_count, self.pieces,
                 self.episode_reward, self._finished, self._np_random_seed,
                 None if numpy_rng is None else numpy_rng.bit_generator.state,
                 None if action_rng is None else action_rng.bit_generator.state)
        return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()

    def _info(self, cleared_lines=0):
        info = super()._info(cleared_lines)
        info["episode_reward"] = self.episode_reward
        return info


def vector_config(task_steps=4096, target_steps=32768, device="cuda", n_envs=8,
                  base_seed=42, hole_penalty_coef=0.0, height_penalty_coef=0.0):
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if (isinstance(hole_penalty_coef, bool) or not isinstance(hole_penalty_coef, (int, float))
            or not math.isfinite(hole_penalty_coef) or hole_penalty_coef < 0):
        raise ValueError("hole_penalty_coef must be a finite nonnegative number")
    if (isinstance(height_penalty_coef, bool) or not isinstance(height_penalty_coef, (int, float))
            or not math.isfinite(height_penalty_coef) or height_penalty_coef < 0):
        raise ValueError("height_penalty_coef must be a finite nonnegative number")
    seeds = worker_seeds(base_seed, n_envs)
    if 4096 % n_envs or task_steps != 4096 or target_steps < task_steps:
        raise ValueError("Each Task must equal one 4096-sample rollout")
    actual_target = ((target_steps + task_steps - 1) // task_steps) * task_steps
    config = default_config(target=actual_target, chunk=task_steps, first_stage=actual_target,
                            eval_every=task_steps, checkpoint_every=task_steps)
    config["run_type"] = "transactional_vector_maskable_ppo"
    config["device"] = device
    config["task_steps"] = task_steps
    config["run_seed"] = base_seed
    config["worker_seed_rule"] = "base_seed + worker_index"
    config["worker_seeds"] = seeds
    config["ppo"]["n_envs"] = n_envs
    config["ppo"]["n_steps"] = task_steps // n_envs
    seed_sets = json.loads(SEEDS_FILE.read_text())
    if (len(seed_sets["validation"]) != 32 or len(set(seed_sets["validation"])) != 32
            or len(seed_sets["final_test"]) != 100 or len(set(seed_sets["final_test"])) != 100
            or set(seed_sets["validation"]) & set(seed_sets["final_test"])
            or set(seeds) & (set(seed_sets["validation"]) | set(seed_sets["final_test"]))):
        raise ValueError("Training, validation and final test seeds must be disjoint")
    config.update({"long_run_version": 1, "requested_target_steps": target_steps,
                   "actual_target_committed_steps": actual_target,
                   "transaction_retention": 3, "milestone_interval": 1000000,
                   "validation_interval": 250000,
                   "periodic_validation_seeds": 16,
                   "periodic_validation_max_pieces": 5000,
                   "milestone_validation_seeds": 32,
                   "milestone_validation_max_pieces": 10000,
                   "final_test_seeds": 100, "final_test_max_pieces": 50000,
                   "evaluation_seed_file": str(SEEDS_FILE.relative_to(ROOT)),
                   "evaluation_seed_sha256": sha256(SEEDS_FILE),
                   "validation_seed_set": seed_sets["validation"],
                   "final_test_seed_set": seed_sets["final_test"],
                   "reward_version": (HOLE_HEIGHT_REWARD_VERSION if height_penalty_coef and hole_penalty_coef
                                      else HEIGHT_REWARD_VERSION if height_penalty_coef
                                      else HOLE_REWARD_VERSION if hole_penalty_coef
                                      else RAW_REWARD_VERSION),
                   "observation_version": "237-float-public-observation-v1",
                   "action_space_version": "1840-legal-placement-v1"})
    if hole_penalty_coef:
        config["hole_penalty_coef"] = hole_penalty_coef
        config["reward"]["new_holes_penalty_coef"] = hole_penalty_coef
    if height_penalty_coef:
        config["height_penalty_coef"] = height_penalty_coef
        config["reward"]["height_risk_penalty_coef"] = height_penalty_coef
    return config


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(task_dir):
    files = {}
    for file in sorted(task_dir.rglob("*")):
        if file.is_file() and file.name != "manifest.json":
            files[str(file.relative_to(task_dir))] = {"size": file.stat().st_size,
                                                      "sha256": sha256(file)}
    if not REQUIRED_FILES.issubset(files):
        raise RuntimeError("Checkpoint is missing required Task files")
    atomic_json(task_dir / "manifest.json", {"files": files})
    return sum(record["size"] for record in files.values()) + (task_dir / "manifest.json").stat().st_size


def verify_task(task_dir, number, config):
    manifest = json.loads((task_dir / "manifest.json").read_text())["files"]
    if not REQUIRED_FILES.issubset(manifest):
        raise RuntimeError(f"Committed Task {number} has an incomplete manifest")
    for name, record in manifest.items():
        file = task_dir / name
        if not file.is_file() or file.stat().st_size != record["size"] or sha256(file) != record["sha256"]:
            raise RuntimeError(f"Committed Task {number} checksum failed: {name}")
    meta = json.loads((task_dir / "task_meta.json").read_text())
    if (meta["task_id"] != number or meta["end_step"] != number * config["task_steps"]
            or meta["device"] != config["device"] or meta["n_envs"] != config["ppo"]["n_envs"]
            or meta["n_steps"] != config["ppo"]["n_steps"]
            or len(meta["worker_digests"]) != config["ppo"]["n_envs"]
            or reward_settings(meta) != reward_settings(config)):
        raise RuntimeError(f"Committed Task {number} metadata disagrees with run config")
    return meta


def observation_and_mask_digests(env):
    import numpy as np

    observations = np.stack(env.env_method("get_observation"))
    masks = np.stack(env.env_method("action_masks"))
    if observations.shape[1:] != (237,) or masks.shape[1:] != (1840,):
        raise RuntimeError("Worker observation or mask shape changed")
    digests = [{"observation": hashlib.sha256(row.tobytes()).hexdigest(),
                "mask": hashlib.sha256(mask.tobytes()).hexdigest()}
               for row, mask in zip(observations, masks)]
    return observations, digests


def restore_task(run_dir, number, config, env):
    import numpy as np
    from sb3_contrib import MaskablePPO

    task_dir = run_dir / "committed" / task_name(number)
    task_meta = verify_task(task_dir, number, config)
    saved = pickle.loads((task_dir / "vector_env_state.pkl").read_bytes())
    trainer = pickle.loads((task_dir / "trainer_state.pkl").read_bytes())
    if len(saved) != config["ppo"]["n_envs"]:
        raise RuntimeError("Committed checkpoint does not contain every worker")
    env.reset()  # Consume the initial per-worker seeds before installing saved logical state.
    model = MaskablePPO.load(str(task_dir / "model.zip"), env=env,
                             device=config["device"], force_reset=False)
    for index, state in enumerate(saved):
        env.env_method("set_state", state, indices=[index])
    observations, digests = observation_and_mask_digests(env)
    if digests != task_meta["worker_digests"]:
        raise RuntimeError("Restored worker observation or action-mask digest differs from commit")
    if model._last_obs is None or not np.array_equal(model._last_obs, observations):
        raise RuntimeError("Restored vector observations disagree with PPO last observation")
    if (model.num_timesteps != trainer["num_timesteps"]
            or not np.array_equal(model._last_episode_starts, trainer["sb3_last_episode_starts"])):
        raise RuntimeError("Restored PPO trainer state disagrees with commit")
    if not model.policy.optimizer.state:
        raise RuntimeError("Committed PPO optimizer state is missing")
    restore_global_rng(trainer)
    return model, trainer, saved


def rebuild_vector_metrics(run_dir, committed_task, config):
    buffer = io.StringIO()
    columns = FIELDS + (HOLE_METRIC_FIELDS if config.get("hole_penalty_coef", 0.0) else ())
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for number in range(1, committed_task + 1):
        task_dir = run_dir / "committed" / task_name(number)
        archived = run_dir / "metric_history" / f"{task_name(number)}.json"
        if not archived.exists():
            verify_task(task_dir, number, config)
            atomic_json(archived, json.loads((task_dir / "metrics.json").read_text()))
        writer.writerow(json.loads(archived.read_text()))
    atomic_bytes(run_dir / "training_metrics.csv", buffer.getvalue().encode())


def ensure_final_model(run_dir, state):
    if state["status"] != "completed":
        return
    source = run_dir / "committed" / task_name(state["committed_task"]) / "model.zip"
    final = run_dir / "final"
    final.mkdir(exist_ok=True)
    destination = final / "model.zip"
    if destination.exists() and sha256(destination) == sha256(source):
        return
    temporary = final / f".model.tmp.{os.getpid()}.zip"
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)
    fsync_dir(final)


def commit_vector_task(run_dir, state, working_dir, model, env, callback, capture,
                       training_seconds, config):
    import numpy as np

    number = state["working_task"]
    start_step = state["committed_steps"]
    end_step = start_step + state["task_steps"]
    if model.num_timesteps != end_step:
        raise RuntimeError("Task ended before its complete PPO rollout and update")
    began = time.monotonic()
    saved_workers = env.env_method("get_state")
    observations, digests = observation_and_mask_digests(env)
    if len(saved_workers) != config["ppo"]["n_envs"] or not np.array_equal(model._last_obs, observations):
        raise RuntimeError("Worker states and PPO observations disagree before commit")
    trainer = global_rng_state(model)
    metrics = metrics_for_task(model, callback, capture, number, start_step, training_seconds)
    if config.get("hole_penalty_coef", 0.0):
        metrics.update(hole_metrics_for_task(callback, state["task_steps"]))
    atomic_json(working_dir / "metrics.json", metrics)
    atomic_bytes(working_dir / "trainer_state.pkl", pickle.dumps(trainer, protocol=5))
    atomic_bytes(working_dir / "vector_env_state.pkl", pickle.dumps(saved_workers, protocol=5))
    model.save(str(working_dir / "model.zip"))
    atomic_json(working_dir / "task_meta.json",
                {"task_id": number, "status": "committed", "start_step": start_step,
                 "end_step": end_step, "committed_at": now(), "device": config["device"],
                 "n_envs": config["ppo"]["n_envs"], "n_steps": config["ppo"]["n_steps"],
                 "reward_version": reward_settings(config)[0],
                 "hole_penalty_coef": reward_settings(config)[1],
                 "height_penalty_coef": reward_settings(config)[2],
                 "worker_seeds": config["worker_seeds"], "worker_digests": digests})
    checkpoint_bytes = write_manifest(working_dir)
    verify_task(working_dir, number, config)
    fsync_tree(working_dir)
    checkpoint_seconds = time.monotonic() - began
    destination = run_dir / "committed" / task_name(number)
    os.replace(working_dir, destination)
    fsync_dir(run_dir / "committed")
    fsync_dir(run_dir / "working")
    atomic_json(run_dir / "metric_history" / f"{task_name(number)}.json",
                json.loads((destination / "metrics.json").read_text()))
    state.update({"committed_task": number, "committed_steps": end_step,
                  "working_task": None, "working_steps": end_step,
                  "status": "completed" if end_step >= state["target_steps"] else "running"})
    write_state(run_dir, state)  # Advance pointer only after every Task artifact is durable.
    rebuild_vector_metrics(run_dir, number, config)
    commit_seconds = time.monotonic() - began
    log_event(run_dir, "COMMITTED", task=number, steps=end_step,
              training_seconds=round(training_seconds, 3),
              training_steps_per_second=round(state["task_steps"] / training_seconds, 2),
              checkpoint_seconds=round(checkpoint_seconds, 3),
              commit_seconds=round(commit_seconds, 3), checkpoint_bytes=checkpoint_bytes)
    if state["status"] == "completed":
        ensure_final_model(run_dir, state)
    return destination


def inspect_status(run_dir):
    from training.train_transaction import inspect_status as base_status

    base_status(run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        print(f"Workers: {config['ppo']['n_envs']}\nn_steps: {config['ppo']['n_steps']}")
        committed = list((run_dir / "committed").glob("task_*"))
        milestones = ([path for path in (run_dir / "milestones").glob("step_*")
                       if path.is_dir() and not path.name.endswith(".tmp")]
                      if (run_dir / "milestones").exists() else [])
        seen_inodes = set()
        disk_bytes = 0
        for path in run_dir.rglob("*"):
            if path.is_file():
                stat = path.stat()
                inode = (stat.st_dev, stat.st_ino)
                if inode not in seen_inodes:
                    disk_bytes += stat.st_blocks * 512
                    seen_inodes.add(inode)
        print(f"Transaction checkpoints: {len(committed)}\nMilestones: {len(milestones)}"
              f"\nRun disk usage: {disk_bytes / (1024 ** 2):.1f} MiB")
        if config.get("step0_baseline"):
            baseline = run_dir / "baseline"
            print("Step-0 baseline: " + ("validated" if (baseline / "evaluation.json").exists()
                                         else "saved; validation pending" if baseline.exists()
                                         else "pending"))
        print(f"Latest milestone: {max(milestones).name if milestones else '-'}")
        validation = run_dir / "evaluations" / "validation"
        results = list(validation.glob("step_*.json")) if validation.exists() else []
        latest = (max(results, key=lambda path: (int(path.name.split("_")[1]),
                                                 path.stat().st_mtime_ns)) if results else None)
        print(f"Latest validation: {latest.name if latest else '-'}")
        best = run_dir / "best" / "evaluation.json"
        if best.exists():
            result = json.loads(best.read_text())
            print(f"Best validation: step={result['committed_steps']} "
                  f"mean_pieces={result['aggregate']['mean_pieces']:.3f} "
                  f"mean_lines={result['aggregate']['mean_lines']:.3f} "
                  f"mean_score={result['aggregate']['mean_score']:.3f}")
        else:
            print("Best validation: -")
        state = read_state(run_dir)
        metrics_path = run_dir / "metric_history" / f"{task_name(state['committed_task'])}.json"
        if state["committed_task"] and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            print(f"Latest training steps/s: {config['task_steps'] / metrics['wall_seconds']:.1f}")
    session = subprocess.run(["tmux", "has-session", "-t", "tetris-transaction"],
                             capture_output=True, check=False).returncode == 0
    print(f"tmux session: {'tetris-transaction' if session else 'none'}")


def run(run_dir, resume=False, target_steps=32768, max_tasks=None,
        crash_during_task=None, config_file=None):
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter, Logger, TensorBoardOutputFormat

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (run_dir / ".train.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError("A trainer is already using this run") from error
    env = None
    try:
        config_path = run_dir / "config.json"
        if resume:
            if not config_path.exists() or not state_path(run_dir).exists():
                raise FileNotFoundError("Resume requires an existing vector transaction run")
            config = json.loads(config_path.read_text())
            state = read_state(run_dir)
            meta = json.loads((run_dir / "metadata.json").read_text())
            if (config["run_type"] != "transactional_vector_maskable_ppo"
                    or meta["device"] != config["device"]
                    or meta["worker_seeds"] != config["worker_seeds"]
                    or reward_settings(meta) != reward_settings(config)):
                raise RuntimeError("Run config and metadata disagree")
            if config_file:
                preset = json.loads(Path(config_file).read_text())
                requested_hole = preset.get("hole_penalty_coef", 0.0)
                requested_height = preset.get("height_penalty_coef", 0.0)
                requested_coef = (requested_hole, requested_height)
                requested_version = (HOLE_HEIGHT_REWARD_VERSION if requested_height and requested_hole
                                     else HEIGHT_REWARD_VERSION if requested_height
                                     else HOLE_REWARD_VERSION if requested_hole else RAW_REWARD_VERSION)
                if (requested_version, requested_hole, requested_height) != reward_settings(config):
                    raise RuntimeError("Resume reward config differs from recorded run")
            if config.get("long_run_version"):
                if sha256(SEEDS_FILE) != config["evaluation_seed_sha256"]:
                    raise RuntimeError("Fixed evaluation seed file changed")
                post_commit(run_dir, state, config, meta, verify_task)
            if state["status"] == "completed":
                rebuild_vector_metrics(run_dir, state["committed_task"], config)
                ensure_final_model(run_dir, state)
                return
        else:
            if config_path.exists() or state_path(run_dir).exists():
                raise FileExistsError("Run already exists; use resume")
            if config_file:
                preset = json.loads(Path(config_file).read_text())
                if (preset.get("task_steps") != 4096 or preset.get("n_steps") != 4096 // preset["n_envs"]
                        or preset.get("final_test_seeds", 100) != 100
                        or preset.get("final_test_max_pieces", 50000) != 50000):
                    raise ValueError("Preset disagrees with fixed rollout or final-test protocol")
                config = vector_config(target_steps=preset["target_total_steps"],
                                       device=preset["device"], n_envs=preset["n_envs"],
                                       base_seed=preset["base_seed"],
                                       hole_penalty_coef=preset.get("hole_penalty_coef", 0.0),
                                       height_penalty_coef=preset.get("height_penalty_coef", 0.0))
                for key in ("transaction_retention", "milestone_interval", "validation_interval",
                            "periodic_validation_seeds", "periodic_validation_max_pieces",
                            "milestone_validation_seeds", "milestone_validation_max_pieces"):
                    if key in preset:
                        config[key] = preset[key]
            else:
                config = vector_config(target_steps=target_steps)
            config["step0_baseline"] = bool(preset.get("step0_baseline", False)) if config_file else False
            if (config["transaction_retention"] < 3
                    or config["milestone_interval"] < config["task_steps"]
                    or config["validation_interval"] < config["task_steps"]
                    or not 1 <= config["periodic_validation_seeds"] <= 32
                    or not 1 <= config["milestone_validation_seeds"] <= 32
                    or config["periodic_validation_max_pieces"] < 1
                    or config["milestone_validation_max_pieces"] < 1):
                raise ValueError("Invalid long-run schedule")
            if config["device"] == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; refusing to create run")
            for name in ("working", "committed", "abandoned", "logs", "reports",
                         "metric_history", "milestones", "best_versions", "evaluations"):
                (run_dir / name).mkdir(exist_ok=True)
            atomic_json(config_path, config)
            meta = metadata(torch, gymnasium, stable_baselines3, sb3_contrib)
            try:
                driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                         "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, check=False)
                driver_version = (driver.stdout.strip().splitlines()[0]
                                  if driver.returncode == 0 and driver.stdout.strip() else None)
            except FileNotFoundError:
                driver_version = None
            meta.update({"device": config["device"], "run_type": config["run_type"],
                         "run_name": run_dir.name,
                         "nvidia_driver": driver_version,
                         "n_envs": config["ppo"]["n_envs"], "n_steps": config["ppo"]["n_steps"],
                         "task_steps": config["task_steps"], "worker_seeds": config["worker_seeds"],
                         "worker_seed_rule": config["worker_seed_rule"],
                         "requested_target_steps": config["requested_target_steps"],
                         "actual_target_committed_steps": config["actual_target_committed_steps"],
                         "transaction_retention": config["transaction_retention"],
                         "milestone_interval": config["milestone_interval"],
                         "validation_interval": config["validation_interval"],
                         "evaluation_seed_sha256": config["evaluation_seed_sha256"],
                         "validation_seed_set": config["validation_seed_set"],
                         "final_test_seed_set": config["final_test_seed_set"],
                         "reward_definition": config["reward"],
                         "reward_version": config["reward_version"],
                         "hole_penalty_coef": reward_settings(config)[1],
                         "height_penalty_coef": reward_settings(config)[2],
                         "observation_version": config["observation_version"],
                         "action_space_version": config["action_space_version"],
                         "resume_episode_policy": "Restore each worker exactly from committed Task."})
            atomic_json(run_dir / "metadata.json", meta)
            state = initial_state(config)
            write_state(run_dir, state)
            log_event(run_dir, "STARTED", target=config["target_total_steps"],
                      workers=config["ppo"]["n_envs"])
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Recorded CUDA device unavailable; refusing fallback")
        # Verify the official pointer before moving any orphaned Task to abandoned.
        first_retained = max(1, state["committed_task"] - config.get("transaction_retention", state["committed_task"]) + 1)
        for number in range(first_retained, state["committed_task"] + 1):
            verify_task(run_dir / "committed" / task_name(number), number, config)
        discard_uncommitted(run_dir, state["committed_task"])
        rebuild_vector_metrics(run_dir, state["committed_task"], config)
        env_kwargs = ({"hole_penalty_coef": reward_settings(config)[1],
                       "height_penalty_coef": reward_settings(config)[2]}
                      if any(reward_settings(config)[1:]) else None)
        env = make_vector_env(config["ppo"]["n_envs"], config["run_seed"],
                              config["max_pieces"], env_class=TransactionTetrisEnv,
                              env_kwargs=env_kwargs)
        atomic_json(run_dir / "logs" / "worker_pids.json",
                    {"parent": os.getpid(), "workers": [process.pid for process in env.processes]})
        if state["committed_task"]:
            if config.get("step0_baseline") and not (run_dir / "baseline" / "evaluation.json").exists():
                raise RuntimeError("Committed formal run is missing its Step-0 baseline")
            model, _, _ = restore_task(run_dir, state["committed_task"], config, env)
            log_event(run_dir, "RESUMED", committed_task=state["committed_task"],
                      committed_steps=state["committed_steps"])
        else:
            if resume and config.get("step0_baseline") and (run_dir / "baseline").exists():
                from training.step0_baseline import load_baseline
                model = load_baseline(run_dir, env, config)
            else:
                ppo = config["ppo"]
                model = MaskablePPO(
                    "MlpPolicy", env,
                    policy_kwargs={"net_arch": {"pi": config["network"]["pi"],
                                                "vf": config["network"]["vf"]},
                                   "activation_fn": torch.nn.Tanh},
                    learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
                    gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
                    n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
                    n_epochs=ppo["n_epochs"], ent_coef=ppo["ent_coef"],
                    vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"],
                    seed=config["run_seed"], device=config["device"], verbose=0)
            if config.get("step0_baseline"):
                from training.step0_baseline import evaluate_baseline
                state.update({"status": "baseline", "pid": os.getpid()})
                write_state(run_dir, state)
                evaluate_baseline(run_dir, model, env, config, meta)
                if max_tasks == 0:
                    state.update({"status": "awaiting_resume", "pid": None})
                    write_state(run_dir, state)
                    log_event(run_dir, "TEST_STOP_AFTER_BASELINE")
                    return

        class Capture(KVWriter):
            def __init__(self):
                self.values = {}

            def write(self, key_values, key_excluded, step=0):
                self.values.update(key_values)

            def close(self):
                pass

        class TaskCallback(BaseCallback):
            def __init__(self, working_dir, number, start_step):
                super().__init__()
                self.working_dir = working_dir
                self.number = number
                self.start_step = start_step
                self.episodes = []
                self.game_overs = 0
                self.truncations = 0
                self.cleared_lines = 0
                self.raw_reward_sum = 0.0
                self.shaped_reward_sum = 0.0
                self.new_holes_sum = 0
                self.hole_penalty_sum = 0.0

            def _on_step(self):
                for done, info in zip(self.locals["dones"], self.locals["infos"]):
                    self.cleared_lines += info.get("cleared_lines", 0)
                    if config.get("hole_penalty_coef", 0.0):
                        self.raw_reward_sum += info["raw_reward"]
                        self.shaped_reward_sum += info["shaped_reward"]
                        self.new_holes_sum += info["new_holes"]
                        self.hole_penalty_sum += info["hole_penalty"]
                    if done:
                        self.episodes.append({"reward": info["episode_reward"],
                                              "length": info["pieces"], "score": info["score"],
                                              "lines": info["lines"], "pieces": info["pieces"]})
                        self.game_overs += int(info["game_over"])
                        self.truncations += int(not info["game_over"])
                if (crash_during_task == self.number
                        and self.model.num_timesteps >= self.start_step + config["ppo"]["n_envs"] * 32):
                    atomic_json(self.working_dir / "progress.json",
                                {"working_steps": self.model.num_timesteps, "updated_at": now()})
                    raise RuntimeError("Injected interruption during working Task")
                return True

            def _on_rollout_end(self):
                atomic_json(self.working_dir / "progress.json",
                            {"working_steps": self.model.num_timesteps, "updated_at": now()})

        state.update({"status": "running", "pid": os.getpid()})
        write_state(run_dir, state)
        completed_this_process = 0
        try:
            while state["committed_steps"] < state["target_steps"]:
                number = state["committed_task"] + 1
                working_dir = run_dir / "working" / f"{task_name(number)}.tmp"
                working_dir.mkdir()
                atomic_json(working_dir / "task_meta.json",
                            {"task_id": number, "status": "in_progress",
                             "start_step": state["committed_steps"], "started_at": now(),
                             "device": config["device"], "n_envs": config["ppo"]["n_envs"],
                             "reward_version": reward_settings(config)[0],
                             "hole_penalty_coef": reward_settings(config)[1],
                             "height_penalty_coef": reward_settings(config)[2]})
                state.update({"working_task": number, "working_steps": state["committed_steps"]})
                write_state(run_dir, state)
                log_event(run_dir, "TASK_STARTED", task=number,
                          committed_steps=state["committed_steps"])
                capture = Capture()
                tensorboard = TensorBoardOutputFormat(str(working_dir / "tensorboard"))
                model.set_logger(Logger(str(working_dir), [tensorboard, capture]))
                callback = TaskCallback(working_dir, number, state["committed_steps"])
                began = time.monotonic()
                try:
                    model.learn(total_timesteps=state["task_steps"], callback=callback,
                                reset_num_timesteps=False, use_masking=True)
                    training_seconds = time.monotonic() - began
                    model.logger.dump(model.num_timesteps)
                finally:
                    model.logger.close()
                commit_vector_task(run_dir, state, working_dir, model, env, callback,
                                   capture, training_seconds, config)
                if config.get("long_run_version"):
                    post_commit(run_dir, state, config, meta, verify_task)
                completed_this_process += 1
                if max_tasks is not None and completed_this_process >= max_tasks:
                    log_event(run_dir, "TEST_STOP_AFTER_COMMIT", task=number)
                    break
            if state["committed_steps"] >= state["target_steps"]:
                state.update({"status": "completed", "pid": None})
            else:
                state["pid"] = None
            write_state(run_dir, state)
        except Exception as error:
            state.update({"status": "failed", "pid": None})
            write_state(run_dir, state)
            log_event(run_dir, "FAILED", error=repr(error))
            raise
    finally:
        if env is not None:
            env.close()
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--target-steps", type=int, default=32768)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--max-tasks", type=int, default=None, help="Test-only stop after commits")
    parser.add_argument("--crash-during-task", type=int, default=None,
                        help="Test-only injected interruption after 32 vector steps")
    args = parser.parse_args()
    if args.status:
        inspect_status(args.run_dir)
    else:
        run(args.run_dir, resume=args.resume, target_steps=args.target_steps,
            max_tasks=args.max_tasks, crash_during_task=args.crash_during_task,
            config_file=args.config_file)


if __name__ == "__main__":
    main()
