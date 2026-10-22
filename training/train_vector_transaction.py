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
from training.env.candidate_env import CandidateTetrisEnv, OBSERVATION_VERSION
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.candidate_features import FEATURE_VERSION
from training.candidate_policy import CandidateScoringPolicy
from training.env.tetris_env import (HEIGHT_REWARD_VERSION, HOLE_HEIGHT_REWARD_VERSION,
                                     HOLE_REWARD_VERSION, RAW_REWARD_VERSION)
from training.long_run import SEEDS_FILE, post_commit
from training.vector_env import make_vector_env, worker_seeds

DEFAULT_RUN = ROOT / "runs" / "ppo_transaction_vec8_cuda_test_seed42"
TRAINING_SOURCE_FILES = (
    "training/train_vector_transaction.py",
    "training/train_transaction.py",
    "training/train_ppo.py",
    "training/vector_env.py",
    "training/long_run.py",
    "training/evaluation/hybrid_queue.py",
    "training/candidate_policy.py",
    "training/candidate_features.py",
    "training/env/__init__.py",
    "training/env/tetris_env.py",
    "training/env/candidate_env.py",
    "training/env/fast_candidate_env.py",
    "training/env/failure_curriculum_env.py",
    "training/tetris_core/__init__.py",
    "training/tetris_core/actions.py",
    "training/tetris_core/core.py",
    "training/tetris_core/legal_placements.py",
    "training/tetris_core/pieces.py",
    "training/tetris_core/rng.py",
)
REQUIRED_FILES = {"model.zip", "trainer_state.pkl", "vector_env_state.pkl",
                  "metrics.json", "task_meta.json"}
HOLE_METRIC_FIELDS = ("mean_raw_reward", "mean_shaped_reward", "mean_new_holes",
                      "new_holes_per_100_pieces", "mean_hole_penalty")
KL_METRIC_FIELDS = ("bc_kl", "explained_variance")
OVERNIGHT_METRIC_FIELDS = ("reference_kl", "gradient_norm", "mean_raw_reward",
                           "mean_shaped_reward", "mean_risk_increase",
                           "mean_height_penalty", "gpu_peak_allocated_mib")
CURRICULUM_METRIC_FIELDS = ("snapshot_steps", "normal_steps", "true_deaths",
                            "window_truncations", "mean_predeath_length",
                            "origin_counts_json", "origin_episode_counts_json",
                            "source_seed_counts_json", "death_sources_json")


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


class TransactionCandidateEnv(CandidateTetrisEnv, TransactionTetrisEnv):
    """Candidate observation with the same transaction metadata hooks."""


class TransactionFastCandidateEnv(FastCandidateTetrisEnv, TransactionTetrisEnv):
    """Opt-in reuse of already validated legal placements during training steps."""


def vector_config(task_steps=4096, target_steps=32768, device="cuda", n_envs=8,
                  base_seed=42, hole_penalty_coef=0.0, height_penalty_coef=0.0,
                  candidate_policy=False, max_pieces=10000):
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if type(max_pieces) is not int or max_pieces < 1:
        raise ValueError("max_pieces must be a positive integer")
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
    config["max_pieces"] = max_pieces
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
    if candidate_policy:
        config.update({"policy": "CandidateScoringPolicy", "candidate_policy": True,
                       "observation_version": OBSERVATION_VERSION,
                       "candidate_feature_version": FEATURE_VERSION})
    return config


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_source_identity(root=ROOT):
    """Record the code that advances and observes a training environment."""
    return {name: sha256(root / name) for name in TRAINING_SOURCE_FILES}


def verify_training_source_identity(meta, root=ROOT):
    recorded = meta.get("training_source_sha256")
    if recorded is None:
        # Runs created before source manifests existed retain their old resume protocol.
        return
    current = training_source_identity(root)
    if recorded != current:
        changed = sorted(name for name in recorded.keys() | current.keys()
                         if recorded.get(name) != current.get(name))
        raise RuntimeError(f"Training source changed since run creation: {changed}")


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
            or reward_settings(meta) != reward_settings(config)
            or meta.get("observation_version", "237-float-public-observation-v1") != config["observation_version"]
            or meta.get("policy", "MlpPolicy") != config["policy"]
            or meta.get("candidate_feature_version") != config.get("candidate_feature_version")):
        raise RuntimeError(f"Committed Task {number} metadata disagrees with run config")
    return meta


def observation_and_mask_digests(env):
    import numpy as np

    worker_observations = env.env_method("get_observation")
    if isinstance(worker_observations[0], dict):
        observations = {key: np.stack([row[key] for row in worker_observations])
                        for key in worker_observations[0]}
        if (observations["state"].shape[1:] != (237,)
                or observations["candidate"].shape[1:] != (1840, 16)):
            raise RuntimeError("Candidate observation shape changed")
        observation_bytes = [b"".join(observations[key][i].tobytes()
                                      for key in sorted(observations))
                             for i in range(len(worker_observations))]
    else:
        observations = np.stack(worker_observations)
        if observations.shape[1:] != (237,):
            raise RuntimeError("Worker observation shape changed")
        observation_bytes = [row.tobytes() for row in observations]
    masks = np.stack(env.env_method("action_masks"))
    if masks.shape[1:] != (1840,):
        raise RuntimeError("Worker mask shape changed")
    digests = [{"observation": hashlib.sha256(row).hexdigest(),
                "mask": hashlib.sha256(mask.tobytes()).hexdigest()}
               for row, mask in zip(observation_bytes, masks)]
    return observations, digests


def observations_equal(left, right):
    import numpy as np
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(np.array_equal(left[key], right[key]) for key in left))
    return np.array_equal(left, right)


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
    algorithm = MaskablePPO
    if config.get("kl_reference"):
        from training.kl_maskable_ppo import KLMaskablePPO
        algorithm = KLMaskablePPO
    model = algorithm.load(str(task_dir / "model.zip"), env=env,
                             device=config["device"], force_reset=False)
    if config.get("kl_reference"):
        reference = config["kl_reference"]
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise RuntimeError("Frozen KL reference changed")
        model.set_reference(MaskablePPO.load(reference["path"], device=config["device"]))
    for index, state in enumerate(saved):
        env.env_method("set_state", state, indices=[index])
    observations, digests = observation_and_mask_digests(env)
    if digests != task_meta["worker_digests"]:
        raise RuntimeError("Restored worker observation or action-mask digest differs from commit")
    if model._last_obs is None or not observations_equal(model._last_obs, observations):
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
    columns = tuple(dict.fromkeys(
        FIELDS + (HOLE_METRIC_FIELDS if config.get("hole_penalty_coef", 0.0) else ())
        + (KL_METRIC_FIELDS if config.get("kl_reference") else ())
        + (OVERNIGHT_METRIC_FIELDS if config.get("overnight_monitor") else ())
        + (CURRICULUM_METRIC_FIELDS if config.get("failure_curriculum") else ())))
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
                       training_seconds, config, gpu_peak=None):
    import numpy as np

    number = state["working_task"]
    start_step = state["committed_steps"]
    end_step = start_step + state["task_steps"]
    if model.num_timesteps != end_step:
        raise RuntimeError("Task ended before its complete PPO rollout and update")
    began = time.monotonic()
    saved_workers = env.env_method("get_state")
    observations, digests = observation_and_mask_digests(env)
    if len(saved_workers) != config["ppo"]["n_envs"] or not observations_equal(model._last_obs, observations):
        raise RuntimeError("Worker states and PPO observations disagree before commit")
    trainer = global_rng_state(model)
    metrics = metrics_for_task(model, callback, capture, number, start_step, training_seconds)
    if config.get("kl_reference"):
        metrics["bc_kl"] = (float(capture.values["train/bc_kl"])
                            if "train/bc_kl" in capture.values else None)
        metrics["explained_variance"] = (float(capture.values["train/explained_variance"])
                                         if "train/explained_variance" in capture.values else None)
    if config.get("hole_penalty_coef", 0.0):
        metrics.update(hole_metrics_for_task(callback, state["task_steps"]))
    if config.get("overnight_monitor"):
        count = state["task_steps"]
        metrics.update(reference_kl=metrics.get("bc_kl"),
                       gradient_norm=float(capture.values["train/gradient_norm"])
                       if "train/gradient_norm" in capture.values else None,
                       mean_raw_reward=callback.raw_reward_sum / count,
                       mean_shaped_reward=callback.shaped_reward_sum / count,
                       mean_risk_increase=callback.risk_increase_sum / count,
                       mean_height_penalty=callback.height_penalty_sum / count,
                       gpu_peak_allocated_mib=gpu_peak["allocated"] if gpu_peak else None)
    if config.get("failure_curriculum"):
        metrics.update(snapshot_steps=callback.snapshot_steps,
                       normal_steps=callback.normal_steps,
                       true_deaths=callback.game_overs,
                       window_truncations=callback.window_truncations,
                       mean_predeath_length=(sum(callback.death_lengths)/len(callback.death_lengths)
                                             if callback.death_lengths else None),
                       origin_counts_json=json.dumps(callback.origin_counts,sort_keys=True),
                       origin_episode_counts_json=json.dumps(callback.origin_episode_counts,sort_keys=True),
                       source_seed_counts_json=json.dumps(callback.source_seed_counts,sort_keys=True),
                       death_sources_json=json.dumps(callback.death_sources,sort_keys=True))
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
                 "observation_version": config["observation_version"], "policy": config["policy"],
                 "candidate_feature_version": config.get("candidate_feature_version"),
                 "gpu_peak_mib": gpu_peak,
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
            verify_training_source_identity(meta)
            if (run_dir / "continuation.json").exists():
                from training.extend_candidate import verify_continuation
                verify_continuation(run_dir, config, state, meta)
            if (config["run_type"] != "transactional_vector_maskable_ppo"
                    or meta["device"] != config["device"]
                    or meta["worker_seeds"] != config["worker_seeds"]
                    or reward_settings(meta) != reward_settings(config)
                    or meta.get("observation_version", "237-float-public-observation-v1") != config["observation_version"]
                    or meta.get("policy", "MlpPolicy") != config["policy"]
                    or meta.get("candidate_feature_version") != config.get("candidate_feature_version")
                    or meta.get("failure_curriculum") != config.get("failure_curriculum")):
                raise RuntimeError("Run config and metadata disagree")
            if config_file:
                preset = json.loads(Path(config_file).read_text())
                requested_hole = preset.get("hole_penalty_coef", 0.0)
                requested_height = preset.get("height_penalty_coef", 0.0)
                requested_coef = (requested_hole, requested_height)
                requested_version = (HOLE_HEIGHT_REWARD_VERSION if requested_height and requested_hole
                                     else HEIGHT_REWARD_VERSION if requested_height
                                     else HOLE_REWARD_VERSION if requested_hole else RAW_REWARD_VERSION)
                if preset.get("failure_curriculum"):
                    from training.env.failure_curriculum_env import REWARD_VERSION
                    requested_version = REWARD_VERSION
                if (requested_version, requested_hole, requested_height) != reward_settings(config):
                    raise RuntimeError("Resume reward config differs from recorded run")
                if bool(preset.get("candidate_policy", False)) != bool(config.get("candidate_policy", False)):
                    raise RuntimeError("Resume candidate policy differs from recorded run")
                if preset.get("initial_model") != config.get("initial_model"):
                    raise RuntimeError("Resume initial model differs from recorded run")
                if preset.get("kl_reference") != config.get("kl_reference"):
                    raise RuntimeError("Resume KL reference differs from recorded run")
                if preset.get("max_pieces", 10000) != config["max_pieces"]:
                    raise RuntimeError("Resume episode cap differs from recorded run")
                if bool(preset.get("overnight_monitor", False)) != bool(config.get("overnight_monitor", False)):
                    raise RuntimeError("Resume monitoring protocol differs from recorded run")
                if preset.get("failure_curriculum") != config.get("failure_curriculum"):
                    raise RuntimeError("Resume failure curriculum differs from recorded run")
                if preset.get("ppo_overrides", {}) != config.get("ppo_overrides", {}):
                    raise RuntimeError("Resume PPO overrides differ from recorded run")
                if bool(preset.get("fast_candidate_env", False)) != bool(config.get("fast_candidate_env", False)):
                    raise RuntimeError("Resume Candidate environment differs from recorded run")
            if config.get("initial_model"):
                initial = config["initial_model"]
                if sha256(Path(initial["path"])) != initial["sha256"]:
                    raise RuntimeError("Initial model changed")
            if config.get("kl_reference"):
                reference = config["kl_reference"]
                if sha256(Path(reference["path"])) != reference["sha256"]:
                    raise RuntimeError("Frozen BC reference changed")
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
                                       height_penalty_coef=preset.get("height_penalty_coef", 0.0),
                                       candidate_policy=bool(preset.get("candidate_policy", False)),
                                       max_pieces=preset.get("max_pieces", 10000))
                overrides = preset.get("ppo_overrides", {})
                if set(overrides) - {"gamma", "gae_lambda"}:
                    raise ValueError("Unsupported PPO override")
                if "gamma" in overrides and not 0.99 < overrides["gamma"] < 1:
                    raise ValueError("Invalid PPO gamma override")
                if "gae_lambda" in overrides and not 0.9 <= overrides["gae_lambda"] <= 1:
                    raise ValueError("Invalid PPO GAE override")
                config["ppo_overrides"] = overrides
                config["ppo"].update(overrides)
                if preset.get("fast_candidate_env", False) and not preset.get("candidate_policy", False):
                    raise ValueError("Fast Candidate environment requires Candidate policy")
                config["fast_candidate_env"] = bool(preset.get("fast_candidate_env", False))
                config["overnight_monitor"] = bool(preset.get("overnight_monitor", False))
                if "failure_curriculum" in preset:
                    from training.env.failure_curriculum_env import REWARD_VERSION, file_sha256
                    course = preset["failure_curriculum"]
                    if (not config.get("candidate_policy") or course.get("mode") not in ("normal","curriculum")
                            or course.get("window_pieces") != 5000
                            or course.get("death_penalty") != -20.0
                            or course.get("snapshot_probability") != 0.7
                            or not 0 < len(course.get("tiers",[])) <= 5):
                        raise ValueError("Invalid frozen failure curriculum")
                    if course["mode"] == "curriculum" and file_sha256(course["dataset_path"]) != course["dataset_sha256"]:
                        raise RuntimeError("Curriculum dataset integrity failed")
                    config["failure_curriculum"] = course
                    config["reward_version"] = REWARD_VERSION
                    config["reward"]["game_over"] = -20.0
                for key in ("transaction_retention", "milestone_interval", "validation_interval",
                            "periodic_validation_seeds", "periodic_validation_max_pieces",
                            "milestone_validation_seeds", "milestone_validation_max_pieces"):
                    if key in preset:
                        config[key] = preset[key]
                if "initial_model" in preset:
                    initial = preset["initial_model"]
                    if not config.get("candidate_policy") or sha256(Path(initial["path"])) != initial["sha256"]:
                        raise RuntimeError("Candidate initial model integrity failed")
                    config["initial_model"] = initial
                if "kl_reference" in preset:
                    reference = preset["kl_reference"]
                    if (not config.get("candidate_policy") or
                            sha256(Path(reference["path"])) != reference["sha256"] or
                            float(reference["beta"]) < 0):
                        raise RuntimeError("KL reference integrity failed")
                    if not config.get("initial_model") or config["initial_model"]["sha256"] != reference["sha256"]:
                        raise RuntimeError("KL run must start from its frozen reference")
                    config["kl_reference"] = reference
                    config["algorithm"] = "training.kl_maskable_ppo.KLMaskablePPO"
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
            meta["training_source_sha256"] = training_source_identity()
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
                         "policy": config["policy"],
                         "candidate_feature_version": config.get("candidate_feature_version"),
                         "failure_curriculum": config.get("failure_curriculum"),
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
                      if any(reward_settings(config)[1:]) else {})
        if config.get("failure_curriculum"):
            from training.env.failure_curriculum_env import FailureCurriculumEnv
            env_kwargs["curriculum"] = config["failure_curriculum"]
            env_class = FailureCurriculumEnv
        else:
            env_class = (TransactionFastCandidateEnv if config.get("fast_candidate_env")
                         else TransactionCandidateEnv) if config.get("candidate_policy") else TransactionTetrisEnv
        env = make_vector_env(config["ppo"]["n_envs"], config["run_seed"],
                              config["max_pieces"],
                              env_class=env_class,
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
                policy = CandidateScoringPolicy if config.get("candidate_policy") else "MlpPolicy"
                policy_kwargs = ({} if config.get("candidate_policy") else
                                 {"net_arch": {"pi": config["network"]["pi"],
                                               "vf": config["network"]["vf"]},
                                  "activation_fn": torch.nn.Tanh})
                algorithm = MaskablePPO
                extra_algorithm = {}
                if config.get("kl_reference"):
                    from training.kl_maskable_ppo import KLMaskablePPO
                    algorithm = KLMaskablePPO
                    extra_algorithm["kl_beta"] = config["kl_reference"]["beta"]
                model = algorithm(
                    policy, env, policy_kwargs=policy_kwargs,
                    learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
                    gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
                    n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
                    n_epochs=ppo["n_epochs"], ent_coef=ppo["ent_coef"],
                    vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"],
                    seed=config["run_seed"], device=config["device"], verbose=0,
                    **extra_algorithm)
                if config.get("initial_model"):
                    source = MaskablePPO.load(config["initial_model"]["path"], device="cpu")
                    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
                    # Preserve the fresh optimizer, timestep zero, and independent RNG.
                    if model.num_timesteps != 0:
                        raise RuntimeError("Fine-tune run must start at PPO step zero")
                if config.get("kl_reference"):
                    model.set_reference(source)
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
                self.risk_increase_sum = 0.0
                self.height_penalty_sum = 0.0
                self.snapshot_steps = 0
                self.normal_steps = 0
                self.window_truncations = 0
                self.death_lengths = []
                self.origin_counts = {}
                self.origin_episode_counts = {}
                self.source_seed_counts = {}
                self.death_sources = {}

            def _on_step(self):
                for done, info in zip(self.locals["dones"], self.locals["infos"]):
                    self.cleared_lines += info.get("cleared_lines", 0)
                    if config.get("failure_curriculum"):
                        origin = info["curriculum_origin"]
                        label = origin["tier"]
                        if label == "normal":
                            self.normal_steps += 1
                        else:
                            self.snapshot_steps += 1
                        self.origin_counts[label] = self.origin_counts.get(label, 0) + 1
                        if info["curriculum_episode_start"]:
                            self.origin_episode_counts[label] = self.origin_episode_counts.get(label, 0) + 1
                        if origin["source_seed"] is not None:
                            key = str(origin["source_seed"])
                            self.source_seed_counts[key] = self.source_seed_counts.get(key, 0) + 1
                        self.window_truncations += int(info["curriculum_window_truncated"])
                        if info["true_game_over"]:
                            self.death_lengths.append(info["pieces"])
                            self.death_sources[label] = self.death_sources.get(label, 0) + 1
                    if config.get("hole_penalty_coef", 0.0) or config.get("overnight_monitor"):
                        self.raw_reward_sum += info["raw_reward"]
                        self.shaped_reward_sum += info["shaped_reward"]
                        self.new_holes_sum += info["new_holes"]
                        self.hole_penalty_sum += info["hole_penalty"]
                    if config.get("overnight_monitor"):
                        self.risk_increase_sum += info["risk_increase"]
                        self.height_penalty_sum += info["height_penalty"]
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
                if config["device"] == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                began = time.monotonic()
                try:
                    model.learn(total_timesteps=state["task_steps"], callback=callback,
                                reset_num_timesteps=False, use_masking=True)
                    training_seconds = time.monotonic() - began
                    model.logger.dump(model.num_timesteps)
                finally:
                    model.logger.close()
                gpu_peak = ({"allocated": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                             "reserved": round(torch.cuda.max_memory_reserved() / 2**20, 2)}
                            if config["device"] == "cuda" else None)
                commit_vector_task(run_dir, state, working_dir, model, env, callback,
                                   capture, training_seconds, config, gpu_peak)
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
