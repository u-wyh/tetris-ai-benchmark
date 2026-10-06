"""Task-atomic MaskablePPO training. transaction_state.json is the commit pointer."""

import argparse
import csv
import fcntl
import io
import json
import os
import pickle
import random
import time
from pathlib import Path

from training.train_ppo import ROOT, atomic_bytes, atomic_json, default_config, metadata, now

DEFAULT_RUN = ROOT / "runs" / "ppo_transaction_test_seed42"
FIELDS = ("task_id", "start_step", "end_step", "wall_seconds", "mean_episode_reward",
          "mean_episode_length", "mean_score", "mean_lines", "mean_pieces",
          "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
          "learning_rate", "game_over_count", "truncated_count", "cleared_lines")


def config_for_tasks(task_steps=4096, target_steps=32768, device="cpu"):
    if task_steps < 1024 or task_steps % 1024 or target_steps < task_steps or target_steps % task_steps:
        raise ValueError("Task and target must be positive multiples of the 1024-step PPO rollout")
    config = default_config(target=target_steps, chunk=task_steps, first_stage=target_steps,
                            eval_every=task_steps, checkpoint_every=task_steps)
    config["task_steps"] = task_steps
    config["run_type"] = "transactional_maskable_ppo"
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    config["device"] = device
    return config


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(root):
    for directory, _, files in os.walk(root):
        for name in files:
            with (Path(directory) / name).open("rb") as file:
                os.fsync(file.fileno())
        fsync_dir(directory)


def log_event(run_dir, name, **values):
    line = f"{now()} {name}" + "".join(f" {key}={value}" for key, value in values.items()) + "\n"
    with (run_dir / "events.log").open("a", encoding="utf8") as file:
        file.write(line)
        file.flush()
        os.fsync(file.fileno())


def task_name(number):
    return f"task_{number:06d}"


def initial_state(config):
    return {"status": "new", "committed_task": 0, "committed_steps": 0,
            "working_task": None, "working_steps": 0,
            "target_steps": config["target_total_steps"], "task_steps": config["task_steps"],
            "pid": None, "updated_at": now()}


def state_path(run_dir):
    return run_dir / "transaction_state.json"


def read_state(run_dir):
    return json.loads(state_path(run_dir).read_text())


def write_state(run_dir, state):
    state["updated_at"] = now()
    atomic_json(state_path(run_dir), state)
    fsync_dir(run_dir)


def mark_abandoned(run_dir, source, reason):
    abandoned = run_dir / "abandoned"
    abandoned.mkdir(exist_ok=True)
    destination = abandoned / f"{source.name}.{time.time_ns()}"
    os.replace(source, destination)
    atomic_json(destination / "abandoned.json", {"status": "abandoned", "reason": reason, "at": now()})
    fsync_dir(abandoned)
    log_event(run_dir, "ABANDONED", task=source.name, reason=reason)


def discard_uncommitted(run_dir, committed_task):
    for source in sorted((run_dir / "working").iterdir()):
        if source.is_dir():
            mark_abandoned(run_dir, source, "working task was never committed")
    for source in sorted((run_dir / "committed").iterdir()):
        if source.is_dir() and source.name.startswith("task_"):
            number = int(source.name.split("_")[1])
            if number > committed_task:
                mark_abandoned(run_dir, source, "directory exists beyond commit pointer")


def rebuild_metrics(run_dir, committed_task):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    for number in range(1, committed_task + 1):
        task_dir = run_dir / "committed" / task_name(number)
        if not task_dir.exists():
            raise FileNotFoundError(f"Committed Task {number} is missing")
        writer.writerow(json.loads((task_dir / "metrics.json").read_text()))
    atomic_bytes(run_dir / "training_metrics.csv", buffer.getvalue().encode())


def save_environment(env, monitor):
    import copy

    return {"core": copy.deepcopy(env.core), "episode_seed": env.episode_seed,
            "pieces": env.pieces, "episode_reward": env.episode_reward,
            "finished": env._finished,
            "numpy_generator": copy.deepcopy(env.np_random.bit_generator.state),
            "action_generator": copy.deepcopy(env.action_space.np_random.bit_generator.state),
            "monitor": {"rewards": list(monitor.rewards), "needs_reset": monitor.needs_reset,
                        "episode_returns": list(monitor.episode_returns),
                        "episode_lengths": list(monitor.episode_lengths),
                        "episode_times": list(monitor.episode_times),
                        "total_steps": monitor.total_steps,
                        "current_reset_info": dict(monitor.current_reset_info)}}


def restore_environment(env, monitor, saved):
    import copy

    env.core = copy.deepcopy(saved["core"])
    env.episode_seed = saved["episode_seed"]
    env.pieces = saved["pieces"]
    env.episode_reward = saved["episode_reward"]
    env._finished = saved["finished"]
    env.np_random.bit_generator.state = saved["numpy_generator"]
    env.action_space.np_random.bit_generator.state = saved["action_generator"]
    for key, value in saved["monitor"].items():
        setattr(monitor, key, copy.deepcopy(value))
    monitor.t_start = time.time() - sum(monitor.episode_times)


def global_rng_state(model):
    import numpy as np
    import torch

    return {"python_random": random.getstate(), "numpy_random": np.random.get_state(),
            "torch_cpu_random": torch.get_rng_state(),
            "torch_cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "num_timesteps": model.num_timesteps, "run_seed": 42,
            "sb3_last_obs": model._last_obs,
            "sb3_last_episode_starts": model._last_episode_starts}


def restore_global_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu_random"])
    if torch.cuda.is_available() and state["torch_cuda_random"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def load_committed(model_class, run_dir, number, env, device):
    import numpy as np

    task_dir = run_dir / "committed" / task_name(number)
    task_meta = json.loads((task_dir / "task_meta.json").read_text())
    if task_meta.get("device", device) != device:
        raise RuntimeError("Committed task device differs from recorded run device")
    trainer = pickle.loads((task_dir / "trainer_state.pkl").read_bytes())
    saved_env = pickle.loads((task_dir / "env_state.pkl").read_bytes())
    model = model_class.load(str(task_dir / "model.zip"), env=env, device=device, force_reset=False)
    monitor = model.get_env().envs[0]
    restore_environment(env, monitor, saved_env)
    if model.num_timesteps != trainer["num_timesteps"]:
        raise RuntimeError("Model timesteps disagree with committed trainer state")
    observation = env._encode_observation(env.core.get_public_observation())
    if model._last_obs is None or not np.array_equal(model._last_obs[0], observation):
        raise RuntimeError("Restored environment does not match SB3's last observation")
    if not model.policy.optimizer.state:
        raise RuntimeError("Committed model has no optimizer state")
    restore_global_rng(trainer)
    return model, trainer, saved_env


def metrics_for_task(model, callback, capture, number, start_step, seconds):
    values = capture.values
    episodes = callback.episodes

    def scalar(key):
        value = values.get(key)
        return float(value) if value is not None else None

    def mean(key):
        samples = [episode[key] for episode in episodes if episode[key] is not None]
        return sum(samples) / len(samples) if samples else None

    entropy_loss = scalar("train/entropy_loss")
    return {"task_id": number, "start_step": start_step, "end_step": model.num_timesteps,
            "wall_seconds": round(seconds, 3), "mean_episode_reward": mean("reward"),
            "mean_episode_length": mean("length"), "mean_score": mean("score"),
            "mean_lines": mean("lines"), "mean_pieces": mean("pieces"),
            "policy_loss": scalar("train/policy_gradient_loss"),
            "value_loss": scalar("train/value_loss"),
            "entropy": -entropy_loss if entropy_loss is not None else None,
            "approx_kl": scalar("train/approx_kl"),
            "clip_fraction": scalar("train/clip_fraction"),
            "learning_rate": scalar("train/learning_rate"),
            "game_over_count": callback.game_overs,
            "truncated_count": callback.truncations,
            "cleared_lines": callback.cleared_lines}


def commit_task(run_dir, state, working_dir, model, env, monitor, callback, capture, seconds):
    number = state["working_task"]
    start = state["committed_steps"]
    end = start + state["task_steps"]
    if model.num_timesteps != end:
        raise RuntimeError(f"Task {number} ended at {model.num_timesteps}, expected {end}")
    metrics = metrics_for_task(model, callback, capture, number, start, seconds)
    atomic_json(working_dir / "metrics.json", metrics)
    atomic_bytes(working_dir / "trainer_state.pkl", pickle.dumps(global_rng_state(model), protocol=5))
    atomic_bytes(working_dir / "env_state.pkl", pickle.dumps(save_environment(env, monitor), protocol=5))
    model.save(str(working_dir / "model.zip"))
    atomic_json(working_dir / "task_meta.json",
                {"task_id": number, "status": "committed", "start_step": start,
                 "end_step": end, "committed_at": now(), "device": str(model.device)})
    fsync_tree(working_dir)
    destination = run_dir / "committed" / task_name(number)
    os.replace(working_dir, destination)
    fsync_dir(run_dir / "committed")
    fsync_dir(run_dir / "working")
    state.update({"committed_task": number, "committed_steps": end,
                  "working_task": None, "working_steps": end,
                  "status": "completed" if end >= state["target_steps"] else "running"})
    # The commit pointer advances only after every artifact and the renamed directory exist.
    write_state(run_dir, state)
    rebuild_metrics(run_dir, number)
    log_event(run_dir, "COMMITTED", task=number, steps=end)
    if state["status"] == "completed":
        final = run_dir / "final"
        final.mkdir(exist_ok=True)
        temporary = final / f".model.tmp.{os.getpid()}.zip"
        os.link(destination / "model.zip", temporary)
        os.replace(temporary, final / "model.zip")
        fsync_dir(final)
    return destination


def inspect_status(run_dir):
    if not state_path(run_dir).exists():
        print(f"Run: {run_dir.name}\nStatus: new")
        return
    state = read_state(run_dir)
    pid = state.get("pid")
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            pass
    current = state.get("working_task")
    progress = None
    if current:
        progress_file = run_dir / "working" / f"{task_name(current)}.tmp" / "progress.json"
        if progress_file.exists():
            progress = json.loads(progress_file.read_text())
    working_steps = progress.get("working_steps") if progress else state["working_steps"]
    checkpoint = run_dir / "committed" / task_name(state["committed_task"]) / "model.zip" if state["committed_task"] else None
    meta = json.loads((run_dir / "metadata.json").read_text())
    print(f"Run: {run_dir.name}\nProcess: {'running' if alive else 'stopped'}"
          f"\nStatus: {state['status']}\nCommitted task: {task_name(state['committed_task']) if state['committed_task'] else '-'}"
          f"\nCommitted steps: {state['committed_steps']} / {state['target_steps']}"
          f"\nCurrent task: {task_name(current) if current else '-'}"
          f"\nCurrent task status: {'in_progress' if current and alive else 'abandoned_pending_resume' if current else 'pending' if state['status'] != 'completed' else 'none'}"
          f"\nWorking steps (uncommitted): {working_steps}"
          f"\nLatest committed checkpoint: {checkpoint if checkpoint and checkpoint.exists() else '-'}"
          f"\nPID: {pid if alive else '-'}\nGPU: {meta.get('gpu') or 'none'}"
          f"\nDevice: {meta.get('device')}")


def run(run_dir, resume=False, task_steps=4096, target_steps=32768, max_tasks=None, device=None):
    import gymnasium
    import numpy as np
    import sb3_contrib
    import stable_baselines3
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter, Logger, TensorBoardOutputFormat
    from training.env import TetrisEnv

    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (run_dir / ".train.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A trainer is already using this run") from error

    config_path = run_dir / "config.json"
    if resume:
        if not config_path.exists() or not state_path(run_dir).exists():
            raise FileNotFoundError("Resume requires an existing transactional run")
        config = json.loads(config_path.read_text())
        state = read_state(run_dir)
        saved_meta = json.loads((run_dir / "metadata.json").read_text())
        recorded_device = config.get("device", saved_meta["device"])
        if saved_meta["device"] != recorded_device:
            raise RuntimeError("Config and metadata disagree on training device")
        if device is not None and device != recorded_device:
            raise ValueError(f"Run was created for device={recorded_device}; requested {device}")
        device = recorded_device
        if state["status"] == "completed":
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            return
    else:
        if config_path.exists() or state_path(run_dir).exists():
            raise FileExistsError("Run already exists; use resume")
        device = device or "cpu"
        config = config_for_tasks(task_steps, target_steps, device=device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to create a CUDA run")
        for name in ("working", "committed", "abandoned", "logs", "reports"):
            (run_dir / name).mkdir(exist_ok=True)
        atomic_json(config_path, config)
        meta = metadata(torch, gymnasium, stable_baselines3, sb3_contrib)
        meta["device"] = device
        meta["resume_episode_policy"] = "Restore full TetrisEnv, TetrisCore, Monitor and SB3 last observation from committed Task."
        meta.update({"ppo_parameters": config["ppo"], "task_steps": task_steps,
                     "target_steps": target_steps, "run_seed": config["run_seed"]})
        atomic_json(run_dir / "metadata.json", meta)
        state = initial_state(config)
        write_state(run_dir, state)
        log_event(run_dir, "STARTED", task_steps=task_steps, target=target_steps)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Recorded CUDA device is unavailable; refusing to switch devices")
    discard_uncommitted(run_dir, state["committed_task"])
    rebuild_metrics(run_dir, state["committed_task"])
    env = TetrisEnv(max_pieces=config["max_pieces"])
    if state["committed_task"]:
        model, _, _ = load_committed(MaskablePPO, run_dir, state["committed_task"], env, device)
        log_event(run_dir, "RESUMED", committed_task=state["committed_task"],
                  committed_steps=state["committed_steps"])
    else:
        ppo = config["ppo"]
        model = MaskablePPO("MlpPolicy", env,
                            policy_kwargs={"net_arch": {"pi": config["network"]["pi"],
                                                        "vf": config["network"]["vf"]},
                                           "activation_fn": torch.nn.Tanh},
                            learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
                            gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
                            n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
                            n_epochs=ppo["n_epochs"], ent_coef=ppo["ent_coef"],
                            vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"],
                            seed=config["run_seed"], device=device, verbose=0)
        if resume:
            log_event(run_dir, "RESUMED", committed_task=0, committed_steps=0)

    class Capture(KVWriter):
        def __init__(self):
            self.values = {}

        def write(self, key_values, key_excluded, step=0):
            self.values.update(key_values)

        def close(self):
            pass

    class TaskCallback(BaseCallback):
        def __init__(self, working_dir):
            super().__init__()
            self.working_dir = working_dir
            self.episodes = []
            self.game_overs = 0
            self.truncations = 0
            self.cleared_lines = 0

        def _on_step(self):
            info = self.locals["infos"][0]
            self.cleared_lines += info.get("cleared_lines", 0)
            if self.locals["dones"][0]:
                episode = info.get("episode", {})
                self.episodes.append({"reward": episode.get("r"), "length": episode.get("l"),
                                      "score": info["score"], "lines": info["lines"],
                                      "pieces": info["pieces"]})
                self.game_overs += int(info["game_over"])
                self.truncations += int(not info["game_over"])
            return True

        def _on_rollout_end(self):
            atomic_json(self.working_dir / "progress.json",
                        {"working_steps": self.model.num_timesteps, "updated_at": now()})

    completed_this_process = 0
    state.update({"status": "running", "pid": os.getpid()})
    write_state(run_dir, state)
    try:
        while state["committed_steps"] < state["target_steps"]:
            number = state["committed_task"] + 1
            working_dir = run_dir / "working" / f"{task_name(number)}.tmp"
            working_dir.mkdir()
            atomic_json(working_dir / "task_meta.json",
                        {"task_id": number, "status": "in_progress",
                         "start_step": state["committed_steps"], "started_at": now(),
                         "device": device})
            state.update({"working_task": number, "working_steps": state["committed_steps"]})
            write_state(run_dir, state)
            log_event(run_dir, "TASK_STARTED", task=number, committed_steps=state["committed_steps"])
            capture = Capture()
            tensorboard = TensorBoardOutputFormat(str(working_dir / "tensorboard"))
            model.set_logger(Logger(str(working_dir), [tensorboard, capture]))
            callback = TaskCallback(working_dir)
            began = time.monotonic()
            model.learn(total_timesteps=state["task_steps"], callback=callback,
                        reset_num_timesteps=False, use_masking=True)
            seconds = time.monotonic() - began
            model.logger.dump(model.num_timesteps)
            model.logger.close()  # Close TensorBoard before its directory is renamed.
            monitor = model.get_env().envs[0]
            destination = commit_task(run_dir, state, working_dir, model, env, monitor,
                                      callback, capture, seconds)
            completed_this_process += 1
            if max_tasks is not None and completed_this_process >= max_tasks:
                log_event(run_dir, "TEST_STOP_AFTER_COMMIT", task=number)
                break
        if state["committed_steps"] >= state["target_steps"]:
            state.update({"status": "completed", "pid": None})
            write_state(run_dir, state)
    except Exception as error:
        # No working data becomes official; the previous commit pointer stays intact.
        state.update({"status": "failed", "pid": None})
        write_state(run_dir, state)
        log_event(run_dir, "FAILED", error=repr(error))
        raise
    finally:
        env.close()
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--task-steps", type=int, default=4096)
    parser.add_argument("--target-steps", type=int, default=32768)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--max-tasks", type=int, default=None, help="Test-only stop after complete commits")
    args = parser.parse_args()
    if args.status:
        inspect_status(args.run_dir)
    else:
        run(args.run_dir, resume=args.resume, task_steps=args.task_steps,
            target_steps=args.target_steps, max_tasks=args.max_tasks, device=args.device)


if __name__ == "__main__":
    main()
