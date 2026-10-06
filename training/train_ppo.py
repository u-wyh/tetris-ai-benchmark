"""Chunked, resumable MaskablePPO smoke experiment. No training starts on import."""

import argparse
import csv
import fcntl
import json
import os
import pickle
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "ppo_smoke_seed42"
METRIC_FIELDS = (
    "timestamp", "timestep", "rollout_mean_reward", "episode_length",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
    "learning_rate", "mean_score", "mean_lines", "mean_pieces",
    "game_over_count", "truncated_count", "cleared_lines", "chunk_seconds",
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def event(run_dir, name, **fields):
    line = f"{now()} {name}" + "".join(f" {key}={value}" for key, value in fields.items()) + "\n"
    with (run_dir / "events.log").open("a", encoding="utf8") as file:
        file.write(line)
        file.flush()
        os.fsync(file.fileno())


def append_csv(path, fields, row):
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def default_config(target=50000, chunk=5120, first_stage=20480, eval_every=10240,
                   checkpoint_every=10240, eval_seeds=None):
    if chunk < 1024 or chunk % 1024 or first_stage % chunk:
        raise ValueError("chunk must align with the 1024-step PPO rollout")
    return {
        "algorithm": "sb3_contrib.MaskablePPO", "policy": "MlpPolicy",
        "network": {"pi": [256, 256], "vf": [256, 256], "activation": "Tanh"},
        "ppo": {"learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.95,
                "clip_range": 0.2, "n_steps": 1024, "batch_size": 256,
                "n_epochs": 5, "ent_coef": 0.01, "vf_coef": 0.5,
                "max_grad_norm": 0.5, "n_envs": 1},
        "reward": {"placement": 0.001, "line_clear": [0, 1, 3, 5, 8], "game_over": -2},
        "observation_size": 237, "action_size": 1840, "max_pieces": 10000,
        "run_seed": 42, "target_total_steps": target, "chunk_steps": chunk,
        "first_stage_steps": first_stage, "eval_every_steps": eval_every,
        "checkpoint_every_steps": checkpoint_every,
        "eval_seeds": eval_seeds or [10001, 10002, 10003],
        "effective_target_boundary": ((target + 1023) // 1024) * 1024,
    }


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def metadata(torch, gymnasium, sb3, contrib):
    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    cuda = torch.cuda.is_available()
    return {"git_commit": git_value("rev-parse", "HEAD"),
            "dirty_worktree": bool(git_value("status", "--porcelain")),
            "python": platform.python_version(), "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "torch": torch.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": sb3.__version__, "sb3_contrib": contrib.__version__,
            "cuda_available": cuda, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "device": "cuda" if cuda else "cpu", "cpu": cpu or f"{os.cpu_count()} logical CPUs",
            "os": platform.platform(), "hostname": socket.gethostname(), "started_at": now(),
            "resume_episode_policy": "Reset the in-progress episode; model, optimizer, timesteps, and global RNG states are restored.",
    }


def save_model_atomic(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.zip")
    try:
        model.save(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_trainer_state(path, model, env, config, phase, best_reward, elapsed):
    import numpy as np
    import torch

    # The environment snapshot is retained for audit. SB3.load(force_reset=True)
    # deliberately starts a new episode, since its Monitor/VecEnv wrappers are new.
    core = env.core
    state = {"python_random": random.getstate(), "numpy_random": np.random.get_state(),
             "torch_cpu_random": torch.get_rng_state(),
             "torch_cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "num_timesteps": model.num_timesteps, "run_seed": config["run_seed"],
             "phase": phase, "target_total_steps": config["target_total_steps"],
             "best_mean_reward": best_reward, "elapsed_seconds": elapsed,
             "environment_snapshot": {"board": core.board, "current": core.current,
                                      "queue": core.queue, "bag": core.bag,
                                      "gameplay_rng_state": core.rng.state,
                                      "hold": core.hold, "hold_used": core.hold_used,
                                      "score": core.score, "lines": core.lines,
                                      "level": core.level, "phase": core.phase,
                                      "episode_seed": env.episode_seed,
                                      "pieces": env.pieces}}
    atomic_bytes(path, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu_random"])
    if torch.cuda.is_available() and state["torch_cuda_random"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def evaluate(model, config):
    from training.env import TetrisEnv

    episodes = []
    for seed in config["eval_seeds"]:
        env = TetrisEnv(max_pieces=config["max_pieces"])
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        while True:
            mask = env.action_masks()
            if not mask.any():
                raise RuntimeError(f"Evaluation seed {seed} has no legal action before termination")
            action, _ = model.predict(observation, deterministic=True, action_masks=mask)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                episodes.append({"seed": seed, "reward": total_reward, "score": info["score"],
                                 "lines": info["lines"], "pieces": info["pieces"],
                                 "death": int(terminated), "truncate": int(truncated)})
                env.close()
                break
    count = len(episodes)
    return {"mean_reward": sum(item["reward"] for item in episodes) / count,
            "mean_score": sum(item["score"] for item in episodes) / count,
            "mean_lines": sum(item["lines"] for item in episodes) / count,
            "mean_pieces": sum(item["pieces"] for item in episodes) / count,
            "death_count": sum(item["death"] for item in episodes),
            "truncate_count": sum(item["truncate"] for item in episodes),
            "episodes": episodes}


def status(run_dir):
    path = run_dir / "status.json"
    if not path.exists():
        print(f"Run: {run_dir.name}\nStatus: not started")
        return
    data = json.loads(path.read_text())
    pid = data.get("pid")
    running = False
    if pid:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            pass
    latest = run_dir / "checkpoints" / "latest.zip"
    latest_time = datetime.fromtimestamp(latest.stat().st_mtime).astimezone().isoformat(timespec="seconds") if latest.exists() else "none"
    metrics = run_dir / "training_metrics.csv"
    last = list(csv.DictReader(metrics.open()))[-1] if metrics.exists() else {}
    safe = not running and data.get("status") in ("awaiting_resume_test", "paused", "completed")
    print(f"Run: {run_dir.name}\nStatus: {data.get('status')}\nRunning: {'YES' if running else 'NO'}"
          f"\nPID: {pid if running else '-'}\nSteps: {data.get('steps', 0)} / {data.get('target_steps', 0)}"
          f"\nLatest: {latest_time}\nBest evaluation: {data.get('best_mean_reward')}"
          f"\nRecent mean reward: {last.get('rollout_mean_reward', '-') }"
          f"\nDevice: {data.get('device', 'unknown')}\nSafe to shutdown: {'YES' if safe else 'NO'}")


def train(run_dir, resume=False, overrides=None):
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
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)
    lock_file = (run_dir / ".train.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("This run already has an active trainer") from error

    config_path = run_dir / "config.json"
    latest = run_dir / "checkpoints" / "latest.zip"
    state_path = run_dir / "trainer_state.pkl"
    if resume:
        if not latest.exists() or not config_path.exists():
            raise FileNotFoundError("Resume requires the existing config.json and checkpoints/latest.zip")
        config = json.loads(config_path.read_text())
        previous_state = pickle.loads(state_path.read_bytes()) if state_path.exists() else None
    else:
        if latest.exists() or config_path.exists():
            raise FileExistsError("Run already exists; use --resume to continue it")
        config = default_config(**(overrides or {}))
        atomic_json(config_path, config)
        atomic_json(run_dir / "metadata.json", metadata(torch, gymnasium, stable_baselines3, sb3_contrib))
        previous_state = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = TetrisEnv(max_pieces=config["max_pieces"])
    if resume:
        model = MaskablePPO.load(str(latest), env=env, device=device, force_reset=True)
        previous_steps = model.num_timesteps
        if previous_state and previous_state["num_timesteps"] == previous_steps:
            restore_rng(previous_state)
        else:
            event(run_dir, "RNG_STATE_MISMATCH", checkpoint_steps=previous_steps)
        event(run_dir, "RESUMED", previous_steps=previous_steps)
    else:
        ppo = config["ppo"]
        model = MaskablePPO(
            "MlpPolicy", env,
            policy_kwargs={"net_arch": {"pi": config["network"]["pi"],
                                        "vf": config["network"]["vf"]},
                           "activation_fn": torch.nn.Tanh},
            learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
            n_steps=ppo["n_steps"], batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
            ent_coef=ppo["ent_coef"], vf_coef=ppo["vf_coef"],
            max_grad_norm=ppo["max_grad_norm"], seed=config["run_seed"],
            device=device, verbose=0)
        event(run_dir, "STARTED", seed=config["run_seed"], target=config["target_total_steps"])

    # The default policy activation is Tanh; policy_kwargs only needs the net architecture.
    class CaptureFormat(KVWriter):
        def __init__(self):
            self.values = {}

        def write(self, key_values, key_excluded, step=0):
            self.values.update(key_values)

        def close(self):
            pass

    capture = CaptureFormat()
    tb = TensorBoardOutputFormat(str(run_dir / "tensorboard"))
    model.set_logger(Logger(str(run_dir / "logs"), [tb, capture]))

    class ChunkMetrics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = []
            self.cleared_lines = 0
            self.game_overs = 0
            self.truncations = 0

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

    def mean(values):
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else ""

    best_reward = previous_state.get("best_mean_reward") if previous_state else None
    elapsed = previous_state.get("elapsed_seconds", 0.0) if previous_state else 0.0
    passed_resume_test = resume or (previous_state and previous_state.get("phase") == "awaiting_resume_test")
    pid = os.getpid()
    atomic_json(run_dir / "status.json", {"status": "running", "steps": model.num_timesteps,
                                        "target_steps": config["target_total_steps"], "pid": pid,
                                        "device": device, "best_mean_reward": best_reward})
    (run_dir / "trainer.pid").write_text(str(pid))
    stop_signal = {"name": None}

    def on_signal(number, frame):
        stop_signal["name"] = signal.Signals(number).name

    old_int = signal.signal(signal.SIGINT, on_signal)
    old_term = signal.signal(signal.SIGTERM, on_signal)
    try:
        while model.num_timesteps < config["target_total_steps"]:
            remaining = config["target_total_steps"] - model.num_timesteps
            chunk = min(config["chunk_steps"], remaining)
            callback = ChunkMetrics()
            started = time.monotonic()
            model.learn(total_timesteps=chunk, callback=callback,
                        reset_num_timesteps=False, use_masking=True,
                        tb_log_name=run_dir.name)
            chunk_seconds = time.monotonic() - started
            elapsed += chunk_seconds
            # SB3 dumps rollout values before the PPO update; flush the final train values.
            model.logger.dump(model.num_timesteps)
            values = capture.values.copy()
            entropy_loss = values.get("train/entropy_loss")
            row = {"timestamp": now(), "timestep": model.num_timesteps,
                   "rollout_mean_reward": mean([episode["reward"] for episode in callback.episodes]),
                   "episode_length": mean([episode["length"] for episode in callback.episodes]),
                   "policy_loss": values.get("train/policy_gradient_loss", ""),
                   "value_loss": values.get("train/value_loss", ""),
                   "entropy": -entropy_loss if entropy_loss is not None else "",
                   "approx_kl": values.get("train/approx_kl", ""),
                   "clip_fraction": values.get("train/clip_fraction", ""),
                   "learning_rate": values.get("train/learning_rate", ""),
                   "mean_score": mean([episode["score"] for episode in callback.episodes]),
                   "mean_lines": mean([episode["lines"] for episode in callback.episodes]),
                   "mean_pieces": mean([episode["pieces"] for episode in callback.episodes]),
                   "game_over_count": callback.game_overs,
                   "truncated_count": callback.truncations,
                   "cleared_lines": callback.cleared_lines,
                   "chunk_seconds": round(chunk_seconds, 3)}

            if model.num_timesteps % config["eval_every_steps"] == 0:
                result = evaluate(model, config)
                append_csv(run_dir / "evaluations.csv",
                           ("timestamp", "timestep", "mean_reward", "mean_score", "mean_lines",
                            "mean_pieces", "death_count", "truncate_count"),
                           {"timestamp": now(), "timestep": model.num_timesteps,
                            **{key: value for key, value in result.items() if key != "episodes"}})
                event(run_dir, "EVALUATED", steps=model.num_timesteps,
                      mean_reward=round(result["mean_reward"], 4))
                if best_reward is None or result["mean_reward"] > best_reward:
                    best_reward = result["mean_reward"]
                    save_model_atomic(model, run_dir / "checkpoints" / "best.zip")
            if model.num_timesteps % config["checkpoint_every_steps"] == 0:
                save_model_atomic(model, run_dir / "checkpoints" / f"step_{model.num_timesteps:06d}.zip")

            if model.num_timesteps >= config["target_total_steps"]:
                phase = "completed"
            elif not passed_resume_test and model.num_timesteps >= config["first_stage_steps"]:
                phase = "awaiting_resume_test"
            elif (run_dir / "PAUSE_REQUESTED").exists() or stop_signal["name"]:
                phase = "paused"
            else:
                phase = "running"

            save_trainer_state(state_path, model, env, config, phase, best_reward, elapsed)
            save_model_atomic(model, latest)
            append_csv(run_dir / "training_metrics.csv", METRIC_FIELDS, row)
            event(run_dir, "CHUNK_SAVED", steps=model.num_timesteps, status=phase)
            atomic_json(run_dir / "status.json", {"status": phase, "steps": model.num_timesteps,
                                                "target_steps": config["target_total_steps"],
                                                "pid": pid if phase == "running" else None,
                                                "device": device, "best_mean_reward": best_reward,
                                                "elapsed_seconds": round(elapsed, 2)})
            if phase != "running":
                if stop_signal["name"]:
                    event(run_dir, "SIGNAL_HANDLED", signal=stop_signal["name"])
                if phase == "completed":
                    save_model_atomic(model, run_dir / "checkpoints" / "final.zip")
                else:
                    (run_dir / "PAUSE_REQUESTED").unlink(missing_ok=True)
                break
    except Exception as error:
        event(run_dir, "FAILED", error=repr(error))
        atomic_json(run_dir / "status.json", {"status": "failed", "steps": model.num_timesteps,
                                            "target_steps": config["target_total_steps"],
                                            "pid": None, "device": device,
                                            "best_mean_reward": best_reward})
        raise
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        (run_dir / "trainer.pid").unlink(missing_ok=True)
        model.logger.close()
        env.close()
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    # Test-only overrides; normal smoke training uses the fixed default config.
    parser.add_argument("--target-steps", type=int, default=50000)
    parser.add_argument("--chunk-steps", type=int, default=5120)
    parser.add_argument("--first-stage-steps", type=int, default=20480)
    parser.add_argument("--eval-every", type=int, default=10240)
    parser.add_argument("--checkpoint-every", type=int, default=10240)
    args = parser.parse_args()
    if args.status:
        status(args.run_dir)
    else:
        train(args.run_dir, resume=args.resume,
              overrides={"target": args.target_steps, "chunk": args.chunk_steps,
                         "first_stage": args.first_stage_steps, "eval_every": args.eval_every,
                         "checkpoint_every": args.checkpoint_every})


if __name__ == "__main__":
    main()
