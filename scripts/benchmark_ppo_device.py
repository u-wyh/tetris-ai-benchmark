"""Short, comparable MaskablePPO device benchmark; run once per device."""

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.env import TetrisEnv
from training.train_ppo import default_config


class TimedEnv(TetrisEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_seconds = 0.0
        self.mask_seconds = 0.0
        self.mask_within_step = 0.0
        self.in_step = False

    def action_masks(self):
        start = time.perf_counter()
        result = super().action_masks()
        elapsed = time.perf_counter() - start
        self.mask_seconds += elapsed
        if self.in_step:
            self.mask_within_step += elapsed
        return result

    def step(self, action):
        start = time.perf_counter()
        self.in_step = True
        try:
            return super().step(action)
        finally:
            self.in_step = False
            self.step_seconds += time.perf_counter() - start


def gpu_sample(stop, samples):
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            try:
                samples.append([float(part.strip()) for part in result.stdout.splitlines()[0].split(",")])
            except (IndexError, ValueError):
                pass
        stop.wait(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1024 or args.steps % 1024:
        parser.error("steps must be a positive multiple of 1024")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.run_dir.exists():
        parser.error("run directory already exists")
    args.run_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    config = default_config(target=args.steps)
    ppo = config["ppo"]
    env = TimedEnv(max_pieces=config["max_pieces"])
    model = MaskablePPO(
        "MlpPolicy", env, policy_kwargs={"net_arch": {"pi": config["network"]["pi"],
                                                    "vf": config["network"]["vf"]},
                                      "activation_fn": torch.nn.Tanh},
        learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
        n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"], ent_coef=ppo["ent_coef"],
        vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"],
        seed=config["run_seed"], device=args.device, verbose=0)
    timing = {"rollout_seconds": 0.0, "update_seconds": 0.0}
    collect = model.collect_rollouts
    train = model.train

    def timed_collect(*a, **kw):
        start = time.perf_counter()
        result = collect(*a, **kw)
        if args.device == "cuda":
            torch.cuda.synchronize()
        timing["rollout_seconds"] += time.perf_counter() - start
        return result

    def timed_train(*a, **kw):
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = train(*a, **kw)
        if args.device == "cuda":
            torch.cuda.synchronize()
        timing["update_seconds"] += time.perf_counter() - start
        return result

    model.collect_rollouts = timed_collect
    model.train = timed_train
    samples = []
    stop = threading.Event()
    sampler = None
    if args.device == "cuda":
        sampler = threading.Thread(target=gpu_sample, args=(stop, samples), daemon=True)
        sampler.start()
    try:
        start = time.perf_counter()
        model.learn(total_timesteps=args.steps, use_masking=True)
        if args.device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=3)
    del model.collect_rollouts
    del model.train
    model.save(str(args.run_dir / "model.zip"))
    env.close()
    report = {
        "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
        "steps": model.num_timesteps, "seed": config["run_seed"],
        "ppo": ppo, "network": config["network"], "wall_seconds": wall,
        "steps_per_second": model.num_timesteps / wall,
        "rollout_seconds": timing["rollout_seconds"],
        "update_seconds": timing["update_seconds"],
        "environment_and_mask_seconds": env.step_seconds + env.mask_seconds - env.mask_within_step,
        "mask_seconds": env.mask_seconds,
        "gpu_samples": len(samples),
    }
    if samples:
        report.update({
            "gpu_util_mean_percent": statistics.mean(row[0] for row in samples),
            "gpu_util_peak_percent": max(row[0] for row in samples),
            "gpu_memory_peak_mib": max(row[1] for row in samples),
            "gpu_temp_mean_c": statistics.mean(row[2] for row in samples),
            "gpu_temp_peak_c": max(row[2] for row in samples),
            "gpu_power_mean_w": statistics.mean(row[3] for row in samples),
            "gpu_power_peak_w": max(row[3] for row in samples),
        })
    (args.run_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
