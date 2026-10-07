"""Compare process-based MaskablePPO rollouts at a fixed 4096-sample batch."""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from training.env import TetrisEnv
from training.train_ppo import default_config
from training.vector_env import make_vector_env


class TimedTetrisEnv(TetrisEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_seconds = 0.0
        self.mask_seconds = 0.0
        self.mask_within_step = 0.0
        self.mask_calls = 0
        self.in_step = False

    def action_masks(self):
        start = time.perf_counter()
        result = super().action_masks()
        elapsed = time.perf_counter() - start
        self.mask_seconds += elapsed
        self.mask_calls += 1
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

    def get_timing(self):
        return {"step_seconds": self.step_seconds, "mask_seconds": self.mask_seconds,
                "mask_within_step": self.mask_within_step, "mask_calls": self.mask_calls}


def cpu_snapshot(pids):
    lines = Path("/proc/stat").read_text().splitlines()
    cpu = []
    for line in lines:
        if not line.startswith("cpu"):
            break
        values = [int(value) for value in line.split()[1:]]
        cpu.append((sum(values), values[3] + values[4]))
    ticks = 0
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            ticks += int(stat[11]) + int(stat[12])
        except FileNotFoundError:
            pass
    return cpu, ticks, time.monotonic()


def monitor_resources(stop, pids, device, samples):
    previous = cpu_snapshot(pids)
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    while not stop.wait(1.0):
        current = cpu_snapshot(pids)
        total_delta = current[0][0][0] - previous[0][0][0]
        idle_delta = current[0][0][1] - previous[0][0][1]
        elapsed = current[2] - previous[2]
        row = {"cpu_percent": 100 * (total_delta - idle_delta) / total_delta,
               "process_core_equivalents": (current[1] - previous[1]) / ticks_per_second / elapsed}
        per_core = []
        for old, new in zip(previous[0][1:], current[0][1:]):
            delta = new[0] - old[0]
            per_core.append(100 * (delta - (new[1] - old[1])) / delta if delta else 0)
        row["busy_logical_cores"] = sum(value > 50 for value in per_core)
        if device == "cuda":
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                try:
                    util, memory, temp, power = (float(part.strip()) for part in result.stdout.splitlines()[0].split(","))
                    row.update({"gpu_percent": util, "gpu_memory_mib": memory,
                                "gpu_temp_c": temp, "gpu_power_w": power})
                except (IndexError, ValueError):
                    pass
        samples.append(row)
        previous = current


def summarize(samples, key):
    values = [sample[key] for sample in samples if key in sample]
    return {"mean": statistics.mean(values), "peak": max(values)} if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_envs > os.cpu_count():
        parser.error("n_envs exceeds logical CPU count")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.run_dir.exists():
        parser.error("run directory already exists")
    args.run_dir.mkdir(parents=True)

    torch.set_num_threads(1)
    config = default_config(target=16384)
    ppo = config["ppo"]
    n_steps = 4096 // args.n_envs
    env = make_vector_env(args.n_envs, base_seed=config["run_seed"],
                          max_pieces=config["max_pieces"], env_class=TimedTetrisEnv)
    try:
        model = MaskablePPO(
            "MlpPolicy", env,
            policy_kwargs={"net_arch": {"pi": config["network"]["pi"],
                                        "vf": config["network"]["vf"]},
                           "activation_fn": torch.nn.Tanh},
            learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
            n_steps=n_steps, batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
            ent_coef=ppo["ent_coef"], vf_coef=ppo["vf_coef"],
            max_grad_norm=ppo["max_grad_norm"], seed=config["run_seed"],
            device=args.device, verbose=0)
        timing = {"rollout_seconds": 0.0, "update_seconds": 0.0}
        collect, train = model.collect_rollouts, model.train

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

        model.collect_rollouts, model.train = timed_collect, timed_train
        samples = []
        stop = threading.Event()
        pids = [os.getpid(), *(process.pid for process in env.processes)]
        sampler = threading.Thread(target=monitor_resources,
                                   args=(stop, pids, args.device, samples), daemon=True)
        sampler.start()
        try:
            start = time.perf_counter()
            model.learn(total_timesteps=16384, use_masking=True)
            if args.device == "cuda":
                torch.cuda.synchronize()
            wall = time.perf_counter() - start
        finally:
            stop.set()
            sampler.join(timeout=3)
        del model.collect_rollouts
        del model.train
        workers = env.env_method("get_timing")
        before = [row["mask_seconds"] for row in workers]
        mask_query_start = time.perf_counter()
        for _ in range(64):
            masks = get_action_masks(env)
            assert masks.shape == (args.n_envs, 1840) and masks.dtype == np.bool_
        mask_query_wall = time.perf_counter() - mask_query_start
        after = env.env_method("get_timing")
        worker_mask_wall = max(a["mask_seconds"] - b for a, b in zip(after, before))
        report = {
            "device": args.device, "n_envs": args.n_envs, "n_steps": n_steps,
            "rollout_batch": args.n_envs * n_steps, "total_timesteps": model.num_timesteps,
            "base_seed": config["run_seed"],
            "worker_seeds": [config["run_seed"] + rank for rank in range(args.n_envs)],
            "torch": torch.__version__, "wall_seconds": wall,
            "steps_per_second": model.num_timesteps / wall,
            "rollout_seconds": timing["rollout_seconds"],
            "rollout_steps_per_second": model.num_timesteps / timing["rollout_seconds"],
            "update_seconds": timing["update_seconds"],
            "worker_step_seconds_total": sum(row["step_seconds"] for row in workers),
            "worker_mask_seconds_total": sum(row["mask_seconds"] for row in workers),
            "worker_environment_and_mask_seconds_total": sum(
                row["step_seconds"] + row["mask_seconds"] - row["mask_within_step"] for row in workers),
            "worker_mask_calls": sum(row["mask_calls"] for row in workers),
            "mask_query_wall_ms": 1000 * mask_query_wall / 64,
            "mask_query_worker_ms": 1000 * worker_mask_wall / 64,
            "mask_query_ipc_and_sync_ms": 1000 * (mask_query_wall - worker_mask_wall) / 64,
            "resource_samples": len(samples),
            "cpu_percent": summarize(samples, "cpu_percent"),
            "process_core_equivalents": summarize(samples, "process_core_equivalents"),
            "busy_logical_cores": summarize(samples, "busy_logical_cores"),
            "gpu_percent": summarize(samples, "gpu_percent"),
            "gpu_memory_mib": summarize(samples, "gpu_memory_mib"),
            "gpu_temp_c": summarize(samples, "gpu_temp_c"),
            "gpu_power_w": summarize(samples, "gpu_power_w"),
        }
        (args.run_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
