"""100k+ random legal-step stress test and throughput benchmark.

Run: .venv/bin/python tests/stress_tetris_env.py --steps 100000
"""

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.env import TetrisEnv  # noqa: E402


def run(steps, max_pieces=10000):
    env = TetrisEnv(max_pieces=max_pieces)
    rng = np.random.default_rng(20261006)
    observation, _ = env.reset(seed=12345)
    if not env.observation_space.contains(observation):
        raise AssertionError("Initial observation is outside observation_space")
    episodes = 1
    terminations = 0
    truncations = 0
    mask_seconds = 0.0
    step_seconds = 0.0
    started = perf_counter()
    for index in range(steps):
        start = perf_counter()
        mask = env.action_masks()
        mask_seconds += perf_counter() - start
        if mask.shape != (1840,) or mask.dtype != np.bool_:
            raise AssertionError(f"Bad action mask at step {index}")
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise AssertionError(f"Playing state has no legal action at step {index}")
        action = int(legal[rng.integers(len(legal))])
        start = perf_counter()
        observation, reward, terminated, truncated, info = env.step(action)
        step_seconds += perf_counter() - start
        if not env.observation_space.contains(observation) or not math.isfinite(reward):
            raise AssertionError(f"Invalid observation or reward at step {index}")
        if info["pieces"] < 1 or info["game_over"] != terminated:
            raise AssertionError(f"Invalid info or termination at step {index}")
        if terminated or truncated:
            terminations += int(terminated)
            truncations += int(truncated)
            observation, reset_info = env.reset(seed=12345 + episodes)
            episodes += 1
            if not env.observation_space.contains(observation) or reset_info["pieces"] != 0:
                raise AssertionError(f"Invalid reset at episode {episodes}")
    elapsed = perf_counter() - started
    return {"steps": steps, "episodes": episodes, "terminations": terminations,
            "truncations": truncations, "elapsedSeconds": round(elapsed, 2),
            "stepsPerSecond": round(steps / elapsed, 1),
            "averageStepMs": round(step_seconds * 1000 / steps, 3),
            "averageMaskMs": round(mask_seconds * 1000 / steps, 3)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--max-pieces", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(run(args.steps, args.max_pieces), indent=2), flush=True)
