"""Deterministic, process-based TetrisEnv construction for PPO benchmarks."""

from functools import partial

from stable_baselines3.common.vec_env import SubprocVecEnv

from training.env import TetrisEnv


def worker_seeds(base_seed, n_envs):
    if type(n_envs) is not int or n_envs < 1:
        raise ValueError("n_envs must be a positive integer")
    if type(base_seed) is not int or not 0 <= base_seed <= (1 << 32) - n_envs:
        raise ValueError("base_seed must leave room for all uint32 worker seeds")
    return [base_seed + rank for rank in range(n_envs)]


def make_vector_env(n_envs, base_seed=42, max_pieces=10000, env_class=TetrisEnv):
    """Create real workers; Gymnasium seeds are applied on the first reset."""
    seeds = worker_seeds(base_seed, n_envs)
    env = SubprocVecEnv([partial(env_class, max_pieces=max_pieces) for _ in seeds],
                        start_method="forkserver")
    if list(env.seed(base_seed)) != seeds:
        env.close()
        raise RuntimeError("SubprocVecEnv did not assign the expected worker seeds")
    return env
