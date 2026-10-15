import gzip
import pickle

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from training.env.candidate_env import CandidateTetrisEnv
from training.env.failure_curriculum_env import FailureCurriculumEnv, file_sha256
from training.evaluation.hybrid_queue import HASHES
from training.evaluation.ppo_death_capture import public_hash


def test_curriculum_core_rng_mask_window_and_sampler_restore(tmp_path):
    source = CandidateTetrisEnv(max_pieces=100)
    source.reset(seed=12345)
    for _ in range(3):
        mask = source.action_masks()
        source.step(int(np.flatnonzero(mask)[0]))
    public = source.core.get_public_observation()
    row = {"seed": 12345, "step": 3, "tier": "mid_251_500", "source": "test",
           "public_hash": public_hash(public), "mask": source.action_masks(),
           "observation_state": source.get_observation()["state"],
           "state": source.get_state()}
    path = tmp_path / "states.pkl.gz"
    with gzip.open(path, "wb") as file:
        pickle.dump({"ppo_sha256": HASHES["ppo"], "snapshots": [row]}, file)
    config = {"mode": "curriculum", "dataset_path": str(path),
              "dataset_sha256": file_sha256(path), "ppo_sha256": HASHES["ppo"],
              "snapshot_probability": 1.0, "window_pieces": 2,
              "death_penalty": -20.0, "tiers": ["mid_251_500"]}
    env = FailureCurriculumEnv(max_pieces=100, curriculum=config)
    try:
        obs, info = env.reset(seed=222)
        assert info["curriculum_origin"]["source_seed"] == 12345
        assert np.array_equal(obs["state"], row["observation_state"])
        assert np.array_equal(env.action_masks(), row["mask"])
        assert env.core.rng.state == row["state"]["core"].rng.state
        saved = env.get_state()
        first = int(np.flatnonzero(env.action_masks())[0])
        _, _, over, capped, info = env.step(first)
        assert not over and not capped
        env.set_state(saved)
        assert env.get_state()["curriculum_sampler"] == saved["curriculum_sampler"]
        assert env.core.rng.state == saved["core"].rng.state
        env.step(first)
        second = int(np.flatnonzero(env.action_masks())[0])
        _, _, over, capped, info = env.step(second)
        assert not over and capped and info["curriculum_window_truncated"]
        assert not info["true_game_over"] and info["death_penalty"] == 0
        assert env.pieces == 2 and env.core.score >= public["score"]
    finally:
        env.close()
        source.close()
    vector = DummyVecEnv([lambda: FailureCurriculumEnv(max_pieces=100,curriculum=config)])
    try:
        vector.seed(222)
        vector.reset()
        for index in range(2):
            action = int(np.flatnonzero(vector.env_method("action_masks")[0])[0])
            _, _, dones, infos = vector.step(np.array([action]))
        assert dones[0]
        assert infos[0]["TimeLimit.truncated"]
        assert "terminal_observation" in infos[0]  # PPO bootstraps timeout value.
    finally:
        vector.close()
