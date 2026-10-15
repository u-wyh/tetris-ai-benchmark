"""On-policy Candidate PPO environment with sealed, replayed risk-state resets."""
import copy
import gzip
import hashlib
import pickle
import random
from pathlib import Path

import numpy as np

from training.env.candidate_env import OBSERVATION_VERSION
from training.evaluation.ppo_death_capture import public_hash
from training.train_vector_transaction import TransactionCandidateEnv

REWARD_VERSION = "placement-reward-death20-v1"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


class FailureCurriculumEnv(TransactionCandidateEnv):
    def __init__(self, max_pieces=100000, curriculum=None, **kwargs):
        super().__init__(max_pieces=max_pieces, **kwargs)
        if curriculum is None or curriculum["mode"] not in ("normal", "curriculum"):
            raise ValueError("Frozen failure curriculum config required")
        self.curriculum = copy.deepcopy(curriculum)
        self.reward_version = REWARD_VERSION
        self.sampler = None
        self.origin = {"tier": "normal", "source_seed": None, "source_step": 0}
        self.window_pieces = int(curriculum["window_pieces"])
        self.snapshots = []
        if curriculum["mode"] == "curriculum":
            path = Path(curriculum["dataset_path"])
            if file_sha256(path) != curriculum["dataset_sha256"]:
                raise RuntimeError("Curriculum dataset SHA256 mismatch")
            with gzip.open(path, "rb") as file:
                dataset = pickle.load(file)  # Trusted locally built immutable experiment data.
            self.snapshots = dataset["snapshots"]
            if not self.snapshots:
                raise ValueError("Curriculum has no validated snapshots")
            if dataset["ppo_sha256"] != curriculum["ppo_sha256"]:
                raise RuntimeError("Curriculum source PPO mismatch")

    def reset(self, *, seed=None, options=None):
        if self.sampler is None:
            if seed is None:
                raise RuntimeError("First curriculum reset requires worker seed")
            self.sampler = random.Random(seed ^ 0xF41C0A)
        observation, info = super().reset(seed=seed, options=options)
        self.origin = {"tier": "normal", "source_seed": None, "source_step": 0}
        if (self.curriculum["mode"] == "curriculum" and
                self.sampler.random() < self.curriculum["snapshot_probability"]):
            # Tier is sampled first; no single long episode can dominate by row count.
            tier = self.sampler.choice(self.curriculum["tiers"])
            pool = [row for row in self.snapshots if row["tier"] == tier]
            seed_choice = self.sampler.choice(sorted({row["seed"] for row in pool}))
            row = self.sampler.choice([row for row in pool if row["seed"] == seed_choice])
            state = copy.deepcopy(row["state"])
            state["max_pieces"] = self.max_pieces
            state["reward_version"] = self.reward_version
            state["hole_penalty_coef"] = self.hole_penalty_coef
            state["height_penalty_coef"] = self.height_penalty_coef
            state["pieces"] = 0  # Relative course window; Core and its Bag/RNG remain exact.
            state["episode_reward"] = 0.0
            state["episode_count"] = self.episode_count
            state["finished"] = False
            super().set_state(state)
            public = self.core.get_public_observation()
            observation = self.get_observation()
            mask = self.action_masks()
            if public_hash(public) != row["public_hash"] or not np.array_equal(mask, row["mask"]):
                raise RuntimeError("Curriculum snapshot public state or legal mask changed")
            if not np.array_equal(observation["state"], row["observation_state"]):
                raise RuntimeError("Curriculum snapshot actor observation changed")
            self.origin = {"tier": row["tier"], "source_seed": row["seed"],
                           "source_step": row["step"]}
            info = self._info()
        info.update(curriculum_origin=self.origin.copy(), curriculum_reset=True)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if terminated:
            reward -= 18.0  # Existing true-death penalty is -2; total is -20.
            self.episode_reward -= 18.0
            info["raw_reward"] -= 18.0
            info["shaped_reward"] -= 18.0
            info["episode_reward"] = self.episode_reward
        if (not terminated and not truncated and self.origin["tier"] != "normal"
                and self.pieces >= self.window_pieces):
            truncated = True
            self._finished = True
        info.update(curriculum_origin=self.origin.copy(),
                    curriculum_episode_start=self.pieces == 1,
                    curriculum_window_truncated=bool(truncated and self.origin["tier"] != "normal"),
                    true_game_over=bool(terminated), death_penalty=-20.0 if terminated else 0.0)
        return observation, reward, terminated, truncated, info

    def get_state(self):
        state = super().get_state()
        state["curriculum"] = self.curriculum
        state["curriculum_sampler"] = None if self.sampler is None else self.sampler.getstate()
        state["curriculum_origin"] = self.origin.copy()
        return state

    def set_state(self, state):
        if state["curriculum"] != self.curriculum:
            raise RuntimeError("Curriculum worker config changed on restore")
        super().set_state(state)
        self.sampler = random.Random()
        self.sampler.setstate(state["curriculum_sampler"])
        self.origin = state["curriculum_origin"].copy()
