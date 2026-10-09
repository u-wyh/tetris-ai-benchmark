"""Versioned Dict observation with cached public candidate features."""

import gymnasium as gym
import numpy as np

from training.candidate_features import FEATURE_NAMES, FEATURE_VERSION, candidate_features
from training.env.tetris_env import TetrisEnv
from training.tetris_core import ACTION_COUNT

OBSERVATION_VERSION = "candidate-dict-state237-features16-v1"


class CandidateTetrisEnv(TetrisEnv):
    def __init__(self, max_pieces=10000, hole_penalty_coef=0.0, height_penalty_coef=0.0):
        super().__init__(max_pieces, hole_penalty_coef, height_penalty_coef)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(0.0, 1.0, shape=(237,), dtype=np.float32),
            "candidate": gym.spaces.Box(0, 255,
                shape=(ACTION_COUNT, len(FEATURE_NAMES)), dtype=np.uint8),
        })
        self._candidate_cache = None
        self.feature_version = FEATURE_VERSION

    def _invalidate_candidates(self):
        self._candidate_cache = None

    def _features_and_mask(self):
        if self.core is None:
            raise RuntimeError("Call reset() before requesting candidates")
        if self._candidate_cache is None:
            self._candidate_cache = candidate_features(self.core.get_public_observation())
        return self._candidate_cache

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self._invalidate_candidates()
        return self.get_observation(), info

    def get_observation(self):
        public = self.core.get_public_observation()
        matrix, _ = self._features_and_mask()
        return {"state": self._encode_observation(public), "candidate": matrix.copy()}

    def action_masks(self):
        return self._features_and_mask()[1].copy()

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        self._invalidate_candidates()
        return self.get_observation(), reward, terminated, truncated, info

    def set_state(self, state):
        if state.get("candidate_feature_version") != FEATURE_VERSION:
            raise ValueError("Checkpoint candidate feature version differs from environment")
        super().set_state(state)
        self._invalidate_candidates()

    def get_state(self):
        state = super().get_state()
        state["candidate_feature_version"] = FEATURE_VERSION
        return state
