"""Evaluation-only Candidate environment that reuses validated placements.

The action mask, candidate features, and Core.step all refer to the same public
state. Core.step ordinarily repeats the legal-placement search; this class
supplies the placements already used to build the observation. The override is
scoped to one step and is never present in a saved Core state.
"""

from training.candidate_features import candidate_features
from training.env.candidate_env import CandidateTetrisEnv
from training.env.tetris_env import TetrisEnv
from training.tetris_core.legal_placements import get_legal_placements


class FastCandidateTetrisEnv(CandidateTetrisEnv):
    def _invalidate_candidates(self):
        super()._invalidate_candidates()
        self._placement_cache = None

    def _features_and_mask(self):
        if self.core is None:
            raise RuntimeError("Call reset() before requesting candidates")
        if self._candidate_cache is None:
            public = self.core.get_public_observation()
            self._placement_cache = get_legal_placements(public)
            self._candidate_cache = candidate_features(public, self._placement_cache)
        return self._candidate_cache

    def step(self, action):
        self._features_and_mask()
        placements = self._placement_cache
        core = self.core
        if "get_legal_placements" in core.__dict__:
            raise RuntimeError("Core legal-placement method was already overridden")
        core.get_legal_placements = lambda: placements
        try:
            _, reward, terminated, truncated, info = TetrisEnv.step(self, action)
        finally:
            del core.get_legal_placements
        self._invalidate_candidates()
        return self.get_observation(), reward, terminated, truncated, info
