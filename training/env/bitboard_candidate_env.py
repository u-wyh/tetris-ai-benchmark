"""Opt-in exact-feature implementation; formal legal-placement search is shared."""

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.fast_candidate_features import candidate_features
from training.tetris_core.legal_placements import get_legal_placements


def validate_feature_backend(preset):
    backend = preset.get('candidate_feature_backend', 'official')
    if backend not in ('official', 'bitboard'):
        raise ValueError('Unknown Candidate feature backend')
    if backend == 'bitboard' and (not preset.get('candidate_policy') or not preset.get('fast_candidate_env')
                                  or preset.get('failure_curriculum')):
        raise ValueError('Bitboard features require the ordinary fast Candidate environment')
    return backend


class BitboardCandidateTetrisEnv(FastCandidateTetrisEnv):
    def _features_and_mask(self):
        if self.core is None:
            raise RuntimeError('Call reset() before requesting candidates')
        if self._candidate_cache is None:
            public = self.core.get_public_observation()
            self._placement_cache = get_legal_placements(public)
            self._candidate_cache = candidate_features(public, self._placement_cache)
        return self._candidate_cache
