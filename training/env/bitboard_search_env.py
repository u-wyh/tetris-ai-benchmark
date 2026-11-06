"""Explicit training search backend; existing placement and feature semantics."""

from training.env.bitboard_candidate_env import BitboardCandidateTetrisEnv
from training.evaluation.fast_candidate_features import candidate_features
from training.evaluation.fast_legal import get_legal_placements
from training.compact_legal import get_legal_placements as compact_placements
from training.env.bitboard_candidate_env import validate_feature_backend as reference_feature_backend
from training.native_feature_runtime import load_runtime


def validate_search_backend(preset):
    backend = preset.get('candidate_search_backend', 'formal')
    if backend not in ('formal', 'bitboard', 'bitboard_compact'):
        raise ValueError('Unknown Candidate search backend')
    if backend != 'formal' and (not preset.get('candidate_policy')
            or not preset.get('fast_candidate_env')
            or preset.get('candidate_feature_backend') not in ('bitboard', 'bitboard_native')
            or preset.get('failure_curriculum')):
        raise ValueError('Bitboard search requires ordinary fast Candidate bitboard features')
    if preset.get('candidate_feature_backend') == 'bitboard_native' and backend != 'bitboard_compact':
        raise ValueError('Native features require compact bitboard search')
    return backend


def validate_training_feature_backend(preset):
    if preset.get('candidate_feature_backend') == 'bitboard_native':
        reference_feature_backend(dict(preset, candidate_feature_backend='bitboard'))
        validate_search_backend(preset)
        return 'bitboard_native'
    return reference_feature_backend(preset)


class BitboardSearchCandidateTetrisEnv(BitboardCandidateTetrisEnv):
    placement_search = staticmethod(get_legal_placements)
    feature_builder = staticmethod(candidate_features)

    def _features_and_mask(self):
        if self.core is None:
            raise RuntimeError('Call reset() before requesting candidates')
        if self._candidate_cache is None:
            public = self.core.get_public_observation()
            self._placement_cache = self.placement_search(public)
            self._candidate_cache = self.feature_builder(public, self._placement_cache)
        return self._candidate_cache


class CompactSearchCandidateTetrisEnv(BitboardSearchCandidateTetrisEnv):
    placement_search = staticmethod(compact_placements)


class NativeSearchCandidateTetrisEnv(CompactSearchCandidateTetrisEnv):
    def __init__(self, *, native_feature_runtime, **kwargs):
        self.feature_builder = load_runtime(native_feature_runtime)
        super().__init__(**kwargs)
