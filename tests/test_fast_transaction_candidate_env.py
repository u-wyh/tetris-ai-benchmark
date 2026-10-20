"""The optional cached placement path must preserve training semantics and restore."""

import numpy as np

from training.train_vector_transaction import TransactionCandidateEnv, TransactionFastCandidateEnv
from training.traditional.v2 import V2, V2Config


def test_cached_transaction_env_matches_official_path_and_snapshot():
    ordinary = TransactionCandidateEnv(max_pieces=250)
    cached = TransactionFastCandidateEnv(max_pieces=250)
    restored = TransactionFastCandidateEnv(max_pieces=250)
    policy = V2(V2Config(mode="hold"))
    try:
        ordinary.reset(seed=100016)
        cached.reset(seed=100016)
        for step in range(200):
            assert ordinary.get_observation()["state"].tolist() == cached.get_observation()["state"].tolist()
            assert np.array_equal(ordinary.action_masks(), cached.action_masks())
            action = policy.choose(ordinary.core.get_public_observation(), ordinary.core.get_legal_placements())
            left = ordinary.step(action)
            right = cached.step(action)
            assert np.array_equal(left[0]["candidate"], right[0]["candidate"])
            assert np.array_equal(left[0]["state"], right[0]["state"])
            assert left[1:] == right[1:]
            if step == 99:
                restored.reset(seed=100016)
                restored.set_state(cached.get_state())
                assert restored.core.get_public_observation() == cached.core.get_public_observation()
                assert np.array_equal(restored.action_masks(), cached.action_masks())
            if step >= 100:
                other = restored.step(action)
                assert np.array_equal(other[0]["candidate"], right[0]["candidate"])
                assert other[1:] == right[1:]
    finally:
        ordinary.close()
        cached.close()
        restored.close()
