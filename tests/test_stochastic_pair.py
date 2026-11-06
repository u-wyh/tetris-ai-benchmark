"""Sampling observation preserves RNG; cache identity pins stochastic protocol."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import evaluate_stochastic_pair as sampling
from training.evaluation.evaluate_resumable import collect_cached_rows


def valid_row(seed=100000, action_seed=2400000):
    return dict(seed=seed, action_rng_seed=action_seed, deterministic=False,
        pieces_survived=4, lines=0, score=0, episode_reward=0.004,
        game_over=True, survived_cap=False, prediction_count=4,
        mean_entropy_nats=0.5, mean_greedy_probability=0.8, sampled_greedy_fraction=0.75)


@pytest.mark.parametrize('patch', [dict(deterministic=True), dict(action_rng_seed=1),
    dict(prediction_count=3), dict(mean_entropy_nats=float('nan')),
    dict(mean_greedy_probability=1.1), dict(sampled_greedy_fraction=-0.1)])
def test_bad_sampling_row_is_rejected(patch):
    with pytest.raises(ValueError):
        sampling.validate_sampling_row(dict(valid_row(), **patch), 100000, 2400000, 2000)


def test_valid_sampling_row():
    sampling.validate_sampling_row(valid_row(), 100000, 2400000, 2000)


def test_cache_rejects_changed_action_seed_or_mode(tmp_path):
    model = tmp_path/'model.zip'
    model.write_bytes(b'frozen test model')
    identity = sampling.sampling_identity(model, [100000], [2400000], 2000, {}, {}, 'test-script')
    assert identity['deterministic'] is False
    rows, _ = collect_cached_rows(tmp_path/'cache', identity, lambda pending: iter([valid_row()]))
    assert rows == [valid_row()]
    for changed in (dict(identity, action_rng_seeds=[2400001]), dict(identity, deterministic=True)):
        with pytest.raises(ValueError, match='identity differs'):
            collect_cached_rows(tmp_path/'cache', changed, lambda pending: pytest.fail('Old rows reused'))


def test_partial_seed_commit_is_reused_without_recomputing(tmp_path):
    model = tmp_path/'model.zip'
    model.write_bytes(b'frozen test model')
    identity = sampling.sampling_identity(model, [100000, 100001], [2400000, 2400001], 2000, {}, {}, 'script')
    def interrupted(pending):
        yield valid_row()
        raise RuntimeError('simulated interrupted evaluation')
    with pytest.raises(RuntimeError, match='interrupted'):
        collect_cached_rows(tmp_path/'cache', identity, interrupted)
    def resumed(pending):
        assert pending == [100001]
        yield valid_row(100001, 2400001)
    rows, reused = collect_cached_rows(tmp_path/'cache', identity, resumed)
    assert reused == [100000] and [r['seed'] for r in rows] == [100000, 100001]


def test_real_policy_observer_leaves_actions_and_rng_exact():
    import numpy as np
    import torch
    from training.candidate_policy import CandidateScoringPolicy
    from training.env.bitboard_candidate_env import BitboardCandidateTetrisEnv
    torch.set_num_threads(1)
    env = BitboardCandidateTetrisEnv(max_pieces=3)
    try:
        obs, _ = env.reset(seed=42)
        mask = env.action_masks()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9137)
            policy = CandidateScoringPolicy(env.observation_space, env.action_space, lambda _: 3e-4)
            for seed in (2400000, 2400001, 2400002):
                torch.manual_seed(seed)
                plain, _ = policy.predict(obs, deterministic=False, action_masks=mask)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                with sampling.observe_distribution(policy) as last:
                    observed, _ = policy.predict(obs, deterministic=False, action_masks=mask)
                np.testing.assert_array_equal(plain, observed)
                assert torch.equal(rng, torch.get_rng_state())
                assert last['calls'] == 1 and 0 <= last['maximum_probability'] <= 1
                assert '_distribution' not in policy.__dict__
            with pytest.raises(RuntimeError):
                with sampling.observe_distribution(policy):
                    raise RuntimeError('probe exception')
            assert '_distribution' not in policy.__dict__
    finally:
        env.close()
