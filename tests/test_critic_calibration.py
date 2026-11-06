"""Durable GAE reconstruction catches episode and bootstrap corruption."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_critic_calibration import residual_metrics, save_dataset, verify_targets


def example():
    # Worker 0 terminates at t=0; worker 1 continues through final bootstrap.
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    rewards = np.array([[5, 6], [7, 8]], dtype=np.float32)
    # gamma=1, lambda=1: worker0 targets 5 then 7, worker1 targets 6+8+10 then 8+10.
    returns = np.array([[5, 24], [7, 18]], dtype=np.float32)
    return dict(values=values, rewards=rewards, returns=returns, advantages=returns-values,
        episode_starts=np.array([[0, 0], [1, 0]], dtype=np.float32),
        step_dones=np.array([[True, False], [True, False]]),
        last_values=np.array([99, 10], dtype=np.float32), last_dones=np.array([True, False]))


def test_bias_and_spread_have_distinct_metrics():
    constant_bias = residual_metrics([0, 1, 2], [2, 3, 4])
    assert constant_bias['explained_variance'] == 1
    assert constant_bias['mean_bias_squared_over_mse'] == 1
    assert constant_bias['rmse'] == 2
    assert residual_metrics([], []) == {'samples': 0}
    assert residual_metrics([1], [1])['mean_bias_squared_over_mse'] is None
    with pytest.raises(ValueError):
        residual_metrics([np.nan], [1])


def test_durable_round_trip_and_refuse_overwrite(tmp_path):
    arrays = example()
    loaded = save_dataset(tmp_path, arrays, dict(gamma=1., gae_lambda=1.))
    for key in arrays:
        np.testing.assert_array_equal(loaded[key], arrays[key])
    with pytest.raises(FileExistsError):
        save_dataset(tmp_path, arrays, dict(gamma=1., gae_lambda=1.))


@pytest.mark.parametrize('key', ['episode_starts', 'last_values', 'rewards', 'step_dones'])
def test_reconstruction_rejects_corrupted_boundaries_or_bootstrap(key):
    arrays = example()
    arrays[key].flat[-1] += 1
    with pytest.raises(AssertionError):
        verify_targets(arrays, 1., 1.)
