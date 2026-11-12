from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_head_endpoint import objective_gradient
from test_head_representation import fixture


@pytest.mark.parametrize('objective', ['pair_softplus', 'all_legal_cross_entropy'])
def test_endpoint_analytic_gradient_matches_finite_differences(tmp_path, objective):
    arrays, metadata = fixture(tmp_path)
    rng = np.random.default_rng(137)
    arrays['h'] = rng.normal(size=arrays['h'].shape).astype(np.float32)
    arrays['raw_score'] = rng.normal(size=arrays['raw_score'].shape).astype(np.float32)
    delta = rng.normal(size=64)*.05
    value = objective_gradient(arrays, metadata, delta, objective)
    direction = rng.normal(size=64);direction /= np.linalg.norm(direction)
    eps = 1e-5
    above = objective_gradient(arrays, metadata, delta+eps*direction, objective)['total']
    below = objective_gradient(arrays, metadata, delta-eps*direction, objective)['total']
    assert value['gradient']@direction == pytest.approx((above-below)/(2*eps), abs=1e-8)
    assert value['normal_states'] == 316 and value['preference_states'] == 4
    assert sum(value['exposures'].values()) == 256*8
    assert value['exposures']['8'] == 4 and value['exposures'][str(8+19*8)] == 3


def test_endpoint_excludes_validation_and_zero_delta_teacher_kl(tmp_path):
    arrays, metadata = fixture(tmp_path);delta = np.zeros(64)
    first = objective_gradient(arrays, metadata, delta, 'all_legal_cross_entropy')
    assert first['teacher_kl'] == 0.
    for i, state in enumerate(metadata['states']):
        if state['split'] == 'validation':
            arrays['h'][arrays['offsets'][i]:arrays['offsets'][i+1]] = np.nan
            arrays['raw_score'][arrays['offsets'][i]:arrays['offsets'][i+1]] = np.nan
    second = objective_gradient(arrays, metadata, delta, 'all_legal_cross_entropy')
    assert first['total'] == second['total'] and np.array_equal(first['gradient'], second['gradient'])
