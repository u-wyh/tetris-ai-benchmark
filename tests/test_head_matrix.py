from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_head_matrix as audit
from test_head_representation import fixture


def test_known_rank_and_projection_residual_do_not_require_a_solver():
    info, basis = audit.spectrum([[2., 0., 0.], [-1., 0., 0.]])
    assert info['machine_rank'] == info['relative_rank'] == 1
    assert info['full_column_condition'] is None
    pair = np.array([1., 3., 0.])
    assert np.linalg.norm(pair - (pair @ basis.T) @ basis) == pytest.approx(3.)


def test_zero_matrix_has_zero_rank_and_empty_row_span():
    info, basis = audit.spectrum(np.zeros((5, 3)))
    assert info['machine_rank'] == info['relative_rank'] == 0
    assert basis.shape == (0, 3) and info['full_column_condition'] is None


def test_full_column_rank_has_known_condition():
    info, _ = audit.spectrum(np.diag([4., 2.]))
    assert info['machine_rank'] == 2 and info['full_column_condition'] == pytest.approx(2.)


def test_necessary_l1_bound_and_exact_zero_contradiction_are_distinct():
    result = audit.necessary_bounds([[2., -1.], [0., 0.], [0., 0.]], [6., 1., -1.])
    assert result['finite_l1_lower_bound'] == pytest.approx(3.)
    assert result['exact_zero_positive_rhs_rows'] == [1]
    assert result['exact_zero_rows'] == 2
    assert result['row_l1_lower_bounds'] == [3., 0., 0.]


def test_validation_changes_cannot_change_training_spectrum_or_pair_bounds(tmp_path):
    arrays, metadata = fixture(tmp_path)
    before = audit.analyze(arrays, metadata)
    for i, state in enumerate(metadata['states']):
        if state['split'] == 'validation':
            arrays['h'][arrays['offsets'][i]+1, 0] = 1.
    after = audit.analyze(arrays, metadata)
    assert after['normal_training_spectrum'] == before['normal_training_spectrum']
    assert after['preferences'] == before['preferences']
    assert after['necessary_bounds'] == before['necessary_bounds']


@pytest.mark.parametrize('matrix', [[[float('nan')]], []])
def test_nonfinite_or_empty_spectrum_is_rejected(matrix):
    with pytest.raises(ValueError):
        audit.spectrum(matrix)
