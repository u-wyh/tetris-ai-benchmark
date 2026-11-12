import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import head_newton as newton
from audit_head_endpoint import objective_gradient
from test_head_preference_probe import extended_fixture
from head_preference_probe import append_training_block


def data(tmp_path):
    arrays, metadata, cases, rows = extended_fixture(tmp_path)
    arrays, metadata = append_training_block(arrays, metadata, cases, rows)
    rng = np.random.default_rng(548)
    arrays['h'] = (rng.normal(size=arrays['h'].shape)*.1).astype(np.float32)
    arrays['raw_score'] = rng.normal(size=arrays['raw_score'].shape).astype(np.float32)
    return arrays, metadata


def receipt():
    return dict(head_full_objective_passes=0, head_hessian_passes=0,
                line_search_objective_passes=0, head_training_forwards=0)


def test_exact_schedule_weights_batch8_and_validation_isolation(tmp_path):
    arrays, metadata = data(tmp_path);rows = newton.training_rows(metadata)
    assert len(rows) == 324 and len(set(i for i, _ in rows)) == 324
    assert sum(w for i, w in rows[:8]) == 1. and sum(w for i, w in rows[8:]) == 1.
    for i, state in enumerate(metadata['states']):
        if state['split'] == 'validation':
            arrays['h'][arrays['offsets'][i]:arrays['offsets'][i+1]] = np.nan
    batches = newton.pack_batches(arrays, metadata, 'cpu')
    assert len(batches) == 41 and all(b['h'].shape[0] == 8 for b in batches)
    assert all(torch.isfinite(b['h']).all() for b in batches)
    assert batches[-1]['weights'][4:].eq(0).all()
    assert all(torch.isfinite(b['logq']).all() for b in batches)


def test_full_objective_gradient_matches_independent_numpy_and_hessian_finite_difference(tmp_path):
    arrays, metadata = data(tmp_path);batches = newton.pack_batches(arrays, metadata, 'cpu')
    delta = torch.linspace(-.2, .2, 64, dtype=torch.float64);counts = receipt()
    value, gradient, matrix = newton.evaluate(batches, delta, counts, lambda: None, hessian=True)
    expected = objective_gradient(arrays, metadata, delta.numpy(), 'all_legal_cross_entropy')
    assert value.item() == pytest.approx(expected['total'], abs=1e-12)
    assert np.allclose(gradient.numpy(), expected['gradient'], atol=1e-12, rtol=0.)
    assert torch.equal(matrix, matrix.T) and torch.linalg.eigvalsh(matrix).min().item() >= .02-1e-12
    direction = torch.sin(torch.arange(64, dtype=torch.float64));direction /= direction.norm();eps = 1e-5
    plus = newton.evaluate(batches, delta+eps*direction, counts, lambda: None)[1]
    minus = newton.evaluate(batches, delta-eps*direction, counts, lambda: None)[1]
    assert torch.allclose((plus-minus)/(2*eps), matrix@direction, atol=1e-9, rtol=0.)
    factor = torch.linalg.cholesky(matrix);step = torch.cholesky_solve(-gradient[:, None], factor).ravel()
    assert (gradient@step).item() < 0
    assert counts['head_training_forwards'] == 3*41 and counts['head_hessian_passes'] == 1


def test_zero_weight_padding_cannot_affect_loss_gradient_or_hessian(tmp_path):
    arrays, metadata = data(tmp_path);batch = newton.pack_batches(arrays, metadata, 'cpu')[-1]
    delta = torch.ones(64, dtype=torch.float64)*.01
    first = newton.batch_statistics(batch, delta, hessian=True)
    batch = copy.deepcopy(batch);batch['h'][4:] = 1e8;batch['raw'][4:] = 1e10
    second = newton.batch_statistics(batch, delta, hessian=True)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_armijo_backtracks_to_a_real_decreasing_step():
    x = torch.tensor([2.], dtype=torch.float64);direction = torch.tensor([-8.], dtype=torch.float64)
    value = lambda v: .5*v.square().sum()
    result = newton.armijo_search(x, direction, value(x), torch.tensor(-16.), value)
    candidate, loss, scale, attempts = result
    assert candidate.item() == 0. and loss.item() == 0. and scale == .25 and attempts == 3
    assert x.item() == 2.


def test_failed_line_search_stops_after_twelve_and_preserves_iterate():
    x = torch.ones(2, dtype=torch.float64);before = x.clone();calls = []
    def reject(candidate):
        calls.append(candidate);return torch.tensor(2., dtype=torch.float64)
    assert newton.armijo_search(x, -x, torch.tensor(1.), torch.tensor(-1.), reject) is None
    assert len(calls) == 12 and torch.equal(x, before)


def test_non_descent_direction_does_not_start_line_search():
    def forbidden(candidate):pytest.fail('Non-descent trial evaluated')
    assert newton.armijo_search(torch.ones(2), torch.ones(2), torch.tensor(1.), torch.tensor(1.), forbidden) is None


@pytest.mark.parametrize('kind', ['full', 'hessian'])
def test_pass_budget_is_checked_before_any_new_batch(kind):
    counts = receipt();counts['head_full_objective_passes' if kind == 'full' else 'head_hessian_passes'] = 417 if kind == 'full' else 32
    before = counts.copy()
    with pytest.raises(RuntimeError, match='budget'):
        newton.evaluate([], torch.zeros(64), counts, lambda: pytest.fail('Budget overrun'), hessian=True)
    assert counts == before


def test_unavailable_cuda_refuses_before_creating_any_cpu_solver(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(newton, 'pack_batches', lambda *a: pytest.fail('CPU fallback'))
    with pytest.raises(RuntimeError, match='CUDA required'):
        newton.fit_newton(None, None, {}, None, None)


def test_validation_tag_inside_registered_training_rows_is_refused(tmp_path):
    _, metadata = data(tmp_path);metadata['states'][640]['split'] = 'validation'
    with pytest.raises(ValueError, match='Validation'):
        newton.training_rows(metadata)
