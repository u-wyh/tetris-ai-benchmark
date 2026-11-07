"""Describe sealed score-head constraints without another solve or policy call."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from freeze_head_representation import load_cache, unchanged
from solve_head_constraints import constraints, state_rows

QUANTILES = [0., .01, .1, .5, .9, .99, 1.]
NEAR_ZERO = 1e-12
RELATIVE_RANK = 1e-8


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(count=0, quantiles=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(count=int(values.size), quantiles=np.quantile(values, QUANTILES).tolist())


def spectrum(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError('A finite nonempty matrix is required')
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    relative_rank = int(np.sum(singular > RELATIVE_RANK * singular[0]))
    return dict(shape=list(matrix.shape), singular_values=singular.tolist(),
                machine_tolerance=float(tolerance), machine_rank=rank,
                relative_rank_tolerance=RELATIVE_RANK, relative_rank=relative_rank,
                full_column_condition=float(singular[0] / singular[-1])
                if rank == matrix.shape[1] else None), right[:rank]


def necessary_bounds(matrix, rhs):
    """Hölder gives ||d||1 >= rhs/||row||inf; it is not a feasible d."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if (matrix.ndim != 2 or not len(matrix) or rhs.shape != (len(matrix),)
            or not np.isfinite(matrix).all() or not np.isfinite(rhs).all()):
        raise ValueError('Invalid necessary-bound inputs')
    maximum = np.max(np.abs(matrix), axis=1)
    impossible = np.flatnonzero((maximum == 0) & (rhs > 0))
    bounds = np.divide(np.maximum(rhs, 0), maximum, out=np.zeros_like(rhs), where=maximum > 0)
    return dict(exact_zero_rows=int(np.sum(maximum == 0)),
                exact_zero_positive_rhs_rows=impossible.tolist(),
                finite_l1_lower_bound=float(np.max(bounds)),
                row_l1_lower_bounds=bounds.tolist(),
                scope='Necessary bound only; exact zero/positive RHS is a contradiction only for this fixed LP.')


def analyze(arrays, metadata):
    matrix, rhs, origin = constraints(arrays, metadata)
    normal_count = sum(row['kind'] == 'normal' for row in origin)
    normal = matrix[:normal_count]
    pairs = matrix[normal_count:]
    if len(pairs) != 4 or any(row['kind'] != 'preference' for row in origin[normal_count:]):
        raise ValueError('Expected four final preference rows')
    info, basis = spectrum(normal)
    lower = necessary_bounds(matrix, rhs)
    pairs_report = []
    weight = arrays['original_weight'].astype(np.float64)
    for i, row in enumerate(pairs):
        projected = (row @ basis.T) @ basis
        residual = float(np.linalg.norm(row - projected))
        length = float(np.linalg.norm(row))
        pairs_report.append(dict(state=origin[normal_count+i]['state'],
            original_margin=float(row @ weight), required_displacement_margin=float(rhs[normal_count+i]),
            l2_norm=length, max_abs=float(np.max(np.abs(row))),
            normal_row_span_projection_residual=residual,
            relative_projection_residual=residual / length if length else None,
            necessary_l1_lower_bound=lower['row_l1_lower_bounds'][normal_count+i]))
    normals = {}
    for split in ('train', 'validation'):
        margins = []; minimum = []; discrepancies = []
        for index, state in enumerate(metadata['states']):
            if state['kind'] != 'normal' or state['split'] != split:
                continue
            actions, h = state_rows(arrays, index)
            ref = int(np.flatnonzero(actions == state['reference_action'])[0])
            scores = h @ weight
            gap = scores[ref] - scores[actions != state['reference_action']]
            margins.extend(gap.tolist())
            minimum.append(float(np.min(gap)))
            if int(actions[np.argmax(scores)]) != state['reference_action']:
                discrepancies.append(index)
        margins = np.asarray(margins)
        normals[split] = dict(margins=distribution(margins), minimum_by_state=distribution(minimum),
            negative_margin_rows=int(np.sum(margins < 0)),
            margin_at_most_1e_minus3_rows=int(np.sum(margins <= 1e-3)),
            original_double_mode_mismatch_states=discrepancies)
    row_max = np.max(np.abs(normal), axis=1)
    h = arrays['h'].astype(np.float64)
    return dict(normal_training_spectrum=info, preferences=pairs_report, normal_by_split=normals,
        normal_row_max_abs=distribution(row_max), near_zero_normal_rows=int(np.sum(row_max <= NEAR_ZERO)),
        normal_column_l2=np.linalg.norm(normal, axis=0).tolist(),
        feature_abs=distribution(np.abs(h)), feature_std_by_dimension=np.std(h, axis=0).tolist(),
        tanh_saturated_entries=int(np.sum(np.abs(h) >= 1 - 1e-6)), feature_entries=int(h.size),
        necessary_bounds={k: v for k, v in lower.items() if k != 'row_l1_lower_bounds'},
        quantile_probabilities=QUANTILES, near_zero_threshold=NEAR_ZERO,
        scope='Descriptive fixed-representation algebra. Row span/rank does not establish cone feasibility, LP cause of timeout, new GPU predictions or Tetris quality.')


def run(folder):
    import torch
    from paired_branch_guard import require_authorization
    from resource_guard import now
    from training.evaluation.evaluate_resumable import durable_json
    design = json.loads((folder/'matrix_design.json').read_text())
    unchanged(design['frozen_files'])
    require_authorization(design, folder)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; preserve progress, no host or CPU model fallback')
    if (folder/'execution.json').exists():
        raise FileExistsError('Matrix audit already attempted; inspect before any retry')
    receipt = dict(at=now(), status='started', model_loads=0, gpu_forward_calls=0,
                   environment_steps=0, policy_optimizer_steps=0, lp_solver_calls=0)
    durable_json(folder/'execution.json', receipt)
    try:
        arrays, metadata = load_cache(Path(design['cache']), design['cache_metadata_sha256'], design['cache_arrays_sha256'])
        result = analyze(arrays, metadata)
        unchanged(design['frozen_files'])
        require_authorization(design, folder)
        result.update(at=now(), status='completed', eligible_for_training=False,
                      candidate_coefficients_generated=False, work={k: v for k, v in receipt.items() if k not in ('at', 'status')})
        durable_json(folder/'matrix_result.json', result)
        receipt.update(at=now(), status='completed')
    except BaseException as exc:
        receipt.update(at=now(), status='failed', error=repr(exc))
        raise
    finally:
        durable_json(folder/'execution.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
