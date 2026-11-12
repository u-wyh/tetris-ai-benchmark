"""Audit a completed head fit and its exact scheduled empirical objective, without fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from head_preference_probe import (append_training_block, batch_indices, current_cases,
    load_cache, reused_extension, sha, unchanged, require_authorization)


def objective_gradient(arrays, metadata, delta, objective):
    """Exact256-minibatch frequency weights; no validation loss or optimizer."""
    counts = Counter(i for step in range(256) for i in batch_indices(metadata, step))
    parts = dict(preference=0., teacher_kl=0., l2=.01*float(delta@delta))
    gradient = .02*delta.copy()
    for index, count in counts.items():
        state = metadata['states'][index]
        if state['split'] != 'train':
            raise ValueError('Validation entered objective audit')
        start, stop = arrays['offsets'][index:index+2]
        h = arrays['h'][start:stop].astype(np.float64)
        raw = arrays['raw_score'][start:stop].astype(np.float64)
        scores = raw+h@delta
        logp = scores-np.max(scores);logp -= np.log(np.exp(logp).sum());p = np.exp(logp)
        weight = count/(256*4)
        if state['kind'] == 'pair':
            actions = arrays['actions'][start:stop]
            good = int(np.flatnonzero(actions == state['good_action'])[0])
            base = int(np.flatnonzero(actions == state['reference_action'])[0])
            if objective == 'all_legal_cross_entropy':
                value = -logp[good];grad = p@h-h[good]
            elif objective == 'pair_softplus':
                margin = scores[good]-scores[base]
                value = np.logaddexp(0., -margin)
                grad = np.exp(-np.logaddexp(0., margin))*(h[base]-h[good])
            else:
                raise ValueError('Unknown objective')
            parts['preference'] += weight*float(value)
        else:
            logq = raw-np.max(raw);logq -= np.log(np.exp(logq).sum());q = np.exp(logq)
            value = q@(logq-logp);grad = (p-q)@h
            parts['teacher_kl'] += weight*float(value)
        gradient += weight*grad
    if not np.isfinite(gradient).all() or not all(np.isfinite(v) for v in parts.values()):
        raise ValueError('Nonfinite objective audit')
    return dict(**parts, total=sum(parts.values()), gradient=gradient,
        preference_states=sum(metadata['states'][i]['kind'] == 'pair' for i in counts),
        normal_states=sum(metadata['states'][i]['kind'] == 'normal' for i in counts),
        exposures={str(i): n for i, n in sorted(counts.items())})


def run(folder):
    import torch
    from audit_head_direction import load_public
    from resource_guard import now
    from training.evaluation.evaluate_resumable import durable_json
    design = json.loads((folder/'audit_design.json').read_text())
    require_authorization(design, folder);unchanged(design['frozen_files'])
    probe = Path(design['probe_folder'])
    source_design = json.loads((probe/'probe_design.json').read_text())
    result = json.loads((probe/'probe_result.json').read_text())
    receipt = json.loads((probe/'execution.json').read_text())
    if (result['status'] != 'completed' or receipt['status'] != 'completed' or
            result['source_state_unchanged'] is not True):
        raise ValueError('Completed source required')
    if (receipt['head_optimizer_steps'] != 256 or result['eligible_for_ppo_resume'] is not False or
            sha(probe/'head_fit_checkpoint.pt') != result['checkpoint_sha256'] or
            sha(probe/'coefficients.npz') != result['coefficients_sha256']):
        raise ValueError('Completed endpoint seals/scope differ')
    # Serialized64-value head optimizer only, with external seals checked first.
    checkpoint = torch.load(probe/'head_fit_checkpoint.pt', map_location='cpu', weights_only=True)
    optimizer = checkpoint['optimizer'];group = optimizer['param_groups'][0]
    state = next(iter(optimizer['state'].values()))
    delta = checkpoint['delta'].numpy()
    if (checkpoint['head_optimizer_steps'] != 256 or checkpoint['source_ppo_counts'] != [30003200, 36625] or
            checkpoint['eligible_for_ppo_resume'] is not False or len(optimizer['state']) != 1 or
            len(optimizer['param_groups']) != 1 or len(group['params']) != 1 or int(state['step'].item()) != 256 or
            group['lr'] != .01 or group['betas'] != (.9, .999) or group['eps'] != 1e-8 or
            delta.shape != (64,) or delta.dtype != np.float64):
        raise ValueError('Serialized fresh head step/scope differs')
    with np.load(probe/'coefficients.npz', allow_pickle=False) as archive:
        if not np.array_equal(delta, archive['delta']):
            raise ValueError('Checkpoint/coefficient endpoint differs')
    arrays, metadata = load_cache(Path(source_design['cache']), source_design['cache_metadata_sha256'],
                                  source_design['cache_arrays_sha256'])
    public, states, _, _ = load_public(Path(source_design['public_folder']), source_design['frozen_files'])
    preferences = json.loads((Path(source_design['public_folder'])/'preferences.json').read_text())
    extended_design = dict(source_design, train_window=source_design['extension_window'], train_probe=source_design['extension_probe'])
    cases = current_cases(extended_design, preferences, public, states, training_block=1)['train']
    rows = reused_extension(source_design, cases, public, arrays['original_weight'], arrays['original_bias'])
    arrays, metadata = append_training_block(arrays, metadata, cases, rows)
    objective = source_design['preference_objective']
    before = objective_gradient(arrays, metadata, np.zeros(64), objective)
    endpoint = objective_gradient(arrays, metadata, delta, objective)
    gradient_norm = float(np.linalg.norm(endpoint.pop('gradient')));before.pop('gradient')
    answer = dict(at=now(), status='passed', serialized_head_adam_step=256, source_ppo_counts=[30003200, 36625],
        objective=objective, before=before, endpoint=endpoint, endpoint_gradient_l2=gradient_norm,
        ideal_exact_arithmetic_objective_gap_upper_bound=gradient_norm**2/.04,
        ideal_exact_arithmetic_distance_to_unique_minimizer_upper_bound=gradient_norm/.02,
        rationale='Fixed .01||d||^2 gives strong convexity .02: gap<=||gradient||^2/.04, distance<=||gradient||/.02.',
        work=dict(model_loads=0, policy_forwards=0, environment_steps=0, optimizer_steps=0),
        scope='Pure cached-array float64 diagnostic, not CPU policy training. Frequency-weighted exact scheduled empirical objective, not population loss. No validation gradients. Bounds assume exact arithmetic; float64 values are not interval-certified. No convergence/quality/promotion claim.')
    require_authorization(design, folder);unchanged(design['frozen_files'])
    durable_json(folder/'audit_result.json', answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
