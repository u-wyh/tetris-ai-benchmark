"""One bounded CUDA Newton solve of the sealed, scheduled CE head objective."""
from collections import Counter
import copy

import numpy as np

NEWTON = dict(max_steps=32, max_backtracks=12, armijo=1e-4, shrink=.5,
              gradient_infinity_tolerance=1e-6, reference_schedule_steps=256,
              batch_size=8, dtype='float64', l2_coefficient=.01)


def training_rows(metadata):
    from head_preference_probe import batch_indices
    counts = Counter(i for step in range(256) for i in batch_indices(metadata, step))
    pairs = metadata.get('training_pair_indices')
    if pairs != [0, 2, 4, 6, 640, 642, 644, 646]:
        raise ValueError('Only the registered eight-episode Newton comparison is supported')
    normal = sorted(i for i in counts if metadata['states'][i]['kind'] == 'normal')
    if (len(normal) != 316 or len(counts) != 324 or any(counts[i] != 128 for i in pairs) or
            any(metadata['states'][i]['split'] != 'train' for i in counts)):
        raise ValueError('Fixed scheduled empirical training weights differ')
    return [(i, counts[i]/1024.) for i in pairs+normal]


def pack_batches(arrays, metadata, device):
    """CUDA in production; CPU is used only by small algebra unit tests."""
    import torch
    rows = training_rows(metadata);batches = []
    for start in range(0, len(rows), 8):
        selected = rows[start:start+8]
        length = max(int(arrays['offsets'][i+1]-arrays['offsets'][i]) for i, _ in selected)
        h = np.zeros((8, length, 64), np.float64);raw = np.zeros((8, length), np.float64)
        mask = np.zeros((8, length), bool);mask[:, 0] = True
        weights = np.zeros(8, np.float64);pair = np.zeros(8, bool);good = np.zeros(8, np.int64)
        for lane, (i, weight) in enumerate(selected):
            left, right = arrays['offsets'][i:i+2];n = int(right-left);state = metadata['states'][i]
            h[lane, :n] = arrays['h'][left:right];raw[lane, :n] = arrays['raw_score'][left:right]
            mask[lane, :n] = True;weights[lane] = weight;pair[lane] = state['kind'] == 'pair'
            if pair[lane]:
                good[lane] = int(np.flatnonzero(arrays['actions'][left:right] == state['good_action'])[0])
        batch = {k: torch.as_tensor(v, device=device) for k, v in
                 dict(h=h, raw=raw, mask=mask, weights=weights, pair=pair, good=good).items()}
        batch['logq'] = torch.log_softmax(batch['raw'].masked_fill(~batch['mask'], -1e30), dim=1)
        batch['q'] = batch['logq'].exp();batches.append(batch)
    if len(batches) != 41:
        raise ValueError('Full training objective must contain exactly41 batches of8')
    return batches


def batch_statistics(batch, delta, derivatives=True, hessian=False):
    import torch
    h = batch['h'];weights = batch['weights']
    scores = batch['raw']+h@delta
    logp = torch.log_softmax(scores.masked_fill(~batch['mask'], -1e30), dim=1);p = logp.exp()
    ce = -logp[torch.arange(8, device=delta.device), batch['good']]
    kl = (batch['q']*(batch['logq']-logp)).sum(dim=1)
    loss = (weights*torch.where(batch['pair'], ce, kl)).sum()
    gradient = matrix = None
    if derivatives:
        residual = p-batch['q']
        residual[batch['pair']] = p[batch['pair']]
        pair_rows = torch.nonzero(batch['pair'], as_tuple=True)[0]
        residual[pair_rows, batch['good'][pair_rows]] -= 1.
        gradient = torch.einsum('nl,nld,n->d', residual, h, weights)
    if hessian:
        flat = h.reshape(-1, 64);probability_weights = (weights[:, None]*p).reshape(-1)
        mean = torch.einsum('nl,nld->nd', p, h)
        matrix = flat.T@(flat*probability_weights[:, None])-mean.T@(mean*weights[:, None])
    return loss, gradient, matrix


def evaluate(batches, delta, receipt, check, derivatives=True, hessian=False, line_search=False):
    import torch
    if receipt['head_full_objective_passes'] >= 417 or (hessian and receipt['head_hessian_passes'] >= 32):
        raise RuntimeError('Registered finite Newton pass budget exceeded')
    receipt['head_full_objective_passes'] += 1
    receipt['head_hessian_passes'] += int(hessian)
    receipt['line_search_objective_passes'] += int(line_search)
    loss = .01*delta.square().sum()
    gradient = .02*delta.clone() if derivatives else None
    matrix = .02*torch.eye(64, dtype=delta.dtype, device=delta.device) if hessian else None
    for batch in batches:
        check();receipt['head_training_forwards'] += 1
        value, grad, hess = batch_statistics(batch, delta, derivatives, hessian)
        loss = loss+value
        if derivatives:gradient = gradient+grad
        if hessian:matrix = matrix+hess
    if not torch.isfinite(loss).item() or any(x is not None and not torch.isfinite(x).all().item() for x in (gradient, matrix)):
        raise ValueError('Nonfinite Newton objective/derivatives')
    if hessian:matrix = (matrix+matrix.T)*.5
    return loss, gradient, matrix


def armijo_search(delta, direction, loss, directional_derivative, objective):
    """A fixed finite line search; a failed attempt never mutates the iterate."""
    import torch
    if not torch.isfinite(directional_derivative).item() or directional_derivative.item() >= 0:
        return None
    for backtrack in range(NEWTON['max_backtracks']):
        scale = NEWTON['shrink']**backtrack;candidate = delta+scale*direction
        value = objective(candidate)
        if torch.isfinite(value).item() and (value <= loss+NEWTON['armijo']*scale*directional_derivative).item():
            return candidate, value, scale, backtrack+1
    return None


class NewtonState:
    """Explicit solver metadata, with no Adam/PPO optimizer or inherited moments."""
    def __init__(self):
        self.state = dict(method='damped_newton_v1', accepted_steps=0, status='started', fixed=copy.deepcopy(NEWTON))

    def state_dict(self):
        return copy.deepcopy(self.state)


def fit_newton(arrays, metadata, receipt, check, checkpoint):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU Newton fallback')
    if metadata.get('preference_objective') != 'all_legal_cross_entropy':
        raise ValueError('Newton comparison requires the fixed all-legal CE objective')
    delta = torch.zeros(64, device='cuda', dtype=torch.float64)
    batches = pack_batches(arrays, metadata, 'cuda');state = NewtonState();history = []
    receipt.update(head_optimizer_initial_state_empty=True, head_full_objective_passes=0,
        head_hessian_passes=0, newton_linear_solve_attempts=0, line_search_objective_passes=0,
        head_fit_converged=False, newton_status='started', head_optimizer_steps=0)
    for accepted in range(NEWTON['max_steps']+1):
        loss, gradient, matrix = evaluate(batches, delta, receipt, check,
            hessian=accepted < NEWTON['max_steps'])
        gradient_inf = float(gradient.abs().max().item())
        state.state.update(accepted_steps=accepted, objective=float(loss.item()), gradient_infinity=gradient_inf,
                           gradient_at='checkpoint_iterate')
        if accepted == 0:
            history.append(dict(step=0, objective=float(loss.item()), gradient_infinity=gradient_inf))
            checkpoint(delta, state, 0)
        if gradient_inf <= NEWTON['gradient_infinity_tolerance']:
            receipt.update(head_fit_converged=True, newton_status='converged', newton_gradient_infinity=gradient_inf)
            state.state['status'] = 'converged';checkpoint(delta, state, accepted)
            break
        if accepted == NEWTON['max_steps']:
            receipt['newton_status'] = 'max_steps';break
        receipt['newton_linear_solve_attempts'] += 1
        factor, info = torch.linalg.cholesky_ex(matrix)
        if int(info.item()) != 0:
            receipt['newton_status'] = 'non_positive_definite_hessian';break
        direction = torch.cholesky_solve(-gradient[:, None], factor).reshape(-1)
        if not torch.isfinite(direction).all().item():
            receipt['newton_status'] = 'nonfinite_direction';break
        answer = armijo_search(delta, direction, loss, gradient@direction,
            lambda candidate: evaluate(batches, candidate, receipt, check,
                derivatives=False, line_search=True)[0])
        if answer is None:
            receipt['newton_status'] = 'line_search_failed';break
        delta, value, scale, attempts = answer
        receipt['head_optimizer_steps'] = accepted+1
        history.append(dict(step=accepted+1, objective=float(value.item()), scale=scale, backtracking_attempts=attempts))
        state.state.update(accepted_steps=accepted+1, objective=float(value.item()),
            gradient_at='previous_iterate', status='accepted_step')
        checkpoint(delta, state, accepted+1)
    receipt['newton_final_objective'] = float(loss.item())
    receipt['newton_gradient_infinity'] = gradient_inf
    if not receipt['head_fit_converged']:
        state.state['status'] = receipt['newton_status'];checkpoint(delta, state, receipt['head_optimizer_steps'])
    if (receipt['head_full_objective_passes'] > 417 or receipt['head_hessian_passes'] > 32 or
            receipt['newton_linear_solve_attempts'] > 32 or receipt['line_search_objective_passes'] > 384 or
            receipt['head_training_forwards'] != receipt['head_full_objective_passes']*41):
        raise ValueError('Actual bounded Newton work counts differ')
    return delta.detach().cpu().numpy(), history
