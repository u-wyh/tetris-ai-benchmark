"""Explicit policy and optimizer-state transfer for a new training transaction."""

from copy import deepcopy

import torch


def validate_initial_optimizer(preset):
    mode = preset.get('initial_optimizer', 'fresh')
    if mode not in ('fresh', 'restore_state'):
        raise ValueError('initial_optimizer must be fresh or restore_state')
    if mode == 'restore_state' and not preset.get('initial_model'):
        raise ValueError('Restoring optimizer state requires an initial_model')
    return mode


def initialize_policy(target, source, mode='fresh'):
    """Copy weights; optionally copy moments/step counters with new run options.

    This does not restore environment state, RNG, or PPO timesteps. It is a new
    transfer run; normal transaction resume restores the complete run instead.
    """
    if mode not in ('fresh', 'restore_state'):
        raise ValueError('Unknown optimizer initialization mode')
    source_parameters = [(name, tuple(p.shape)) for name,p in source.named_parameters()]
    if source_parameters != [(name, tuple(p.shape)) for name,p in target.named_parameters()]:
        raise ValueError('Source and target parameter order or shapes differ')
    target.load_state_dict(source.state_dict(), strict=True)
    if mode == 'restore_state':
        if type(source.optimizer) is not type(target.optimizer):
            raise ValueError('Source and target optimizers differ')
        state = deepcopy(source.optimizer.state_dict())
        if not state['state']:
            raise ValueError('Source checkpoint contains no optimizer state')
        for entry in state['state'].values():
            for value in entry.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('Source optimizer state is nonfinite')
        groups = target.optimizer.param_groups
        if [len(g['params']) for g in state['param_groups']] != [len(g['params']) for g in groups]:
            raise ValueError('Source and target optimizer groups differ')
        options = [{k:deepcopy(v) for k,v in g.items() if k != 'params'} for g in groups]
        target.optimizer.load_state_dict(state)
        for group, configured in zip(target.optimizer.param_groups, options):
            params = group['params']
            group.clear()
            group.update(params=params, **configured)
    steps = [float(s['step']) for s in target.optimizer.state.values() if 'step' in s]
    return dict(optimizer_mode=mode, optimizer_state_entries=len(target.optimizer.state),
                source_optimizer_state_entries=len(source.optimizer.state),
                minimum_optimizer_step=min(steps) if steps else None,
                maximum_optimizer_step=max(steps) if steps else None,
                learning_rates=[group['lr'] for group in target.optimizer.param_groups])
