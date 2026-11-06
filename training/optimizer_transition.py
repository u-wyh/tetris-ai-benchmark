"""Explicit, one-time optimizer transitions after a full source-context restore."""


def zero_actor_first_moment(policy):
    import torch
    if not isinstance(policy.optimizer, torch.optim.Adam):
        raise ValueError('Selective transition requires Adam')
    actors = {id(p):name for name,p in policy.named_parameters() if not name.startswith('value_net.')}
    critics = {id(p) for name,p in policy.named_parameters() if name.startswith('value_net.')}
    if not actors or not critics:
        raise ValueError('Expected distinct candidate actor and value_net parameters')
    targets, known = [], set()
    for group in policy.optimizer.param_groups:
        for p in group['params']:
            if id(p) in known or id(p) not in actors.keys() | critics:
                raise ValueError('Unexpected or duplicate optimizer parameter')
            known.add(id(p))
            state = policy.optimizer.state.get(p, {})
            if not {'exp_avg','exp_avg_sq','step'} <= state.keys():
                raise ValueError('Incomplete Adam state')
            if (state['exp_avg'].shape != p.shape or state['exp_avg_sq'].shape != p.shape
                    or not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all()):
                raise ValueError('Invalid Adam moments')
            if id(p) in actors:
                targets.append(p)
    if known != actors.keys() | critics:
        raise ValueError('Optimizer is missing policy parameters')
    with torch.no_grad():
        for p in targets:
            policy.optimizer.state[p]['exp_avg'].zero_()
    return dict(transition='zero_actor_exp_avg_only',actor_parameters=[actors[id(p)] for p in targets],
                second_moments_steps_groups_and_critic_unchanged=True)
