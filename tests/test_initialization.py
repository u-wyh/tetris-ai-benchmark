from copy import deepcopy

import pytest
import torch
from torch import nn

from training.initialization import initialize_policy, validate_initial_optimizer


def policy(lr):
    p=nn.Linear(2,1)
    p.optimizer=torch.optim.Adam(p.parameters(),lr=lr,eps=1e-5)
    return p


def step(p):
    p.optimizer.zero_grad()
    p(torch.ones(1,2)).sum().backward()
    p.optimizer.step()


def test_restore_state_preserves_moments_steps_new_options_and_independence():
    source,target=policy(3e-4),policy(3e-5)
    step(source);step(source)
    original=deepcopy(source.optimizer.state_dict())
    result=initialize_policy(target,source,'restore_state')
    assert result['minimum_optimizer_step']==2
    assert result['optimizer_state_entries']==2
    assert result['learning_rates']==[3e-5]
    for a,b in zip(source.parameters(),target.parameters()):
        assert torch.equal(a,b)
        assert torch.equal(source.optimizer.state[a]['exp_avg'],target.optimizer.state[b]['exp_avg'])
    step(target)
    assert all(float(s['step'])==2 for s in source.optimizer.state.values())
    assert all(float(s['step'])==3 for s in target.optimizer.state.values())
    for key,entry in original['state'].items():
        assert torch.equal(entry['exp_avg'],source.optimizer.state_dict()['state'][key]['exp_avg'])


def test_default_transfer_leaves_optimizer_fresh():
    source,target=policy(3e-4),policy(3e-5);step(source)
    result=initialize_policy(target,source)
    assert not target.optimizer.state and result['optimizer_mode']=='fresh'
    assert result['minimum_optimizer_step'] is None


@pytest.mark.parametrize('preset',[{'initial_optimizer':'bad'}, {'initial_optimizer':'restore_state'}])
def test_invalid_preset_is_rejected(preset):
    with pytest.raises(ValueError):validate_initial_optimizer(preset)


def test_empty_or_nonfinite_optimizer_state_cannot_be_restored():
    source,target=policy(3e-4),policy(3e-5)
    with pytest.raises(ValueError,match='no optimizer state'):initialize_policy(target,source,'restore_state')
    step(source)
    next(iter(source.optimizer.state.values()))['exp_avg'].fill_(float('nan'))
    with pytest.raises(ValueError,match='nonfinite'):initialize_policy(target,source,'restore_state')
