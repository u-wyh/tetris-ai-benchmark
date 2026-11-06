import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_actor_momentum import zero_actor_first_moment
from verify_feature_backend_benchmark import identical


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=torch.nn.Linear(2,1)
        self.value_net=torch.nn.Linear(2,1)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=.01,amsgrad=True)
        for p in self.parameters():p.grad=torch.ones_like(p)
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)


def test_only_actor_first_moments_change_and_weights_do_not():
    policy=Policy();weights=copy.deepcopy(policy.state_dict())
    before={name:copy.deepcopy(policy.optimizer.state[p]) for name,p in policy.named_parameters()}
    groups=copy.deepcopy(policy.optimizer.state_dict()['param_groups'])
    result=zero_actor_first_moment(policy)
    assert set(result['actor_parameters'])=={'actor.weight','actor.bias'}
    for name,p in policy.named_parameters():
        expected=before[name]
        if name.startswith('actor.'):
            expected['exp_avg'].zero_()
            assert torch.count_nonzero(before[name]['exp_avg_sq'])>0
        assert identical(expected,policy.optimizer.state[p])
    assert identical(weights,policy.state_dict())
    assert identical(groups,policy.optimizer.state_dict()['param_groups'])


def test_incomplete_later_state_fails_before_mutating_any_actor():
    policy=Policy();params=list(policy.parameters())
    del policy.optimizer.state[params[-1]]['exp_avg_sq']
    before=copy.deepcopy(policy.optimizer.state_dict())
    with pytest.raises(ValueError,match='Incomplete'):
        zero_actor_first_moment(policy)
    assert identical(before,policy.optimizer.state_dict())


def test_nonfinite_moments_are_refused():
    policy=Policy();p=next(policy.parameters())
    policy.optimizer.state[p]['exp_avg'].fill_(float('inf'))
    with pytest.raises(ValueError,match='Invalid'):
        zero_actor_first_moment(policy)


def test_other_optimizers_are_refused():
    policy=Policy();policy.optimizer=torch.optim.SGD(policy.parameters(),lr=.01)
    with pytest.raises(ValueError,match='Adam'):
        zero_actor_first_moment(policy)
