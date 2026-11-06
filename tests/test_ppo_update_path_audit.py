import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_ppo_update_path import ObservedBuffer, clone_policy, train_clone
from verify_feature_backend_benchmark import identical
from training.train_transaction import global_rng_state


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Linear(2,1)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=.01)
    def _get_constructor_parameters(self): return {}
    def evaluate_actions(self,*args,**kwargs): raise AssertionError('Unexpected evaluation')


def test_policy_clone_has_independent_weights_adam_and_preserves_rng():
    policy=Policy()
    for p in policy.parameters():p.grad=torch.ones_like(p)
    policy.optimizer.step();policy.optimizer.zero_grad(set_to_none=True)
    weights=copy.deepcopy(policy.state_dict());state=copy.deepcopy(policy.optimizer.state_dict())
    rng=torch.get_rng_state().clone()
    clone=clone_policy(policy,torch.device('cpu'))
    assert torch.equal(rng,torch.get_rng_state())
    assert identical(weights,clone.state_dict()) and identical(state,clone.optimizer.state_dict())
    for p in clone.parameters():p.grad=torch.ones_like(p)
    clone.optimizer.step()
    assert identical(weights,policy.state_dict()) and identical(state,policy.optimizer.state_dict())


def test_observed_buffer_preserves_underlying_order_and_arrays():
    class Buffer:
        values=np.array([1.,2.])
        def get(self,size):
            assert size==1
            yield 'first'
            yield 'second'
    source=Buffer();wrapper=ObservedBuffer(source)
    before=source.values.copy()
    assert list(wrapper.get(1))==['first','second']
    assert wrapper.last=='second' and wrapper.values is source.values
    assert np.array_equal(source.values,before)


def test_failed_clone_train_restores_source_rng_policy_and_optimizer(tmp_path):
    class Source:
        device=torch.device('cpu')
        num_timesteps=0
        seed=1
        _last_obs=None
        _last_episode_starts=None
        _n_updates=0
        env='unused'
        rollout_buffer=object()
        def __init__(self):self.policy=Policy()
        def set_logger(self,logger):self._logger=logger
        @property
        def logger(self):return self._logger
        def train(self):
            assert self.env is None
            random.random();np.random.random();torch.rand(2)
            for p in self.policy.parameters():p.grad=torch.ones_like(p)
            self.policy.optimizer.step()
            raise RuntimeError('Injected failure')
    source=Source();before_rng=global_rng_state(source)
    before=copy.deepcopy(source.policy.state_dict());optimizer=copy.deepcopy(source.policy.optimizer.state_dict())
    with pytest.raises(RuntimeError,match='Injected failure'):
        train_clone(source,.01,None,{},None,tmp_path,'failure',traced=False)
    assert identical(before_rng,global_rng_state(source))
    assert identical(before,source.policy.state_dict())
    assert identical(optimizer,source.policy.optimizer.state_dict())
