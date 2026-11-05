from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training.initialization import initialize_policy, load_source_model, validate_initial_optimizer
from training.train_transaction import global_rng_state
from training.train_ppo import ROOT


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


@pytest.mark.parametrize('failing',[False,True])
def test_source_loading_preserves_run_rng_even_if_loading_fails(failing):
    target=SimpleNamespace(seed=1700000,num_timesteps=0,_last_obs=None,_last_episode_starts=None)
    random.seed(1700000);np.random.seed(1700000);torch.manual_seed(1700000)
    expected=global_rng_state(target)
    class Loader:
        @staticmethod
        def load(path,device):
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            random.random();np.random.random();torch.rand(3)
            if failing:raise RuntimeError('injected load failure')
            return 'source'
    if failing:
        with pytest.raises(RuntimeError,match='injected'):load_source_model(Loader,'source.zip',target)
    else:assert load_source_model(Loader,'source.zip',target)=='source'
    actual=global_rng_state(target)
    assert actual['python_random']==expected['python_random']
    assert np.array_equal(actual['numpy_random'][1],expected['numpy_random'][1])
    assert actual['numpy_random'][2:]==expected['numpy_random'][2:]
    assert torch.equal(actual['torch_cpu_random'],expected['torch_cpu_random'])
    if actual['torch_cuda_random'] is not None:
        assert all(torch.equal(a,b) for a,b in zip(actual['torch_cuda_random'],expected['torch_cuda_random']))
    assert actual['run_seed']==1700000


def test_real_sb3_source_load_preserves_configured_rng():
    from sb3_contrib import MaskablePPO

    checkpoint=ROOT/'runs/ppo_candidate_raw_30m_seed42/final/model.zip'
    if not checkpoint.exists():
        pytest.skip('Frozen PPO checkpoint unavailable')
    target=SimpleNamespace(seed=1700000,num_timesteps=0,_last_obs=None,_last_episode_starts=None)
    random.seed(target.seed);np.random.seed(target.seed);torch.manual_seed(target.seed)
    expected=global_rng_state(target)
    source=load_source_model(MaskablePPO,str(checkpoint),target)
    actual=global_rng_state(target)
    assert source.seed==42 and target.seed==1700000
    assert actual['python_random']==expected['python_random']
    assert np.array_equal(actual['numpy_random'][1],expected['numpy_random'][1])
    assert actual['numpy_random'][2:]==expected['numpy_random'][2:]
    assert torch.equal(actual['torch_cpu_random'],expected['torch_cpu_random'])
    if expected['torch_cuda_random'] is not None:
        assert all(torch.equal(a,b) for a,b in zip(actual['torch_cuda_random'],expected['torch_cuda_random']))
