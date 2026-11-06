import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_adam_updates import cloned_step, distribution_change, flat_parameters
from audit_ppo_gradients import snapshot_policy
from verify_feature_backend_benchmark import identical


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=torch.nn.Parameter(torch.tensor([1.,2.]))
        self.critic=torch.nn.Parameter(torch.tensor([3.]))
        self.optimizer=torch.optim.Adam(self.parameters(),lr=.01,eps=1e-8)


def warmed_policy():
    policy=ToyPolicy()
    for _ in range(5):
        policy.actor.grad=torch.tensor([.2,-.1])
        policy.critic.grad=torch.tensor([1.])
        policy.optimizer.step()
    policy.optimizer.zero_grad(set_to_none=True)
    return policy


def test_warm_adam_moves_even_with_zero_gradient_without_changing_source():
    source=warmed_policy();clone=ToyPolicy()
    weights=snapshot_policy(source);state=copy.deepcopy(source.optimizer.state_dict())
    before=flat_parameters(source)
    zero=cloned_step(clone,weights,state,torch.tensor([.2,-.1,100.]),{'actor'},.5,'restored_zero_global')
    assert (zero-before).norm()>0
    assert identical(weights,source.state_dict())
    assert identical(state,source.optimizer.state_dict())
    assert all(float(s['step'].item())==6 for s in clone.optimizer.state.values())
    first=cloned_step(clone,weights,state,torch.tensor([.2,-.1,100.]),{'actor'},.5,'restored_global')
    second=cloned_step(clone,weights,state,torch.tensor([.2,-.1,100.]),{'actor'},.5,'restored_global')
    assert torch.equal(first,second)  # Independent resets, never sequential updates.


def test_fresh_adam_step_is_not_proportional_to_global_clip_scale():
    source=ToyPolicy();weights=snapshot_policy(source);state=copy.deepcopy(source.optimizer.state_dict())
    clone=ToyPolicy();gradient=torch.tensor([1.,-1.,100.])
    globally=cloned_step(clone,weights,state,gradient,{'actor'},.5,'fresh_global')
    # Independent clipping changes actor gradient by ~70x, but fresh Adam normalizes it.
    separate=cloned_step(clone,weights,state,gradient,{'actor'},.5,'restored_separate')
    before=flat_parameters(source)
    ratio=(separate[:2]-before[:2]).norm()/(globally[:2]-before[:2]).norm()
    assert ratio.item()==pytest.approx(1.,abs=1e-4)
    assert all(float(s['step'].item())==1 for s in clone.optimizer.state.values())


def test_masked_distribution_comparison_handles_zero_probabilities():
    before=torch.tensor([[.5,.5,0.],[1.,0.,0.]])
    same=distribution_change(before,before)
    assert same==dict(mean_forward_kl=0.,mean_total_variation=0.,greedy_agreement=1.)
    changed=distribution_change(before,torch.tensor([[.25,.75,0.],[1.,0.,0.]]))
    assert changed['mean_forward_kl']>0
    assert changed['mean_total_variation']==pytest.approx(.125)
    assert changed['greedy_agreement']==.5
