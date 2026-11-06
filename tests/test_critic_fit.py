"""Critic updates preserve nonempty actor Adam moments; all gates are mandatory."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_critic_calibration import actor_snapshot, feasibility_gate, fit
from verify_feature_backend_benchmark import identical


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 1)
        self.value_net = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def predict_values(self, obs):
        return self.value_net(obs['state'])


def test_critic_updates_preserve_existing_actor_weights_and_moments():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    policy = ToyPolicy()
    states = torch.tensor([[1., 2.], [3., 4.], [2., 1.], [4., 3.]])
    # Populate nonzero actor moments, then leave its previous gradient in place.
    (policy.actor(states).square().mean()+policy.value_net(states).square().mean()).backward()
    policy.optimizer.step()
    before = actor_snapshot(policy)
    assert all(state['exp_avg'].abs().sum() > 0 for state in before['adam'].values())
    critic_before = policy.value_net.weight.detach().clone()
    count = fit(policy, states, torch.zeros(4), epochs=2, batch_size=2, seed=2800000,
                value_coef=.5, max_grad_norm=.5)
    assert count == 4 and identical(before, actor_snapshot(policy))
    assert not torch.equal(critic_before, policy.value_net.weight)
    assert all(p.grad is None for p in policy.parameters())


@pytest.mark.parametrize('failed_group', ['overall', 'terminal_window', 'outside_terminal_window', None])
def test_no_partial_feasibility_gate(failed_group):
    thresholds = dict(overall=.95, terminal_window=1., outside_terminal_window=1.05)
    baseline = {key: dict(samples=10, rmse=2.) for key in thresholds}
    fitted = {key: dict(samples=10, rmse=1.8) for key in thresholds}
    if failed_group:
        fitted[failed_group]['rmse'] = thresholds[failed_group]*2+.01
    assert feasibility_gate(baseline, fitted, thresholds) == (failed_group is None)
    fitted['terminal_window']['samples'] = 0
    assert not feasibility_gate(baseline, fitted, thresholds)
