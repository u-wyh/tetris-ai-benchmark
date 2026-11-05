import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('gradient_audit',
    Path(__file__).resolve().parents[1]/'scripts/audit_ppo_gradients.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('entropy_coefficient', [0., .01])
def test_gradient_attribution_isolates_actor_and_critic_without_updates(entropy_coefficient):
    actor = torch.nn.Parameter(torch.tensor(2.))
    critic = torch.nn.Parameter(torch.tensor(3.))
    parameters = [actor, critic]
    pg = audit.gradient_vector(actor.square(), parameters)
    entropy = audit.gradient_vector(-actor, parameters)
    value = audit.gradient_vector(critic.square(), parameters)
    result = audit.gradient_summary(pg, entropy, value, torch.tensor([True, False]),
                                   entropy_coefficient, .5, .5)
    actor_gradient = 4 - entropy_coefficient
    total = (actor_gradient ** 2 + 3 ** 2) ** .5
    assert result['combined_global_norm'] == pytest.approx(total)
    assert result['actor_value_gradient_norm'] == result['critic_policy_gradient_norm'] == 0
    assert result['weighted_entropy_to_policy_norm_ratio'] == pytest.approx(entropy_coefficient/4)
    assert result['global_clip_multiplier'] == pytest.approx(.5/(total+1e-6))
    assert result['actor_policy_entropy_cosine'] == (-1 if entropy_coefficient else None)
    assert (actor.item(), critic.item()) == (2., 3.)
    assert actor.grad is None and critic.grad is None
