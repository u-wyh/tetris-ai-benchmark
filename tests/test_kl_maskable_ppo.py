"""KL regularizer uses legal actions and updates only the current actor."""
import torch
from sb3_contrib import MaskablePPO

from training.imitation_bc import batch, records
from training.kl_maskable_ppo import KLMaskablePPO, masked_policy_kl
from training.train_ppo import ROOT


def test_masked_kl_gradient_and_save_load(tmp_path):
    torch.set_num_threads(1)
    source = ROOT / "runs/imitation_v2_hold_20m/bc/epoch_03.zip"
    if not source.exists():
        import pytest
        pytest.skip("BC checkpoint unavailable")
    row = next(records(ROOT / "runs/imitation_v2_hold_20m/data", [600100], 1))
    obs, mask, _ = batch([row], "cpu")
    reference = MaskablePPO.load(str(source), device="cpu")
    model = KLMaskablePPO.load(str(source), device="cpu")
    model.set_reference(reference)
    before = masked_policy_kl(model.policy, model.reference_policy, obs, mask)
    assert abs(before.item()) < 1e-6
    assert all(not p.requires_grad for p in reference.policy.parameters())
    with torch.no_grad():
        model.policy.score_head[-1].bias.add_(0.1)
        model.policy.score_head[-1].weight[0, 0].add_(0.1)
    kl = masked_policy_kl(model.policy, model.reference_policy, obs, mask)
    assert kl.item() >= -1e-6
    kl.backward()
    assert model.policy.score_head[-1].weight.grad is not None
    assert model.policy.score_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in reference.policy.parameters())
    model.save(str(tmp_path / "model.zip"))
    loaded = KLMaskablePPO.load(str(tmp_path / "model.zip"), device="cpu")
    assert loaded.reference_policy is None
    loaded.set_reference(reference)
    assert torch.allclose(kl, masked_policy_kl(loaded.policy, loaded.reference_policy, obs, mask))


def test_kl_ppo_one_rollout_updates_actor_and_critic():
    from training.candidate_policy import CandidateScoringPolicy
    from training.env.candidate_env import CandidateTetrisEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: CandidateTetrisEnv()])
    reference = MaskablePPO(CandidateScoringPolicy, env, n_steps=16, batch_size=16,
                            n_epochs=1, device="cpu", seed=19)
    model = KLMaskablePPO(CandidateScoringPolicy, env, n_steps=16, batch_size=16,
                          n_epochs=1, kl_beta=0.02, device="cpu", seed=20)
    model.set_reference(reference)
    ref_before = {k: v.clone() for k, v in reference.policy.state_dict().items()}
    actor_before = model.policy.score_head[-1].weight.detach().clone()
    critic_before = model.policy.value_net[-1].weight.detach().clone()
    model.learn(16, use_masking=True)
    assert not torch.equal(actor_before, model.policy.score_head[-1].weight)
    assert not torch.equal(critic_before, model.policy.value_net[-1].weight)
    assert all(torch.equal(v, ref_before[k]) for k, v in reference.policy.state_dict().items())
    env.close()
