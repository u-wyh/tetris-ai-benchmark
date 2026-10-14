"""Local imitation must preserve official masking and the frozen reference."""
import torch

import training.local_correction_data as data
from training.evaluation.hybrid_queue import load_models
from training.imitation_bc import batch
from training.train_local_correction import masked_kl
from training.env.fast_candidate_env import FastCandidateTetrisEnv


def test_masked_kl_and_legal_actions():
    models = load_models()
    env = FastCandidateTetrisEnv(max_pieces=100)
    try:
        observation, _ = env.reset(seed=730901)
        mask = env.action_masks()
        ids = mask.nonzero()[0]
        row = {"state": observation["state"].tolist(), "legal_ids": ids.tolist(),
               "legal_features": observation["candidate"][ids].tolist(),
               "action": int(models["ppo"].predict(observation, action_masks=mask,
                                                     deterministic=True)[0])}
        obs, legal, actions = batch([row], "cpu")
        policy = models["ppo"].policy
        assert torch.allclose(masked_kl(policy, policy, obs, legal), torch.zeros(1), atol=1e-6)
        distribution = policy.get_distribution(obs, action_masks=legal)
        assert legal[0, actions[0]]
        assert distribution.distribution.probs[~legal].sum() == 0
        logits = distribution.distribution.logits
        assert torch.isfinite(logits[legal]).all()
    finally:
        env.close()


def test_local_gradient_does_not_update_reference_or_critic():
    models = load_models()
    env = FastCandidateTetrisEnv(max_pieces=100)
    try:
        observation, _ = env.reset(seed=730902)
        mask = env.action_masks()
        ids = mask.nonzero()[0]
        row = {"state": observation["state"].tolist(), "legal_ids": ids.tolist(),
               "legal_features": observation["candidate"][ids].tolist(),
               "action": int(ids[-1])}
        obs, legal, actions = batch([row], "cpu")
        reference = models["ppo"].policy
        learner = models["bc"].policy
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        critic_before = {name: value.clone() for name, value in learner.value_net.state_dict().items()}
        loss = -learner.get_distribution(obs, action_masks=legal).log_prob(actions).mean()
        loss = loss + masked_kl(reference, learner, obs, legal).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.score_head.parameters())
        assert all(p.grad is None for p in reference.parameters())
        assert all(torch.equal(value, critic_before[name]) for name, value in learner.value_net.state_dict().items())
    finally:
        env.close()


def test_counterfactual_snapshot_is_reproducible(monkeypatch):
    models = load_models()
    monkeypatch.setattr(data, "HORIZON", 20)
    env = FastCandidateTetrisEnv(max_pieces=data.CAP)
    try:
        obs, _ = env.reset(seed=730903)
        for _ in range(3):
            action = int(models["ppo"].predict(obs, action_masks=env.action_masks(),
                                                deterministic=True)[0])
            obs, _, _, _, _ = env.step(action)
        state = env.get_state()
        action = int(models["ppo"].predict(obs, action_masks=env.action_masks(),
                                            deterministic=True)[0])
        first = data._branch(models, state, action)
        second = data._branch(models, state, action)
        assert first == second
        assert first["pieces"] == 20
    finally:
        env.close()


def test_sampler_keeps_every_candidate_until_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "RUN", tmp_path)
    monkeypatch.setattr(data, "CAP", 200)
    result = data.capture(730904, load_models())
    assert result["risky_candidates"] == result["risk_disagreements_seen"]
    assert result["normal_samples"] == result["normal_candidates_seen"]
