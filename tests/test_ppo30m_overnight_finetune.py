"""Long episodes and PPO30M reference behavior for the isolated fine-tune."""
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from training.env.candidate_env import CandidateTetrisEnv
from training.env.tetris_env import HEIGHT_REWARD_VERSION, RAW_REWARD_VERSION
from training.kl_maskable_ppo import KLMaskablePPO, masked_policy_kl
from training.train_vector_transaction import vector_config
from training.train_ppo import ROOT
from training.evaluation.ppo30m_overnight_queue import _gate


PPO = ROOT / "runs/ppo_candidate_raw_30m_seed42/final/model.zip"


def test_100k_cap_state_restore_and_reward():
    first = CandidateTetrisEnv(max_pieces=100000, height_penalty_coef=.003)
    second = CandidateTetrisEnv(max_pieces=100000, height_penalty_coef=.003)
    try:
        first.reset(seed=880000)
        state = first.get_state()
        state["pieces"] = 99999
        first.set_state(state)
        second.reset(seed=1)
        second.set_state(first.get_state())
        action = int(np.flatnonzero(first.action_masks())[0])
        one = first.step(action)
        two = second.step(action)
        assert one[2:] == two[2:]
        assert one[2] is False and one[3] is True
        assert one[4]["game_over"] is False
        assert one[4]["pieces"] == 100000
        assert one[4]["height_penalty"] == pytest.approx(.003 * one[4]["risk_increase"])
        assert one[1] == pytest.approx(one[4]["raw_reward"] - one[4]["height_penalty"])
        assert one[4]["risk_increase"] >= 0
        assert first.reward_version == HEIGHT_REWARD_VERSION
        with pytest.raises(RuntimeError):
            first.step(action)
    finally:
        first.close(); second.close()


def test_vec_bootstrap_marks_truncation():
    env = DummyVecEnv([lambda: CandidateTetrisEnv(max_pieces=1)])
    try:
        env.reset()
        mask = env.env_method("action_masks")[0]
        _, _, done, infos = env.step(np.asarray([int(np.flatnonzero(mask)[0])]))
        assert done[0]
        assert infos[0]["TimeLimit.truncated"] is True
        assert "terminal_observation" in infos[0]
        assert infos[0]["game_over"] is False
    finally:
        env.close()


def test_isolated_cap_and_reward_config():
    old = vector_config(candidate_policy=True)
    new = vector_config(candidate_policy=True, max_pieces=100000, height_penalty_coef=.003)
    assert old["max_pieces"] == 10000 and old["reward_version"] == RAW_REWARD_VERSION
    assert new["max_pieces"] == 100000 and new["reward_version"] == HEIGHT_REWARD_VERSION
    assert new["height_penalty_coef"] == .003
    with pytest.raises(ValueError):
        vector_config(max_pieces=0)


def test_ppo30m_initialization_frozen_reference_and_masked_kl():
    if not PPO.exists(): pytest.skip("PPO30M checkpoint unavailable")
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: CandidateTetrisEnv(max_pieces=100000)])
    try:
        source = MaskablePPO.load(str(PPO), device="cpu")
        model = KLMaskablePPO(source.policy.__class__, env, n_steps=16,
                              batch_size=16, n_epochs=1, device="cpu", kl_beta=.03)
        fresh_optimizer = model.policy.optimizer
        model.policy.load_state_dict(source.policy.state_dict(), strict=True)
        model.set_reference(source)
        assert model.policy.optimizer is fresh_optimizer and not fresh_optimizer.state
        assert model.num_timesteps == 0
        obs = env.reset()
        mask = np.stack(env.env_method("action_masks"))
        observation, _ = model.policy.obs_to_tensor(obs)
        action_mask = torch.as_tensor(mask, dtype=torch.bool)
        kl = masked_policy_kl(model.policy, model.reference_policy, observation, action_mask)
        assert abs(kl.item()) < 1e-5
        assert all(not p.requires_grad for p in model.reference_policy.parameters())
        with torch.no_grad():
            model.policy.score_head[-1].weight[0, 0].add_(.1)
        new_kl = masked_policy_kl(model.policy, model.reference_policy, observation, action_mask)
        assert new_kl.item() >= -1e-5
        new_kl.backward()
        assert model.policy.score_head[-1].weight.grad is not None
        assert all(p.grad is None for p in model.reference_policy.parameters())
    finally:
        env.close()


def test_frozen_continue_and_confirm_gates():
    base = {"four_event_share": .93, "mean_pieces": 30000,
            "p10_pieces": 10000, "cap_rate": .2}
    candidate = {"four_event_share": .90, "mean_common_score_ratio": .95,
                 "mean_pieces": 31000, "p10_pieces": 11000, "cap_rate": .2}
    summary = {"variants":{"ppo":base,"raw":candidate}}
    assert _gate(summary,"raw","continue")
    assert _gate(summary,"raw","confirm")
    summary["variants"]["raw"]["four_event_share"] = .86
    assert not _gate(summary,"raw","continue")
    assert not _gate(summary,"raw","confirm")
