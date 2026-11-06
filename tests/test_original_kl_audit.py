import copy
import hashlib
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_original_kl import FixedBuffer, load_data, prepare_clone, validate_actions
from audit_ppo_update_path import clone_policy
from training.candidate_policy import CandidateScoringPolicy
from training.kl_maskable_ppo import KLMaskablePPO, masked_policy_kl
from verify_feature_backend_benchmark import identical


class ShapeOnlyEnv(gym.Env):
    observation_space = gym.spaces.Dict(dict(
        state=gym.spaces.Box(-100, 100, (237,), np.float32),
        candidate=gym.spaces.Box(0, 255, (1840, 16), np.uint8)))
    action_space = gym.spaces.Discrete(1840)

    def reset(self, **kwargs):
        raise AssertionError('Offline tests must not collect a rollout')

    def step(self, action):
        raise AssertionError('Offline tests must not step an environment')


def tiny_case(tmp_path):
    torch.set_num_threads(1)
    source = MaskablePPO(CandidateScoringPolicy, ShapeOnlyEnv(), n_steps=4,
        batch_size=2, n_epochs=2, seed=1907, device='cpu')
    # Nonempty, nonzero Adam history for every parameter, including critic.
    source.policy.optimizer.zero_grad()
    sum(p.square().sum() for p in source.policy.parameters()).backward()
    source.policy.optimizer.step()
    source.policy.optimizer.zero_grad(set_to_none=True)
    reference = copy.copy(source)
    reference.policy = clone_policy(source.policy, source.device)
    rng = np.random.RandomState(1908)
    masks = np.zeros((4, 1840), np.float32)
    masks[:, :3] = 1
    flat = dict(obs_state=rng.normal(size=(4, 237)).astype(np.float32),
        obs_candidate=rng.randint(0, 10, size=(4, 1840, 16), dtype=np.uint8),
        actions=np.array([[0], [1], [2], [0]], np.int64), action_masks=masks,
        values=np.zeros((4, 1), np.float32), log_probs=np.zeros((4, 1), np.float32),
        returns=np.array([[1], [-2], [3], [4]], np.float32),
        advantages=np.array([[1], [-2], [3], [4]], np.float32))
    with torch.no_grad():
        obs = dict(state=torch.from_numpy(flat['obs_state']), candidate=torch.from_numpy(flat['obs_candidate']))
        values, logs, _ = source.policy.evaluate_actions(obs,
            torch.from_numpy(flat['actions']).flatten(), action_masks=torch.from_numpy(masks))
    flat['values'] = values.numpy()
    flat['log_probs'] = logs.numpy().reshape(-1, 1)
    permutations = [np.array([3, 1, 0, 2]), np.array([2, 0, 3, 1])]
    return source, reference, flat, permutations


def test_zero_beta_actual_sb3_train_matches_policy_adam_and_epochs(tmp_path):
    source, reference, flat, permutations = tiny_case(tmp_path)
    source_before = copy.deepcopy((source.policy.state_dict(), source.policy.optimizer.state_dict()))
    ref_before = copy.deepcopy((reference.policy.state_dict(), reference.policy.optimizer.state_dict()))
    data_before = copy.deepcopy(flat)
    left = prepare_clone(source, flat, permutations, reference, 3e-5, 0., configure(str(tmp_path/'left'), []))
    right = prepare_clone(source, flat, permutations, reference, 3e-5, 0., configure(str(tmp_path/'right'), []))
    try:
        MaskablePPO.train(left)
        KLMaskablePPO.train(right)
        assert identical(left.policy.state_dict(), right.policy.state_dict())
        assert identical(left.policy.optimizer.state_dict(), right.policy.optimizer.state_dict())
        assert left._n_updates == right._n_updates == source._n_updates+2
        assert left.rollout_buffer.epochs == right.rollout_buffer.epochs == 2
        assert identical(source_before, (source.policy.state_dict(), source.policy.optimizer.state_dict()))
        assert identical(ref_before, (reference.policy.state_dict(), reference.policy.optimizer.state_dict()))
        assert identical(data_before, flat)
        assert all(p.grad is None and not p.requires_grad for p in reference.policy.parameters())
        assert all(int(s['step']) == 5 for s in right.policy.optimizer.state.values())
    finally:
        left.logger.close(); right.logger.close(); source.env.close()


def test_anchor_gradient_reaches_current_actor_only(tmp_path):
    source, reference, flat, permutations = tiny_case(tmp_path)
    model = prepare_clone(source, flat, permutations, reference, 3e-5, .02, configure(str(tmp_path/'g'), []))
    try:
        batch = next(FixedBuffer(flat, permutations, 'cpu').get(2))
        with torch.no_grad():
            model.policy.candidate_encoder[0].weight[0, 0].add_(.1)
        loss = masked_policy_kl(model.policy, reference.policy, batch.observations, batch.action_masks)
        assert loss >= -1e-6
        model.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert model.policy.candidate_encoder[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for name, p in model.policy.named_parameters() if name.startswith('value_net.'))
        assert all(p.grad is None and not p.requires_grad for p in reference.policy.parameters())
    finally:
        model.logger.close(); source.env.close()


@pytest.mark.parametrize('masks,actions', [
    ([[0, 0]], [0]), ([[1, .5]], [0]), ([[1, float('nan')]], [0]),
    ([[1, 0]], [1]), ([[1, 0]], [2]), ([[1, 0]], [.5])])
def test_bad_masks_or_actions_fail_closed(masks, actions):
    with pytest.raises(ValueError):
        validate_actions(np.array(masks), np.array(actions))


def test_wrong_archive_rejected_before_numpy_load(tmp_path, monkeypatch):
    (tmp_path/'dataset.json').write_text('{}')
    (tmp_path/'rollout.npz').write_bytes(b'changed')
    def forbidden(*args, **kwargs):
        raise AssertionError('Untrusted archive reached numpy')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='before loading'):
        load_data(tmp_path, hashlib.sha256(b'{}').hexdigest(), 'wrong-sha')
