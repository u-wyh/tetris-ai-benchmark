"""Public feature parity, fixed action alignment and masked policy behavior."""

import copy

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import (FEATURE_NAMES, FEATURE_SCALE, board_features,
                                         candidate_features)
from training.candidate_policy import CandidateScoringPolicy
from training.env.candidate_env import CandidateTetrisEnv
from training.tetris_core import ACTION_COUNT, TetrisCore
from training.tetris_core.pieces import make_piece
from training.train_vector_transaction import (observation_and_mask_digests,
                                                observations_equal, vector_config)


def test_all_action_rows_and_exact_uint8_ranges_match_core():
    core = TetrisCore(42)
    for _ in range(4):
        public = core.get_public_observation()
        placements = core.get_legal_placements()
        features, mask = candidate_features(public, placements)
        assert features.shape == (1840, 16) and features.dtype == np.uint8
        assert np.array_equal(mask, core.get_action_mask())
        assert np.count_nonzero(mask) == len(placements)
        assert not features[~mask].any()
        assert np.all(features[mask] <= FEATURE_SCALE)
        for placement in placements:
            action = placement["actionId"]
            simulated = copy.deepcopy(core)
            simulated.step(action)
            after = board_features(simulated.board)
            values = features[action]
            assert values[5] == simulated.lines - core.lines
            assert tuple(values[6:11]) == after
            assert values[11] == max(0, after[0] - board_features(core.board)[0])
            assert values[1] == placement["hold"]
            assert values[2] == placement["rotation"]
            assert values[3] == placement["x"]
            assert values[4] == placement["y"] + 3
        action = placements[0]["actionId"]
        core.step(action)
        if core.phase == "over":
            break


def test_hidden_queue_bag_rng_cannot_change_candidate_features():
    core = TetrisCore(12345)
    public = core.get_public_observation()
    first = candidate_features(public)
    core.queue[3:] = list(reversed(core.queue[3:]))
    core.bag.reverse()
    core.rng.random()
    second = candidate_features(core.get_public_observation())
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize("fixture", ["single_clear", "lock_out", "partial_lock"])
def test_candidate_features_match_core_at_scoring_and_topout_edges(fixture):
    core = TetrisCore(42)
    if fixture == "single_clear":
        core.current = make_piece("O")
        for x in range(10):
            if x not in (3, 4):
                core.board[19][x] = "Z"
    elif fixture == "lock_out":
        core.current = make_piece("I")
        core.current["y"] = -2
        for x in range(3, 7):
            core.board[0][x] = "Z"
    else:
        core.current = make_piece("O")
        core.current["x"], core.current["y"] = 0, -1
        core.board[1][0] = "Z"
    public = core.get_public_observation()
    features, mask = candidate_features(public)
    placements = {p["actionId"]: p for p in core.get_legal_placements()}
    for action in np.flatnonzero(mask):
        future = copy.deepcopy(core)
        future.step(int(action))
        assert tuple(features[action, 6:11]) == board_features(future.board)
        assert features[action, 5] == future.lines - core.lines
        assert bool(features[action, 15]) == (future.phase == "over" and
            future.board == core.board and all(cell["y"] < 0
                                               for cell in placements[action]["occupiedCells"]))


def test_dict_observation_and_mask_digest_changes_with_candidate_row():
    env = CandidateTetrisEnv()
    env.reset(seed=42)

    class OneWorker:
        def env_method(self, method):
            return [getattr(env, method)()]

    first, digests = observation_and_mask_digests(OneWorker())
    second, same = observation_and_mask_digests(OneWorker())
    assert observations_equal(first, second) and digests == same
    modified = {key: value.copy() for key, value in first.items()}
    modified["candidate"][0, int(np.flatnonzero(env.action_masks())[0]), 0] ^= 1
    assert not observations_equal(first, modified)


def test_candidate_env_cache_restore_and_raw_reward():
    env = CandidateTetrisEnv()
    obs, _ = env.reset(seed=42)
    before = env.get_state()
    mask = env.action_masks()
    assert env.observation_space.contains(obs)
    assert mask.shape == (ACTION_COUNT,)
    obs["candidate"][:] = 0
    assert env.get_observation()["candidate"].any()
    clone = CandidateTetrisEnv()
    clone.set_state(before)
    assert before["candidate_feature_version"] == env.feature_version
    wrong = dict(before, candidate_feature_version="unknown")
    with pytest.raises(ValueError, match="candidate feature version"):
        clone.set_state(wrong)
    assert observations_equal(env.get_observation(), clone.get_observation())
    assert np.array_equal(mask, clone.action_masks())
    action = int(np.flatnonzero(mask)[0])
    output = env.step(action)
    assert output[1] == output[4]["raw_reward"]
    assert env.observation_space.contains(output[0])


def test_policy_mask_sampling_training_prediction_and_save_load(tmp_path):
    torch.set_num_threads(1)
    env = CandidateTetrisEnv()
    obs, _ = env.reset(seed=42)
    mask = env.action_masks()
    model = MaskablePPO(CandidateScoringPolicy, env, n_steps=8, batch_size=8,
                        n_epochs=1, seed=42, device="cpu", verbose=0)
    assert model.rollout_buffer.observations["candidate"].dtype == np.uint8
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        actions, values, log_probs = model.policy(tensor, action_masks=mask[None])
        checked_values, checked_log_probs, entropy = model.policy.evaluate_actions(
            tensor, actions.flatten(), action_masks=torch.as_tensor(mask[None]))
    assert mask[int(actions[0])]
    assert torch.equal(values, checked_values)
    assert torch.allclose(log_probs, checked_log_probs)
    assert torch.isfinite(entropy).all()
    with pytest.raises(ValueError, match="without legal actions"):
        model.policy(tensor, action_masks=np.zeros((1, ACTION_COUNT), dtype=bool))
    selected, _ = model.predict(obs, deterministic=True, action_masks=mask)
    assert mask[int(selected)]
    path = tmp_path / "candidate.zip"
    model.save(path)
    restored = MaskablePPO.load(path, env=env, device="cpu")
    other, _ = restored.predict(obs, deterministic=True, action_masks=mask)
    assert int(selected) == int(other)
    assert model.policy.state_dict().keys() == restored.policy.state_dict().keys()
    assert all(torch.equal(a, restored.policy.state_dict()[key])
               for key, a in model.policy.state_dict().items())


def test_candidate_config_keeps_raw_training_hyperparameters():
    raw = vector_config(target_steps=2_000_000)
    candidate = vector_config(target_steps=2_000_000, candidate_policy=True)
    assert candidate["reward"] == raw["reward"]
    assert candidate["ppo"] == raw["ppo"]
    assert candidate["worker_seeds"] == raw["worker_seeds"]
    assert candidate["actual_target_committed_steps"] == 2_002_944
    assert candidate["candidate_feature_version"]


def test_seed42_candidate_initialization_reproduces_parameters():
    torch.set_num_threads(1)
    first = MaskablePPO(CandidateScoringPolicy, CandidateTetrisEnv(), n_steps=8,
                        batch_size=8, seed=42, device="cpu")
    second = MaskablePPO(CandidateScoringPolicy, CandidateTetrisEnv(), n_steps=8,
                         batch_size=8, seed=42, device="cpu")
    assert first.num_timesteps == second.num_timesteps == 0
    assert all(torch.equal(value, second.policy.state_dict()[key])
               for key, value in first.policy.state_dict().items())
