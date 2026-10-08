"""Reward-only change: the game state and action policy stay identical."""

import copy
import json

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from training.env import TetrisEnv
from training.env.tetris_env import count_holes, placement_reward, shape_reward
from training.tetris_core import encode_action
from training.tetris_core.pieces import make_piece
from training.train_vector_transaction import hole_metrics_for_task, run, vector_config
from training.evaluation.compare_hole_v1 import paired_summary
from training.evaluation.evaluate import evaluate_model
from training.step0_baseline import compare_raw_initial_parameters


@pytest.mark.parametrize("before,after,expected_new,expected_penalty", [
    (0, 0, 0, 0.0), (2, 3, 1, 0.1), (2, 5, 3, 0.3), (5, 2, 0, 0.0),
])
def test_only_net_new_holes_are_penalized(before, after, expected_new, expected_penalty):
    shaped, new, penalty = shape_reward(1.001, before, after, 0.1)
    assert new == expected_new
    assert penalty == pytest.approx(expected_penalty)
    assert shaped == pytest.approx(1.001 - expected_penalty)


def test_hole_definition_counts_every_empty_cell_below_a_block():
    board = [[None] * 10 for _ in range(20)]
    board[0][2] = "T"
    board[3][2] = "T"
    for y in range(4, 20):
        board[y][2] = "T"
    assert count_holes(board) == 2
    board[1][2] = "I"
    assert count_holes(board) == 1


def test_shaping_uses_post_clear_real_board_and_preserves_core():
    raw, shaped = TetrisEnv(hole_penalty_coef=0), TetrisEnv(hole_penalty_coef=0.1)
    for env in (raw, shaped):
        env.reset(seed=42)
        env.core.current = make_piece("O")
        for x in range(10):
            if x not in (3, 4):
                env.core.board[19][x] = "Z"
    action = encode_action(0, 0, 3, 18)
    assert raw.action_masks()[action] and shaped.action_masks()[action]
    raw_obs, raw_reward, *_ = raw.step(action)
    shaped_obs, shaped_reward, terminated, truncated, info = shaped.step(action)
    assert not terminated and not truncated and info["cleared_lines"] == 1
    assert info["holes_before"] == 0
    assert info["holes_after"] == count_holes(shaped.core.board)
    assert info["new_holes"] == max(0, info["holes_after"] - info["holes_before"])
    assert info["raw_reward"] == pytest.approx(raw_reward)
    assert info["shaped_reward"] == pytest.approx(shaped_reward)
    assert shaped_reward == pytest.approx(raw_reward - 0.1 * info["new_holes"])
    assert np.array_equal(raw_obs, shaped_obs)
    assert raw.core.board == shaped.core.board
    assert raw.core.score == shaped.core.score == 100
    assert np.array_equal(raw.action_masks(), shaped.action_masks())


def test_game_over_penalty_is_preserved_and_invalid_action_rejected():
    env = TetrisEnv(hole_penalty_coef=0.1)
    env.reset(seed=1)
    env.core.current = make_piece("O")
    env.core.board[2][4] = "Z"
    action = encode_action(0, 0, 4, 0)
    assert env.action_masks()[action]
    _, reward, terminated, _, info = env.step(action)
    assert terminated and info["game_over"]
    assert info["raw_reward"] == placement_reward(info["cleared_lines"], True)
    assert reward == pytest.approx(info["raw_reward"] - info["hole_penalty"])
    env.reset(seed=1)
    invalid = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError, match="not a legal placement"):
        env.step(invalid)


def test_zero_coefficient_matches_raw_trajectory_and_checkpoint_rejects_change():
    first, second = TetrisEnv(), TetrisEnv(hole_penalty_coef=0.0)
    first.reset(seed=123)
    second.reset(seed=123)
    for _ in range(12):
        action = int(np.flatnonzero(first.action_masks())[0])
        a = first.step(action)
        b = second.step(action)
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]
        if a[2] or a[3]:
            break
    state = first.get_state()
    assert state["reward_version"] == "placement-reward-v1"
    second.set_state(copy.deepcopy(state))
    with pytest.raises(ValueError, match="reward configuration"):
        TetrisEnv(hole_penalty_coef=0.1).set_state(state)
    shaped = TetrisEnv(hole_penalty_coef=0.1)
    shaped.reset(seed=123)
    changed = shaped.get_state()
    changed["hole_penalty_coef"] = 0.2
    with pytest.raises(ValueError, match="reward configuration"):
        shaped.set_state(changed)


def test_shaped_training_config_isolated_from_raw():
    raw = vector_config()
    shaped = vector_config(target_steps=2_000_000, hole_penalty_coef=0.1)
    assert raw["reward_version"] == "placement-reward-v1"
    assert "hole_penalty_coef" not in raw
    assert shaped["reward_version"] == "placement-reward-hole-v1"
    assert shaped["hole_penalty_coef"] == 0.1
    assert shaped["actual_target_committed_steps"] == 2_002_944
    assert shaped["ppo"] == raw["ppo"]
    assert shaped["network"] == raw["network"]
    for invalid in (-1, float("nan"), True):
        with pytest.raises(ValueError, match="hole_penalty_coef"):
            TetrisEnv(hole_penalty_coef=invalid)


def test_task_reward_diagnostics_use_all_steps_without_partial_episodes():
    callback = type("Callback", (), dict(raw_reward_sum=3.0, shaped_reward_sum=2.7,
                                        new_holes_sum=3, hole_penalty_sum=0.3))()
    metrics = hole_metrics_for_task(callback, 10)
    assert metrics["mean_raw_reward"] == 0.3
    assert metrics["mean_shaped_reward"] == pytest.approx(0.27)
    assert metrics["mean_new_holes"] == 0.3
    assert metrics["new_holes_per_100_pieces"] == 30
    assert metrics["mean_hole_penalty"] == pytest.approx(0.03)


def test_paired_comparison_requires_matching_protocol():
    row = dict(seed=100000, pieces_survived=10, lines=2, score=300, new_holes_total=4)
    raw = dict(seeds=[100000], max_pieces=5000, per_seed=[row])
    shaped = dict(seeds=[100000], max_pieces=5000,
                  per_seed=[dict(row, pieces_survived=15, new_holes_total=2)])
    result = paired_summary(raw, shaped)
    assert result["pieces_survived"]["mean_difference"] == 5
    assert result["new_holes_total"]["mean_difference"] == -2
    with pytest.raises(ValueError, match="identical seeds"):
        paired_summary(raw, dict(shaped, seeds=[100001]))


def test_resume_rejects_changed_reward_coefficient_before_workers_start(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = vector_config(target_steps=4096, device="cpu", hole_penalty_coef=0.1)
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "transaction_state.json").write_text(json.dumps({"status": "running"}))
    (run_dir / "metadata.json").write_text(json.dumps({
        "device": "cpu", "worker_seeds": config["worker_seeds"],
        "reward_version": config["reward_version"], "hole_penalty_coef": 0.1}))
    preset = tmp_path / "changed.json"
    preset.write_text(json.dumps({"hole_penalty_coef": 0.2}))
    with pytest.raises(RuntimeError, match="Resume reward config differs"):
        run(run_dir, resume=True, config_file=preset)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    metadata["hole_penalty_coef"] = 0.2
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="config and metadata disagree"):
        run(run_dir, resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Formal Step-0 uses CUDA initialization")
def test_shaped_step0_matches_raw_cuda_parameters_exactly():
    torch.set_num_threads(1)
    config = vector_config(target_steps=2_000_000, hole_penalty_coef=0.1)
    ppo = config["ppo"]
    env = DummyVecEnv([lambda: TetrisEnv(hole_penalty_coef=0.1) for _ in range(8)])
    try:
        model = MaskablePPO(
            "MlpPolicy", env,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]},
                           "activation_fn": torch.nn.Tanh},
            learning_rate=ppo["learning_rate"], gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
            n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"], ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"],
            seed=42, device="cuda", verbose=0)
        assert compare_raw_initial_parameters(model, config)
    finally:
        env.close()


def test_evaluation_diagnostics_do_not_change_game_results(monkeypatch):
    class FirstLegal:
        def predict(self, observation, deterministic, action_masks):
            assert deterministic
            return int(np.flatnonzero(action_masks)[0]), None

    monkeypatch.setattr(MaskablePPO, "load", lambda *args, **kwargs: FirstLegal())
    plain = evaluate_model("unused.zip", [100000, 100001], 5, "test", 0)
    detailed = evaluate_model("unused.zip", [100000, 100001], 5, "test", 0,
                              diagnostics=True)
    for a, b in zip(plain["per_seed"], detailed["per_seed"]):
        assert all(b[key] == value for key, value in a.items())
        assert b["new_holes_total"] >= b["new_holes_events"]
    assert detailed["aggregate"]["new_holes_per_100_pieces"] >= 0
