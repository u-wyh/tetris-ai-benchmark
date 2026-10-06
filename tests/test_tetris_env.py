"""Run with .venv/bin/python -m pytest tests/test_tetris_env.py -q."""

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from gymnasium.utils.env_checker import check_env

from training.env import TetrisEnv
from training.env.tetris_env import placement_reward
from training.tetris_core import encode_action
from training.tetris_core.pieces import make_piece


def setup_clear(env, rows, piece_type, gap):
    env.reset(seed=42)
    env.core.current = make_piece(piece_type)
    for y in range(20 - rows, 20):
        for x in range(10):
            if x not in gap:
                env.core.board[y][x] = "Z"


def test_spaces_and_observation_layout():
    env = TetrisEnv()
    observation, info = env.reset(seed=12345)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 1840
    assert observation.shape == (237,) and observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert observation[:200].sum() == 0
    current = env.core.current["type"]
    assert observation[200:207].tolist() == [float(t == current) for t in "IJLOSTZ"]
    assert observation[207:228].tolist() == [float(t == piece)
                                               for piece in env.core.queue[:3]
                                               for t in "IJLOSTZ"]
    assert observation[228:236].tolist() == [1.0] + [0.0] * 7
    assert observation[236] == 1.0
    assert info == {"score": 0, "lines": 0, "level": 1, "pieces": 0,
                    "cleared_lines": 0, "episode_seed": 12345, "game_over": False}
    # Score, level and lines are log-only fields, never extra observation features.
    before = observation.copy()
    env.core.score, env.core.level, env.core.lines = 999, 2, 10
    assert np.array_equal(before, env._encode_observation(env.core.get_public_observation()))
    env.core.board[19][0] = "T"
    board_observation = env._encode_observation(env.core.get_public_observation())
    assert board_observation[190] == 1.0 and board_observation[:200].sum() == 1.0


def test_hold_encoding_and_reset_state():
    env = TetrisEnv()
    initial, _ = env.reset(seed=42)
    held_type = env.core.current["type"]
    placement = next(item for item in env.core.get_legal_placements() if item["hold"] == 1)
    observation, _, terminated, _, _ = env.step(placement["actionId"])
    assert not terminated
    assert observation[228:236].tolist() == [0.0] + [float(t == held_type) for t in "IJLOSTZ"]
    assert observation[236] == 1.0  # The next spawned piece begins a new Hold cycle.
    env.core.hold_used = True
    assert env._encode_observation(env.core.get_public_observation())[236] == 0.0
    reset_observation, info = env.reset(seed=42)
    assert np.array_equal(initial, reset_observation)
    assert info["pieces"] == 0 and env.core.hold is None and env.episode_reward == 0.0


def test_action_mask_and_illegal_action():
    env = TetrisEnv()
    env.reset(seed=42)
    mask = env.action_masks()
    assert mask.shape == (1840,) and mask.dtype == np.bool_
    assert np.array_equal(mask, np.asarray(env.core.get_action_mask(), dtype=bool))
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError, match="not a legal placement"):
        env.step(illegal)
    with pytest.raises(ValueError, match="integer ID"):
        env.step(1.5)
    assert env.pieces == 0


@pytest.mark.parametrize("rows,piece_type,gap,action,expected_reward", [
    (1, "O", (3, 4), encode_action(0, 0, 3, 18), 1.001),
    (2, "O", (3, 4), encode_action(0, 0, 3, 18), 3.001),
    (3, "I", (4,), encode_action(0, 1, 4, 16), 5.001),
    (4, "I", (4,), encode_action(0, 1, 4, 16), 8.001),
])
def test_line_rewards(rows, piece_type, gap, action, expected_reward):
    env = TetrisEnv()
    setup_clear(env, rows, piece_type, gap)
    assert env.action_masks()[action]
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == pytest.approx(expected_reward)
    assert not terminated and not truncated
    assert info["cleared_lines"] == rows and info["lines"] == rows
    assert info["score"] == (0, 100, 300, 500, 800)[rows]


def test_no_clear_and_game_over_reward():
    env = TetrisEnv()
    env.reset(seed=1)
    action = int(np.flatnonzero(env.action_masks())[0])
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == pytest.approx(0.001)
    assert not terminated and not truncated and info["cleared_lines"] == 0
    env.reset(seed=1)
    env.core.current = make_piece("O")
    env.core.board[2][4] = "Z"
    action = encode_action(0, 0, 4, 0)
    assert env.action_masks()[action]
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == pytest.approx(-1.999)
    assert terminated and not truncated and info["game_over"]
    assert not env.action_masks().any()
    with pytest.raises(RuntimeError, match="finished"):
        env.step(action)


def test_truncation_does_not_add_game_over_penalty():
    env = TetrisEnv(max_pieces=1)
    env.reset(seed=1)
    action = int(np.flatnonzero(env.action_masks())[0])
    _, reward, terminated, truncated, info = env.step(action)
    assert not terminated and truncated and not info["game_over"]
    assert reward == pytest.approx(0.001)
    assert info["pieces"] == 1
    with pytest.raises(RuntimeError, match="finished"):
        env.step(action)
    with pytest.raises(ValueError, match="positive"):
        TetrisEnv(max_pieces=0)


def test_reward_is_separate_from_game_score():
    assert placement_reward(0, False) == pytest.approx(0.001)
    assert placement_reward(4, False) == pytest.approx(8.001)
    assert placement_reward(4, True) == pytest.approx(6.001)
    with pytest.raises(ValueError):
        placement_reward(5, False)


def test_fixed_seed_trajectory_repeats_exactly():
    first, second = TetrisEnv(), TetrisEnv()
    initial_a, info_a = first.reset(seed=12345)
    initial_b, info_b = second.reset(seed=12345)
    assert np.array_equal(initial_a, initial_b) and info_a == info_b
    for step in range(40):
        mask_a, mask_b = first.action_masks(), second.action_masks()
        assert np.array_equal(mask_a, mask_b)
        action = int(np.flatnonzero(mask_a)[(step * 37) % np.count_nonzero(mask_a)])
        result_a, result_b = first.step(action), second.step(action)
        assert np.array_equal(result_a[0], result_b[0])
        assert result_a[1:] == result_b[1:]
        if result_a[2] or result_a[3]:
            initial_a, info_a = first.reset(seed=12345 + step + 1)
            initial_b, info_b = second.reset(seed=12345 + step + 1)
            assert np.array_equal(initial_a, initial_b) and info_a == info_b


def test_gymnasium_checker_with_legal_action_sampler():
    env = TetrisEnv()
    # Gymnasium's checker samples Discrete uniformly and knows nothing about masks.
    # It samples before reset(seed=123), so resolve the sampled action when step runs.
    class LegalSample(int):
        def __new__(cls):
            return int.__new__(cls, 0)

        def __int__(self):
            return int(np.flatnonzero(env.action_masks())[0])

    env.action_space.sample = LegalSample
    check_env(env, skip_render_check=True)


def test_random_legal_smoke():
    env = TetrisEnv(max_pieces=250)
    rng = np.random.default_rng(17)
    env.reset(seed=17)
    for step in range(1000):
        mask = env.action_masks()
        assert mask.shape == (1840,) and mask.dtype == np.bool_ and mask.any()
        action = int(rng.choice(np.flatnonzero(mask)))
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert info["pieces"] >= 1
        if terminated or truncated:
            env.reset(seed=17 + step + 1)
