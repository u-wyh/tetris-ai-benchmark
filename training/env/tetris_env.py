"""One Gymnasium step is one parity-tested TetrisCore legal placement."""

import copy
import math

import gymnasium as gym
import numpy as np

from training.tetris_core import ACTION_COUNT, TetrisCore

PIECE_TYPES = "IJLOSTZ"
LINE_REWARDS = (0.0, 1.0, 3.0, 5.0, 8.0)
RAW_REWARD_VERSION = "placement-reward-v1"
HOLE_REWARD_VERSION = "placement-reward-hole-v1"
HEIGHT_REWARD_VERSION = "placement-reward-height-v1"
HOLE_HEIGHT_REWARD_VERSION = "placement-reward-hole-height-v1"


def count_holes(board):
    """Count empty cells with an occupied cell above in the same column."""
    occupied_above = [False] * 10
    holes = 0
    for row in board:
        for x, cell in enumerate(row):
            if cell is not None:
                occupied_above[x] = True
            elif occupied_above[x]:
                holes += 1
    return holes


def shape_reward(raw_reward, holes_before, holes_after, coefficient):
    new_holes = max(0, holes_after - holes_before)
    penalty = coefficient * new_holes
    return raw_reward - penalty, new_holes, penalty


def height_risk(max_height):
    return max(0, max_height - 12) ** 2


def board_max_height(board):
    return max((20 - y for y, row in enumerate(board) if any(row)), default=0)


def placement_reward(cleared_lines, game_over):
    """RL reward is independent of the Benchmark game's score."""
    if not 0 <= cleared_lines <= 4:
        raise ValueError("A placement can clear zero through four lines")
    return 0.001 + LINE_REWARDS[cleared_lines] - (2.0 if game_over else 0.0)


class TetrisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_pieces=10000, hole_penalty_coef=0.0, height_penalty_coef=0.0):
        super().__init__()
        if type(max_pieces) is not int or max_pieces < 1:
            raise ValueError("max_pieces must be a positive integer")
        if (isinstance(hole_penalty_coef, bool) or not isinstance(hole_penalty_coef, (int, float))
                or not math.isfinite(hole_penalty_coef) or hole_penalty_coef < 0):
            raise ValueError("hole_penalty_coef must be a finite nonnegative number")
        self.max_pieces = max_pieces
        self.hole_penalty_coef = float(hole_penalty_coef)
        if (isinstance(height_penalty_coef, bool) or not isinstance(height_penalty_coef, (int, float))
                or not math.isfinite(height_penalty_coef) or height_penalty_coef < 0):
            raise ValueError("height_penalty_coef must be a finite nonnegative number")
        self.height_penalty_coef = float(height_penalty_coef)
        self.reward_version = (HOLE_HEIGHT_REWARD_VERSION if self.height_penalty_coef and self.hole_penalty_coef
                               else HEIGHT_REWARD_VERSION if self.height_penalty_coef
                               else HOLE_REWARD_VERSION if self.hole_penalty_coef
                               else RAW_REWARD_VERSION)
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(237,), dtype=np.float32)
        self.core = None
        self.episode_seed = None
        self.pieces = 0
        self.episode_reward = 0.0
        self.episode_count = 0
        self._finished = False

    @staticmethod
    def _encode_observation(public):
        result = np.zeros(237, dtype=np.float32)
        result[:200] = np.fromiter(
            (float(cell is not None) for row in public["board"] for cell in row),
            dtype=np.float32, count=200)
        current = public["currentPiece"]
        if current is not None:
            result[200 + PIECE_TYPES.index(current["type"])] = 1.0
        for index, piece_type in enumerate(public["next"]):
            result[207 + index * 7 + PIECE_TYPES.index(piece_type)] = 1.0
        hold = public["hold"]
        result[228 + (0 if hold is None else PIECE_TYPES.index(hold) + 1)] = 1.0
        result[236] = float(public["holdAvailable"])
        return result

    def _info(self, cleared_lines=0):
        return {"score": self.core.score, "lines": self.core.lines,
                "level": self.core.level, "pieces": self.pieces,
                "cleared_lines": cleared_lines, "episode_seed": self.episode_seed,
                "game_over": self.core.phase == "over"}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 1 << 32, dtype=np.uint32))
        self.episode_seed = seed
        self.episode_count += 1
        if self.core is None:
            self.core = TetrisCore(seed)
        else:
            self.core.reset(seed)
        self.pieces = 0
        self.episode_reward = 0.0
        self._finished = False
        return self._encode_observation(self.core.get_public_observation()), self._info()

    def get_observation(self):
        if self.core is None:
            raise RuntimeError("Call reset() before requesting an observation")
        return self._encode_observation(self.core.get_public_observation())

    def get_state(self):
        """Snapshot all trajectory-relevant state; TetrisCore owns board, bag and gameplay RNG."""
        if self.core is None:
            raise RuntimeError("Call reset() before snapshotting")
        return {"core": copy.deepcopy(self.core), "max_pieces": self.max_pieces,
                "reward_version": self.reward_version, "hole_penalty_coef": self.hole_penalty_coef,
                "height_penalty_coef": self.height_penalty_coef,
                "episode_seed": self.episode_seed, "episode_count": self.episode_count,
                "pieces": self.pieces, "episode_reward": self.episode_reward,
                "finished": self._finished,
                "numpy_generator": copy.deepcopy(self.np_random.bit_generator.state),
                "action_generator": copy.deepcopy(self.action_space.np_random.bit_generator.state),
                "numpy_seed": self._np_random_seed}

    def set_state(self, state):
        if state["max_pieces"] != self.max_pieces:
            raise ValueError("Checkpoint max_pieces differs from environment")
        if (state.get("reward_version", RAW_REWARD_VERSION) != self.reward_version
                or state.get("hole_penalty_coef", 0.0) != self.hole_penalty_coef
                or state.get("height_penalty_coef", 0.0) != self.height_penalty_coef):
            raise ValueError("Checkpoint reward configuration differs from environment")
        self.core = copy.deepcopy(state["core"])
        self.episode_seed = state["episode_seed"]
        self.episode_count = state["episode_count"]
        self.pieces = state["pieces"]
        self.episode_reward = state["episode_reward"]
        self._finished = state["finished"]
        self.np_random.bit_generator.state = copy.deepcopy(state["numpy_generator"])
        self.action_space.np_random.bit_generator.state = copy.deepcopy(state["action_generator"])
        self._np_random_seed = state["numpy_seed"]

    def action_masks(self):
        if self.core is None:
            raise RuntimeError("Call reset() before requesting action masks")
        return np.asarray(self.core.get_action_mask(), dtype=np.bool_)

    def step(self, action):
        if self.core is None:
            raise RuntimeError("Call reset() before step()")
        if self._finished:
            raise RuntimeError("Episode has finished; call reset() before step()")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer ID from 0 to 1839")
        action = int(action)
        if not self.action_space.contains(action) or not self.action_masks()[action]:
            raise ValueError(f"Action {action} is not a legal placement in this state")
        holes_before = count_holes(self.core.board)
        height_before = board_max_height(self.core.board)
        previous_lines = self.core.lines
        public = self.core.step(action)
        cleared_lines = self.core.lines - previous_lines
        terminated = self.core.phase == "over"
        self.pieces += 1
        truncated = self.pieces >= self.max_pieces and not terminated
        self._finished = terminated or truncated
        raw_reward = placement_reward(cleared_lines, terminated)
        holes_after = count_holes(self.core.board)
        height_after = board_max_height(self.core.board)
        reward, new_holes, hole_penalty = shape_reward(
            raw_reward, holes_before, holes_after, self.hole_penalty_coef)
        risk_increase = max(0, height_risk(height_after) - height_risk(height_before))
        height_penalty = self.height_penalty_coef * risk_increase
        reward -= height_penalty
        self.episode_reward += reward
        info = self._info(cleared_lines)
        info.update(raw_reward=raw_reward, shaped_reward=reward,
                    holes_before=holes_before, holes_after=holes_after,
                    new_holes=new_holes, hole_penalty=hole_penalty)
        info.update(height_before=height_before, height_after=height_after,
                    risk_increase=risk_increase, height_penalty=height_penalty)
        return self._encode_observation(public), reward, terminated, truncated, info
