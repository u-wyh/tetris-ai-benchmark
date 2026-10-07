"""One Gymnasium step is one parity-tested TetrisCore legal placement."""

import copy

import gymnasium as gym
import numpy as np

from training.tetris_core import ACTION_COUNT, TetrisCore

PIECE_TYPES = "IJLOSTZ"
LINE_REWARDS = (0.0, 1.0, 3.0, 5.0, 8.0)


def placement_reward(cleared_lines, game_over):
    """RL reward is independent of the Benchmark game's score."""
    if not 0 <= cleared_lines <= 4:
        raise ValueError("A placement can clear zero through four lines")
    return 0.001 + LINE_REWARDS[cleared_lines] - (2.0 if game_over else 0.0)


class TetrisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_pieces=10000):
        super().__init__()
        if type(max_pieces) is not int or max_pieces < 1:
            raise ValueError("max_pieces must be a positive integer")
        self.max_pieces = max_pieces
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
                "episode_seed": self.episode_seed, "episode_count": self.episode_count,
                "pieces": self.pieces, "episode_reward": self.episode_reward,
                "finished": self._finished,
                "numpy_generator": copy.deepcopy(self.np_random.bit_generator.state),
                "action_generator": copy.deepcopy(self.action_space.np_random.bit_generator.state),
                "numpy_seed": self._np_random_seed}

    def set_state(self, state):
        if state["max_pieces"] != self.max_pieces:
            raise ValueError("Checkpoint max_pieces differs from environment")
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
        previous_lines = self.core.lines
        public = self.core.step(action)
        cleared_lines = self.core.lines - previous_lines
        terminated = self.core.phase == "over"
        self.pieces += 1
        truncated = self.pieces >= self.max_pieces and not terminated
        self._finished = terminated or truncated
        reward = placement_reward(cleared_lines, terminated)
        self.episode_reward += reward
        return self._encode_observation(public), reward, terminated, truncated, self._info(cleared_lines)
