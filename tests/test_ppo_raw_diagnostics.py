"""Focused checks for read-only PPO-Raw analysis and its evaluation protocol."""

import json
import tempfile
import unittest
from pathlib import Path

from analysis.ppo_raw_diagnostics import RUN, board_features, quantile
from training.evaluation.evaluate import aggregate
from training.train_ppo import atomic_json


class DiagnosticsTests(unittest.TestCase):
    def test_board_features_and_quantile(self):
        board = [[None] * 10 for _ in range(20)]
        board[17][0] = "I"
        board[19][0] = "I"
        result = board_features(board)
        self.assertEqual(result["max_height"], 3)
        self.assertEqual(result["aggregate_height"], 3)
        self.assertEqual(result["holes"], 1)
        self.assertEqual(result["covered_holes"], 1)
        self.assertEqual(quantile([1, 2, 3, 4], .25), 1.75)

    @unittest.skipUnless((RUN / "best/model.zip").is_file(), "trained run unavailable")
    def test_checkpoint_mask_determinism_and_atomic_result(self):
        from sb3_contrib import MaskablePPO
        from training.env import TetrisEnv

        model = MaskablePPO.load(str(RUN / "best/model.zip"), device="cpu")
        env = TetrisEnv(max_pieces=50000)
        try:
            actions = []
            for _ in range(2):
                observation, _ = env.reset(seed=200000)
                mask = env.action_masks()
                action, _ = model.predict(observation, deterministic=True,
                                          action_masks=mask)
                action = int(action)
                self.assertTrue(mask[action])
                observation, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
            self.assertEqual(actions[0], actions[1])
            row = {"seed": 200000, "pieces_survived": info["pieces"],
                   "lines": info["lines"], "score": info["score"],
                   "episode_reward": env.episode_reward,
                   "game_over": bool(terminated), "survived_cap": bool(truncated)}
            result = {"num_seeds": 1, "max_pieces": 50000,
                      "per_seed": [row], "aggregate": aggregate([row], 50000)}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "result.json"
                atomic_json(path, result)
                self.assertEqual(json.loads(path.read_text()), result)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
