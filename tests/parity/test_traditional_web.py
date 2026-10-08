"""Browser strategy and executed path must match official Python Core stepwise."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.tetris_core import TetrisCore  # noqa: E402
from training.traditional.v1 import V1Adapter  # noqa: E402
from training.traditional.v2 import V2, V2Config  # noqa: E402


class TraditionalWebParityTests(unittest.TestCase):
    def test_fixed_seed_games(self):
        for mode in ("v1", "hold", "beam"):
            for seed in (7, 100000):
                # A complete fixed-cap episode, including every decision and
                # resulting official state through the cap or earlier Game Over.
                steps = (50 if seed == 7 else 20) if mode == "beam" else (100 if seed == 7 else 30)
                with self.subTest(mode=mode, seed=seed):
                    result = subprocess.run(
                        ["node", "tests/parity/traditional-js-runner.cjs"],
                        input=json.dumps(dict(mode=mode, seed=seed, steps=steps)),
                        text=True, capture_output=True, check=True, cwd=ROOT)
                    js = json.loads(result.stdout)
                    core = TetrisCore(seed)
                    policy = V1Adapter(seed) if mode == "v1" else V2(V2Config(mode=mode))
                    self.assertEqual(len(js["states"]), len(js["actions"]) + 1)
                    for i, state in enumerate(js["states"]):
                        expected = core.get_public_observation()
                        expected["placedCount"] = i
                        self.assertEqual(state, expected, f"{mode} seed={seed} step={i}")
                        if i < len(js["actions"]):
                            action = policy.choose(expected)
                            self.assertEqual(js["actions"][i], action,
                                             f"{mode} seed={seed} step={i}")
                            self.assertIn(action, [p["actionId"] for p in core.get_legal_placements()])
                            core.step(action)

    def test_worker_rules_match_game_rules(self):
        script = r'''
const fs = require("fs"), vm = require("vm");
const {game} = require("./tests/game-harness.cjs");
const g = game(); g.run("resetGame({seed:7})");
const observation = JSON.parse(g.run("JSON.stringify(getPublicObservation())"));
const context = vm.createContext({});
for (const name of ["traditional-rules.js", "legal-placements.js", "traditional-ai.js"])
  vm.runInContext(fs.readFileSync("tetris-game/" + name, "utf8"), context);
context.observation = observation;
const worker = vm.runInContext("traditionalPlacements(observation)", context);
const gamePlacements = g.run("getLegalPlacements()");
if (JSON.stringify(worker) !== JSON.stringify(gamePlacements)) process.exit(1);
for (const mode of ["hold", "beam"])
  if (vm.runInContext(`traditionalChooseV2(observation, "${mode}")`, context) !==
      g.run(`traditionalChooseV2(getPublicObservation(), "${mode}")`)) process.exit(2);
'''
        subprocess.run(["node", "-e", script], check=True, cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
