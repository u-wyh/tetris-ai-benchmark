"""Run with: python3 -m unittest discover -s tests/parity -v"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.tetris_core import TetrisCore, decode_action, encode_action  # noqa: E402
from training.tetris_core.rng import Mulberry32  # noqa: E402


def runner(command, request):
    result = subprocess.run(command, input=json.dumps(request), text=True,
                            capture_output=True, check=True, cwd=ROOT)
    return json.loads(result.stdout)


def js(request):
    return runner(["node", "tests/parity/js-runner.cjs"], request)


def py(request):
    return runner([sys.executable, "tests/parity/python_runner.py"], request)


def compare(seed, request, reference, candidate):
    if reference["actions"] != candidate["actions"]:
        raise AssertionError(f"seed={seed} actions diverge: JS={reference['actions']} Python={candidate['actions']}")
    if len(reference["states"]) != len(candidate["states"]):
        raise AssertionError(f"seed={seed} number of state snapshots differs")
    for index, (expected, actual) in enumerate(zip(reference["states"], candidate["states"])):
        action = reference["actions"][index - 1] if index else None
        step = index - 1
        for field in expected:
            if expected[field] == actual[field]:
                continue
            detail = f"seed={seed} step={step} action={action} fixture={request.get('fixture')} field={field}"
            if field == "actionMask":
                first = next((i for i, (a, b) in enumerate(zip(expected[field], actual[field])) if a != b), None)
                detail += (f" firstDifferentActionId={first}\nJS mask={expected[field]}"
                           f"\nPython mask={actual[field]}")
            elif field == "board":
                detail += (f"\nJS board={expected['board']}\nPython board={actual['board']}"
                           f"\nJS current={expected['current']} Python current={actual['current']}"
                           f"\nJS hold={expected['hold']} Python hold={actual['hold']}"
                           f"\nJS next={expected['next3']} Python next={actual['next3']}")
            else:
                detail += f"\nJS={expected[field]}\nPython={actual[field]}"
            raise AssertionError(detail)


class ParityTests(unittest.TestCase):
    def test_all_1840_action_ids_match_js(self):
        script = '''const {game}=require("./tests/game-harness.cjs");
          const g=game(); process.stdout.write(g.run("JSON.stringify(Array.from({length:1840},(_,i)=>decodeAction(i)))"));'''
        expected = json.loads(subprocess.check_output(["node", "-e", script], cwd=ROOT))
        for action, decoded in enumerate(expected):
            self.assertEqual(decode_action(action), decoded)
            self.assertEqual(encode_action(**decoded), action)

    def test_mulberry32_and_first_100_pieces_match_js(self):
        script = '''const {game}=require("./tests/game-harness.cjs");
          const g=game(); g.run("resetGame({seed:12345})");
          process.stdout.write(g.run("JSON.stringify({random:Array.from({length:100},()=>nextRandom(gameplayRngState)),pieces:(()=>{let result=[];for(let i=0;i<100;i++){result.push(active.type);spawnPiece()}return result})()})"));'''
        # Compare the pure RNG separately from 7-Bag's own state consumption.
        pure_script = '''const {game}=require("./tests/game-harness.cjs"); const g=game();
          process.stdout.write(g.run("JSON.stringify((()=>{let state=12345,result=[];for(let i=0;i<100;i++){let next=nextRandom(state);state=next.state;result.push(next)}return result})())"));'''
        js_rng = json.loads(subprocess.check_output(["node", "-e", pure_script], cwd=ROOT))
        py_rng = Mulberry32(12345)
        for sample in js_rng:
            self.assertEqual(py_rng.random(), sample["value"])
            self.assertEqual(py_rng.state, sample["state"])
        js_pieces = json.loads(subprocess.check_output(["node", "-e", script], cwd=ROOT))["pieces"]
        core = TetrisCore(12345)
        py_pieces = []
        for _ in range(100):
            py_pieces.append(core.current["type"])
            core._spawn()
        self.assertEqual(py_pieces, js_pieces)
        for index in range(0, 98, 7):
            self.assertEqual(set(py_pieces[index:index + 7]), set("IJLOSTZ"))

    def test_mask_queries_are_pure_and_deterministic(self):
        core = TetrisCore(20261006)
        before = json.dumps((core.board, core.current, core.queue, core.bag, core.hold,
                             core.score, core.lines, core.level, core.rng.state))
        first = core.get_action_mask()
        self.assertEqual(len(first), 1840)
        self.assertEqual(first, core.get_action_mask())
        self.assertEqual(before, json.dumps((core.board, core.current, core.queue, core.bag,
                                             core.hold, core.score, core.lines, core.level,
                                             core.rng.state)))
        observation = core.get_public_observation()
        self.assertEqual(len(observation["next"]), 3)
        self.assertFalse({"queue", "bag", "rng", "initial_seed"} & observation.keys())
        observation["board"][0][0] = "T"
        self.assertIsNone(core.board[0][0])

    def test_stepwise_parity(self):
        scenarios = []
        for seed in (1, 42, 12345, 20261006):
            for strategy in ("mixed", "hold", "left", "right"):
                scenarios.append({"seed": seed, "strategy": strategy, "steps": 12,
                                  "includePlacements": seed == 1 and strategy == "mixed"})
        scenarios.extend({"seed": seed, "strategy": "mixed", "steps": 60}
                         for seed in (12345, 20261006))
        scenarios.extend({"seed": 42, "fixture": fixture, "actions": [action]}
                         for fixture, action in (("single_clear", 90), ("double_clear", 90),
                                                 ("triple_clear", 341), ("quad_clear", 341),
                                                 ("level_up", 90), ("block_out", 95),
                                                 ("lock_out", 71), ("partial_lock", 2)))
        scenarios.extend({"seed": 42, "fixture": fixture, "actions": [action],
                          "includePlacements": True}
                         for fixture, action in (("srs_floor", 319), ("tuck", 62),
                                                 ("tuck", 244)))
        snapshots = 0
        for request in scenarios:
            with self.subTest(request=request):
                reference = js(request)
                actions = reference["actions"] if "steps" in request else request["actions"]
                candidate = py({"seed": request["seed"], "actions": actions,
                                "includePlacements": request.get("includePlacements", False),
                                **({"fixture": request["fixture"]} if "fixture" in request else {})})
                compare(request["seed"], request, reference, candidate)
                snapshots += len(reference["states"])
                if "fixture" in request and request["fixture"].endswith("clear"):
                    self.assertGreater(reference["states"][-1]["lines"], 0)
                if request.get("fixture") in ("block_out", "lock_out"):
                    self.assertTrue(reference["states"][-1]["gameOver"])
                if request.get("fixture") == "partial_lock":
                    self.assertFalse(reference["states"][-1]["gameOver"])
        self.assertEqual(js({"seed": 12345, "strategy": "mixed", "steps": 12}),
                         js({"seed": 12345, "strategy": "mixed", "steps": 12}))
        repeat_actions = js({"seed": 12345, "strategy": "hold", "steps": 12})["actions"]
        self.assertEqual(py({"seed": 12345, "actions": repeat_actions}),
                         py({"seed": 12345, "actions": repeat_actions}))
        print(f"Parity scenarios={len(scenarios)} snapshots={snapshots}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
