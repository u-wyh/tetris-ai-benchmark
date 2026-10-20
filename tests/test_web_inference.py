import json
import subprocess

import pytest

from training.tetris_core.core import TetrisCore
from training.web_inference import Inference


@pytest.fixture(scope="module")
def engine():
    return Inference()


def browser_states(seed, actions):
    request = {"seed": seed, "actions": actions}
    process = subprocess.run(["node", "tests/parity/js-runner.cjs"],
                             input=json.dumps(request), text=True,
                             capture_output=True, check=True)
    return json.loads(process.stdout)["states"]


@pytest.mark.parametrize("mode", ["ppo", "bc", "b12"])
def test_fixed_seed_browser_python_core_and_model_actions(engine, mode):
    seed = 12345
    core = TetrisCore(seed)
    actions = []
    for _ in range(20):
        browser = browser_states(seed, actions)[-1]
        public = core.get_public_observation()
        assert browser["board"] == public["board"]
        assert browser["current"] == public["currentPiece"]
        assert browser["next3"] == public["next"]
        assert browser["hold"] == public["hold"]
        assert browser["score"] == public["score"]
        assert browser["lines"] == public["lines"]
        assert browser["gameOver"] == public["gameOver"]
        legal = [p["actionId"] for p in core.get_legal_placements()]
        browser_legal = [i for i, allowed in enumerate(browser["actionMask"]) if allowed]
        assert browser_legal == sorted(legal)
        if public["gameOver"]:
            break
        browser_public = {"board": browser["board"], "currentPiece": browser["current"],
                          "hold": browser["hold"], "holdAvailable": browser["holdAvailable"],
                          "next": browser["next3"], "score": browser["score"],
                          "level": browser["level"], "lines": browser["lines"],
                          "phase": "playing", "gameOver": False}
        action = engine.decide(mode, browser_public, legal)["actionId"]
        assert action in legal
        assert engine.decide(mode, browser_public, legal)["actionId"] == action
        actions.append(action)
        core.step(action)
    assert len(actions) == 20


def test_request_rejects_changed_browser_mask(engine):
    public = TetrisCore(12345).get_public_observation()
    with pytest.raises(ValueError, match="legal actions differ"):
        engine.decide("ppo", public, [1, 2])
