import json

import pytest

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.ppo_lookahead_queue import load_models
from training.evaluation.ppo_lookahead_v3_audit import RUN, terms
from training.ppo_lookahead_v3 import V3Config, V3Controller
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public


def test_scoring_decomposition_and_no_four_bonus():
    env = FastCandidateTetrisEnv(max_pieces=2)
    try:
        env.reset(seed=991000)
        public = env.core.get_public_observation()
        placement = get_legal_placements(public)[0]
        child, cleared = simulate_public(public, placement)
        parts = terms(child, public, cleared)
        assert parts["extra_tetris"] == 2 * (cleared == 4)
        assert parts["sum"] - parts["extra_tetris"] == pytest.approx(
            sum(value for key, value in parts.items() if key not in ("sum", "score_gain_raw", "extra_tetris")))
    finally:
        env.close()


def test_v3_is_deterministic_legal_and_hidden_rng_independent():
    model = load_models()["ppo"]
    env = FastCandidateTetrisEnv(max_pieces=2)
    try:
        obs, _ = env.reset(seed=991000)
        mask = env.action_masks()
        public = env.core.get_public_observation()
        controller = V3Controller(model, V3Config("V3B", hold_extra_margin=0.45))
        action, first = controller.decide(obs, mask, public, env._placement_cache)
        env.core.queue[3:] = ["I", "I"]
        env.core.bag = ["Z", "Z"]
        env.core.rng.state = 123456
        repeated, second = controller.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
        assert mask[action] and action == repeated
        assert {k:v for k,v in first.items() if k != "elapsed_ms"} == {
            k:v for k,v in second.items() if k != "elapsed_ms"}
    finally:
        env.close()


def test_known_failure_scoring_regression():
    path = RUN / "diagnostics/S1_score_audit_seed_990109.json"
    if not path.exists():
        pytest.skip("Failure audit not generated yet")
    events = json.loads(path.read_text())["events"]
    assert len(events) == 24
    last = next(e for e in events if e["step"] == 2849)
    assert last["search"]["action"] == 249 and last["ppo"]["action"] == 383
    assert last["no_tetris_bonus_gap"] < last["switch_margin"]
    assert not last["hold_changed"]


def test_snapshot_restores_exact_next_piece_and_rng():
    left = FastCandidateTetrisEnv(max_pieces=8)
    right = FastCandidateTetrisEnv(max_pieces=8)
    try:
        left.reset(seed=991001)
        right.reset(seed=991001)
        action = int(next(iter(get_legal_placements(left.core.get_public_observation())))["actionId"])
        left.step(action)
        right.set_state(left.get_state())
        assert left.core.get_public_observation() == right.core.get_public_observation()
        assert left.core.rng.state == right.core.rng.state
        for _ in range(3):
            legal = get_legal_placements(left.core.get_public_observation())
            chosen = legal[0]["actionId"]
            a = left.step(chosen)
            b = right.step(chosen)
            assert a[-1] == b[-1]
            assert left.core.get_public_observation() == right.core.get_public_observation()
            assert left.core.rng.state == right.core.rng.state
    finally:
        left.close()
        right.close()
