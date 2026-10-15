import copy

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.ppo_lookahead_policy import LookaheadConfig, LookaheadController, board_value, public_key
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public


def test_public_simulation_matches_core_and_does_not_mutate_branch():
    env = FastCandidateTetrisEnv(max_pieces=4)
    try:
        env.reset(seed=990000)
        before = env.core.get_public_observation()
        saved = copy.deepcopy(before)
        placements = get_legal_placements(before)
        for placement in placements[:5]:
            child, cleared = simulate_public(before, placement)
            assert before == saved
            assert child["phase"] in ("playing", "unknown", "over")
            assert isinstance(board_value(child, before, cleared), float)
        placement = next(p for p in placements if p["hold"])
        child, _ = simulate_public(before, placement)
        env.step(placement["actionId"])
        actual = env.core.get_public_observation()
        assert child["board"] == actual["board"]
        assert child["score"] == actual["score"]
        assert child["hold"] == actual["hold"]
        assert child["currentPiece"]["type"] == actual["currentPiece"]["type"]
    finally:
        env.close()


def test_visible_three_step_simulation_stops_at_unknown():
    env = FastCandidateTetrisEnv(max_pieces=4)
    try:
        env.reset(seed=990004)
        public = env.core.get_public_observation()
        for index in range(3):
            placement = next(p for p in get_legal_placements(public) if not p["hold"])
            child, _ = simulate_public(public, placement)
            env.step(placement["actionId"])
            actual = env.core.get_public_observation()
            assert child["board"] == actual["board"]
            assert child["score"] == actual["score"]
            assert child["hold"] == actual["hold"]
            if child["phase"] == "unknown":
                assert index == 2
                break
            assert child["currentPiece"]["type"] == actual["currentPiece"]["type"]
            public = child
    finally:
        env.close()


def test_unknown_is_not_death_and_hold_is_in_key():
    env = FastCandidateTetrisEnv(max_pieces=2)
    try:
        env.reset(seed=990001)
        public = env.core.get_public_observation()
        first = get_legal_placements(public)[0]
        child, _ = simulate_public(public, first)
        child["next"] = []
        child["phase"] = "playing"
        child["holdAvailable"] = False
        second = get_legal_placements(child)[0]
        leaf, _ = simulate_public(child, second)
        assert leaf["phase"] == "unknown" and not leaf["gameOver"]
        altered = copy.deepcopy(leaf); altered["hold"] = "I" if leaf["hold"] != "I" else "O"
        assert public_key(altered) != public_key(leaf)
    finally:
        env.close()


def test_hidden_bag_rng_does_not_change_search_decision():
    from training.evaluation.ppo_lookahead_queue import load_models
    env = FastCandidateTetrisEnv(max_pieces=2)
    try:
        obs, _ = env.reset(seed=990002)
        models = load_models()
        policy = LookaheadController(models["ppo"], LookaheadConfig("S2", trigger_height=0))
        mask = env.action_masks()
        public = env.core.get_public_observation()
        original = policy.decide(obs, mask, public, env._placement_cache)
        env.core.queue[3:] = ["I", "I"]
        env.core.bag = ["Z", "Z"]
        env.core.rng.state = 123456
        assert env.core.get_public_observation() == public
        repeated = policy.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
        assert original[0] == repeated[0]
        assert {k: v for k, v in original[1].items() if k != "elapsed_ms"} == {
            k: v for k, v in repeated[1].items() if k != "elapsed_ms"}
        assert mask[original[0]]
    finally:
        env.close()


def test_failed_four_clear_screen_stops_long_extension(monkeypatch):
    from training.evaluation import ppo_lookahead_queue as queue
    monkeypatch.setattr(queue, "common_score_ratio", lambda *args: 1.0)
    baseline = {"four_fraction": 0.93, "p10_pieces": 10000,
                "mean_score": 500_000_000, "top1_retention": 1.0}
    failed = {**baseline, "four_fraction": 0.88, "mean_score": 510_000_000}
    assert not queue.plausible_for_50k({"ppo": baseline, "S1": failed, "S2": failed})
