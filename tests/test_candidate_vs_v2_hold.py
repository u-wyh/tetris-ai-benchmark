"""Formal paired protocol uses the frozen model and the same game core."""

import json

import training.evaluation.candidate_vs_v2_hold as paired
from training.env.candidate_env import CandidateTetrisEnv
from training.tetris_core import TetrisCore
from training.traditional.v2 import V2


def test_frozen_sources_and_historical_v2_configuration():
    final, policy, protocol = paired.preflight()
    assert policy.mode == "hold"
    assert len(protocol["seeds"]) == len(final["per_seed"]) == 100
    assert protocol["max_pieces"] == 50000
    assert protocol["candidate_final_reused"] is True


def test_hold_only_actions_replay_identically_in_candidate_core():
    _, config, _ = paired.preflight()
    core = TetrisCore(200000)
    env = CandidateTetrisEnv(max_pieces=80)
    policy = V2(config)
    env.reset(seed=200000)
    try:
        for _ in range(60):
            action = policy.choose(core.get_public_observation(), core.get_legal_placements())
            assert env.action_masks()[action]
            core.step(action)
            _, _, terminated, truncated, info = env.step(action)
            assert core.board == env.core.board
            assert (core.score, core.lines) == (info["score"], info["lines"])
            assert core.get_public_observation() == env.core.get_public_observation()
            if terminated or truncated:
                break
    finally:
        env.close()


def test_paired_result_preserves_censoring_and_seed_order(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "OUT", tmp_path)
    monkeypatch.setattr(paired, "TRADITIONAL", tmp_path / "traditional")
    paired.TRADITIONAL.mkdir()
    (paired.TRADITIONAL / "config.json").write_text(json.dumps({
        "created_at": "2026-10-10T00:00:00+00:00"}))
    seeds = [200000, 200001]
    candidate = {"per_seed": [
        {"seed": 200000, "pieces_survived": 50000, "lines": 20000,
         "score": 100, "game_over": False, "survived_cap": True},
        {"seed": 200001, "pieces_survived": 2000, "lines": 700,
         "score": 20, "game_over": True, "survived_cap": False}]}
    for seed, pieces, capped in ((200000, 50000, True), (200001, 3000, False)):
        (paired.TRADITIONAL / f"seed_{seed}.json").write_text(json.dumps({
            "seed": seed, "pieces_survived": pieces, "lines": pieces // 3,
            "score": 40, "game_over": not capped, "truncated": capped,
            "wall_seconds": 1.0}))
    result = paired.comparison(candidate, paired.TRADITIONAL, {"seeds": seeds})
    assert result["paired"]["both_capped"] == 1
    assert result["paired"]["survival_ties"] == 1
    assert result["paired"]["traditional_survival_wins"] == 1
    assert result["traditional_runtime"]["pieces_per_second"] == 26500
