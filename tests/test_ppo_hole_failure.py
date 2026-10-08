"""Trajectory diagnostics are read-only and exclude dead episodes from time curves."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.ppo_hole_failure import (
    CAP, FEATURES, audit_against_paired, column_heights, curve_before_death,
    curve_by_piece, plot_curves, summary, trace_seed, visible_board,
)
from training.env import TetrisEnv


class FirstLegal:
    def predict(self, observation, deterministic, action_masks):
        assert deterministic
        return int(np.flatnonzero(action_masks)[0]), None


def test_trace_uses_real_core_reward_and_preserves_game_rules():
    model = FirstLegal()
    raw, shaped = TetrisEnv(max_pieces=5), TetrisEnv(max_pieces=5, hole_penalty_coef=0.1)
    try:
        a = trace_seed(model, raw, 100000, "raw")
        b = trace_seed(model, shaped, 100000, "shaped")
    finally:
        raw.close()
        shaped.close()
    assert a["pieces_survived"] == b["pieces_survived"] == 5
    assert a["score"] == b["score"] and a["lines"] == b["lines"]
    assert a["survived_cap"] and b["survived_cap"]
    assert [s["action_id"] for s in a["steps"]] == [s["action_id"] for s in b["steps"]]
    assert [s["board_after"] for s in a["steps"]] == [s["board_after"] for s in b["steps"]]
    assert all(step["shaped_reward"] is None for step in a["steps"])
    assert b["actual_episode_reward"] == pytest.approx(
        b["raw_reward_total"] - b["hole_penalty_at_lambda_010"])
    for step in b["steps"]:
        assert step["new_holes"] == max(0, step["after"]["holes"] - step["before"]["holes"])
        assert len(step["board_after"]) == 20
        assert all(len(row) == 10 for row in step["board_after"])
    assert b["steps"][-1]["lines_total"] == b["lines"]
    metrics = summary([b])
    assert metrics["pieces_total"] == 5 and metrics["num_seeds"] == 1
    assert metrics["survival_cap_rate"] == 1
    assert metrics["new_holes_total"] == sum(s["new_holes"] for s in b["steps"])


def test_curves_count_survivors_instead_of_padding_dead_episodes_with_zero():
    def step(index, height):
        after = {key: height for key in FEATURES}
        return {"piece_index": index, "after": after}
    episodes = [
        {"game_over": True, "steps": [step(1, 2), step(2, 4)]},
        {"game_over": True, "steps": [step(1, 6), step(2, 8), step(3, 10)]},
    ]
    by_piece = curve_by_piece(episodes)
    assert [row["survivors"] for row in by_piece] == [2, 2, 1]
    assert [row["max_height"] for row in by_piece] == [4, 6, 10]
    before_death = curve_before_death(episodes, window=3)
    assert [row["episodes"] for row in before_death] == [1, 2, 2]
    assert [row["max_height"] for row in before_death] == [6, 5, 7]


def test_reference_audit_rejects_changed_trajectory():
    episode = {"seed": 100000, "pieces_survived": 1, "lines": 0, "score": 10,
               "game_over": True, "steps": [{"new_holes": 2}]}
    paired = {"raw": {"per_seed": [{"seed": 100000, "pieces_survived": 1,
                                     "lines": 0, "score": 10, "game_over": True,
                                     "new_holes_total": 2}]}}
    audit_against_paired("raw", [episode], paired)
    with pytest.raises(AssertionError, match="Trace differs"):
        audit_against_paired("raw", [dict(episode, score=11)], paired)
    with pytest.raises(AssertionError, match="new-hole"):
        audit_against_paired("raw", [dict(episode, steps=[{"new_holes": 1}])], paired)


def test_visible_board_has_stable_compact_shape():
    board = [[None] * 10 for _ in range(20)]
    board[19][9] = "I"
    rows = visible_board(board)
    assert rows[0] == "." * 10
    assert rows[-1] == "." * 9 + "I"
    assert column_heights(rows) == [0] * 9 + [1]
    assert CAP == 5000


def test_plots_include_survivor_counts_and_terminal_shape(tmp_path):
    def make_episode(label):
        board = ["." * 10 for _ in range(19)] + ["I" + "." * 9]
        steps = [{"piece_index": index, "after": {key: index for key in FEATURES},
                  "board_after": board} for index in (1, 2)]
        return {"game_over": True, "steps": steps, "label": label}
    episodes = {label: [make_episode(label)] for label in ("raw", "shaped")}
    curves = {label: {"by_piece": curve_by_piece(rows),
                      "before_death": curve_before_death(rows)}
              for label, rows in episodes.items()}
    plot_curves(tmp_path, episodes, curves)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5
    assert curves["raw"]["by_piece"][1]["survivors"] == 1


def test_archived_per_seed_traces_match_summary_and_seed_policy():
    root = Path(__file__).resolve().parents[1] / "reports/experiments/ppo_hole_v1_failure_traces"
    result = json.loads((root / "summary.json").read_text())
    assert result["seeds"] == list(range(100000, 100032))
    for label in ("raw", "shaped"):
        files = sorted(root.glob(f"{label}_seed_*.json.gz"))
        assert len(files) == 32
        episodes = [json.loads(gzip.decompress(path.read_bytes())) for path in files]
        assert [episode["seed"] for episode in episodes] == result["seeds"]
        assert sum(episode["pieces_survived"] for episode in episodes) == result["models"][label]["pieces_total"]
        assert all(episode["steps"][-1]["lines_total"] == episode["lines"] for episode in episodes)
