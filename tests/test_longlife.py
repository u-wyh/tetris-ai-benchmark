import unittest
import tempfile
from pathlib import Path

from training.evaluation.longlife import bootstrap_mean, common_continuation, find_disagreements, markdown, score_at_targets, score_breakdown, summarize
from training.evaluation.extend_traditional import prepare
from training.evaluation.profile_longlife import profile
from training.evaluation.traditional import atomic
from training.tetris_core import TetrisCore


class LonglifeAnalysisTests(unittest.TestCase):
    def test_equal_piece_replay_and_capped_ties(self):
        core = TetrisCore(7)
        actions = []
        for _ in range(4):
            action = core.get_legal_placements()[0]["actionId"]
            core.step(action)
            actions.append(action)
        self.assertEqual(score_at_targets(7, actions, {2, 4})[4], core.score)
        breakdown = score_breakdown(dict(seed=7, actions=actions, score=core.score,
                                         lines=core.lines, pieces_survived=4))
        self.assertEqual(breakdown["line_points"] + breakdown["drop_points"], core.score)
        row = dict(seed=7, actions=actions, pieces_survived=4, score=core.score,
                   lines=core.lines, game_over=False, truncated=True,
                   decision_ms=[1.0] * 4, wall_seconds=0.01)
        base = dict(seeds=[7], max_pieces=4, git_commit="same")
        v1 = dict(base, strategy="v1-adapter-1")
        v2 = dict(base, strategy="v2-1", code_sha256="v2",
                  policy=dict(mode="hold", weights={"x": 1}, death_penalty=1e6,
                              lookahead=.58, beam_width=8))
        configs = dict(v1=v1, hold=v2,
                       beam={**v2, "policy": {**v2["policy"], "mode": "beam"}})
        rows = {mode: {7: row} for mode in ("v1", "hold", "beam")}
        result = summarize(configs, rows)
        pair = result["pairs"]["beam_minus_hold"]
        self.assertEqual(pair["both_truncated"], 1)
        self.assertEqual(pair["restricted_mean_difference"], 0)
        self.assertEqual(pair["mean_equal_piece_score_per_piece_difference"], 0)
        self.assertIn("Both capped", markdown(result))
        wrong = {**configs, "beam": {**configs["beam"], "max_pieces": 5}}
        with self.assertRaisesRegex(ValueError, "caps must match"):
            summarize(wrong, rows)

    def test_bootstrap_is_reproducible(self):
        self.assertEqual(bootstrap_mean([1, 3], samples=100),
                         bootstrap_mean([1, 3], samples=100))

    def test_counterfactual_uses_legal_first_action(self):
        core = TetrisCore(7)
        action = core.get_legal_placements()[0]["actionId"]
        rows = {"hold": {7: {"actions": [action]}}, "beam": {7: {"actions": [action]}}}
        case = dict(seed=7, source="hold", step=0)
        outcome = common_continuation(rows, case, action, horizon=3)
        self.assertEqual(outcome["pieces"], 3)
        self.assertTrue(outcome["truncated"])

    def test_disagreement_scan_keeps_public_fixtures(self):
        core = TetrisCore(7)
        actions = []
        for _ in range(3):
            action = core.get_legal_placements()[0]["actionId"]
            core.step(action)
            actions.append(action)
        rows = {mode: {7: {"actions": actions}} for mode in ("hold", "beam")}
        result = find_disagreements(rows, [7], max_per_source=3, stride=1,
                                    keep=2, counterfactual_horizon=3)
        for case in result["saved"]:
            self.assertNotIn("bag", case["observation"])
            self.assertNotIn("rng", case["observation"])

    def test_completed_seed_import_for_same_cap_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "pilot", Path(temporary) / "formal"
            source.mkdir()
            atomic(source / "config.json", dict(seeds=[7], max_pieces=4,
                  strategy="v1-adapter-1", git_commit="same", code_sha256="digest"))
            atomic(source / "status.json", dict(state="complete"))
            atomic(source / "seed_7.json", dict(seed=7, actions=[1] * 4,
                  pieces_survived=4, decision_ms=[1.0] * 4,
                  game_over=False, truncated=True))
            prepare(source, output, [7, 8])
            self.assertTrue((output / "seed_7.json").exists())
            self.assertFalse((output / "seed_8.json").exists())
            self.assertTrue((output / "import_manifest.json").exists())

    def test_matched_state_profile_reads_only_public_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "seed_7.json"
            core = TetrisCore(7)
            action = core.get_legal_placements()[0]["actionId"]
            atomic(source, dict(seed=7, actions=[action]))
            result = profile(source, repeats=1, steps=(0,))
            self.assertEqual(result["summary"]["states"], 1)
            self.assertEqual(result["summary"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
