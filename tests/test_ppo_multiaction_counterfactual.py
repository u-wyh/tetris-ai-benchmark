import unittest
import json
import pickle

from training.evaluation.ppo_multiaction_counterfactual import (RUN, V2, branch,
    candidates, compare, freeze_selection, grade, load_models, _verified)


class MultiActionCounterfactualTests(unittest.TestCase):
    def test_legal_topk_dedup_and_snapshot_isolation(self):
        item = freeze_selection()["states"][0]
        row = json.loads((V2/"branches"/f"seed_{item['seed']}"/f"step_{item['step']}.json").read_text())["row"]
        models = load_models()
        choices = candidates(row, models["ppo"])
        ids = [choice["action"] for choice in choices]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids) <= set(row["legal_ids"]))
        self.assertEqual(ids[0], row["ppo_action"])
        self.assertEqual(sorted(c["ppo_rank"] for c in choices if c["ppo_rank"] <= 5),
                         sorted(set(c["ppo_rank"] for c in choices if c["ppo_rank"] <= 5)))
        snapshot = pickle.loads((V2/"snapshots"/f"seed_{item['seed']}.pkl").read_bytes())[item["step"]]
        original = pickle.dumps(snapshot)
        first = branch(snapshot, ids[0], models, horizon=4)
        second = branch(snapshot, ids[0], models, horizon=4)
        self.assertEqual(first["scores"], second["scores"])
        self.assertEqual(first["piece_trace"], second["piece_trace"])
        self.assertEqual(pickle.dumps(snapshot), original)

    def test_resume_identity_rejects_mismatch(self):
        selection = RUN/"selection.json"
        self.assertIsNotNone(_verified(selection, {"version": freeze_selection()["version"]}))
        with self.assertRaises(RuntimeError):
            _verified(selection, {"version": "wrong"})

    def test_common_window_and_right_censor(self):
        p = {"pieces": 3, "scores": [10, 20, 30], "fours": [0, 1, 1],
             "game_over": True, "censored": False}
        c = {"pieces": 5, "scores": [9, 19, 29, 39, 49], "fours": [0, 1, 1, 2, 2],
             "game_over": False, "censored": True}
        result = compare(p, c)
        self.assertEqual(result["common_pieces"], 3)
        self.assertEqual(result["candidate_score"], 29)
        self.assertEqual(result["candidate_fours"], 1)
        self.assertEqual(result["survival_delta"], 2)

    def test_hold_sequence_change_prevents_a(self):
        p = {"pieces": 100, "scores": list(range(1, 101)), "fours": [0]*100,
             "game_over": True}
        c = {"pieces": 400, "scores": list(range(1, 401)), "fours": [0]*400,
             "game_over": False}
        base = {"risk": {"lock_out": False, "central_top4": 2, "height": 16,
                         "height_growth": 0}, "risk_score": 100, "well_after": 4,
                "after_pieces": {"current": "I", "hold": "T", "next": ["O"]}}
        other = {**base, "risk_score": 90, "after_pieces": {"current": "O", "hold": "I", "next": ["T"]}}
        self.assertEqual(grade(p, c, base, other)["category"], "B")
        other["after_pieces"] = base["after_pieces"]
        self.assertEqual(grade(p, c, base, other)["category"], "A")


if __name__ == "__main__":
    unittest.main()
