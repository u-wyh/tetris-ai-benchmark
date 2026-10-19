from training.evaluation import hybrid_b12_expert_dataset as study


def test_hash_sampler_is_stable_and_categories_have_separate_draws():
    a = study.sample_uniform(1450000, 1234, "A")
    assert a == study.sample_uniform(1450000, 1234, "A")
    assert 0 <= a < 1
    assert a != study.sample_uniform(1450000, 1234, "B")


def test_online_sampling_uses_quota_spacing_and_current_actions(monkeypatch):
    class Expert:
        def choose(self, public, placements):
            return 3

    monkeypatch.setattr(study, "sample_uniform", lambda *args: 0.0)
    state = {"counts": {x: 0 for x in study.SAMPLER},
             "eligible": {x: 0 for x in study.SAMPLER}, "last_step": -1}
    decision = {"ppo_action": 1, "bc_action": 2, "source": "ppo", "reason": "low_risk"}
    a = study.classify_state(1450000, 100, {}, [], decision, Expert(), (0, 0, 17, 26, 0), state)
    assert a["category"] == "A" and a["independent_v2"]
    assert a["actions"] == {"ppo": 1, "bc": 2, "b12": 1, "v2": 3}
    assert study.classify_state(1450000, 101, {}, [], decision, Expert(), (0, 0, 17, 26, 0), state) is None
    assert study.classify_state(1450000, 600, {}, [], decision, Expert(), (0, 0, 15, 20, 0), state)["category"] == "B"
    state["counts"]["A"] = study.SAMPLER["A"]["quota"]
    assert study.classify_state(1450000, 1200, {}, [], decision, Expert(), (0, 0, 17, 26, 0), state) is None


def test_right_censored_branches_do_not_create_success_labels():
    a = {"pieces_after": 3000, "game_over": True}
    b = {"pieces_after": 5000, "game_over": False}
    assert study.classify_pair(a, b) == "beneficial"
    assert study.classify_pair(b, a) == "harmful"
    assert study.classify_pair({"pieces_after": 5000, "game_over": False},
                               {"pieces_after": 5000, "game_over": False}) == "uncertain"
    assert study.classify_pair({"pieces_after": 4700, "game_over": True}, b) == "uncertain"
