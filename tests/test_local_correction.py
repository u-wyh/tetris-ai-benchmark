"""Local imitation must preserve official masking and the frozen reference."""
import torch
import io

import training.local_correction_data as data
from training.evaluation.hybrid_queue import load_models
from training.imitation_bc import batch
from training.train_local_correction import masked_kl
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.local_correction_queue import _healthy, _promising


def test_masked_kl_and_legal_actions():
    models = load_models()
    env = FastCandidateTetrisEnv(max_pieces=100)
    try:
        observation, _ = env.reset(seed=730901)
        mask = env.action_masks()
        ids = mask.nonzero()[0]
        row = {"state": observation["state"].tolist(), "legal_ids": ids.tolist(),
               "legal_features": observation["candidate"][ids].tolist(),
               "action": int(models["ppo"].predict(observation, action_masks=mask,
                                                     deterministic=True)[0])}
        obs, legal, actions = batch([row], "cpu")
        policy = models["ppo"].policy
        assert torch.allclose(masked_kl(policy, policy, obs, legal), torch.zeros(1), atol=1e-6)
        distribution = policy.get_distribution(obs, action_masks=legal)
        assert legal[0, actions[0]]
        assert distribution.distribution.probs[~legal].sum() == 0
        logits = distribution.distribution.logits
        assert torch.isfinite(logits[legal]).all()
    finally:
        env.close()


def test_local_gradient_does_not_update_reference_or_critic():
    models = load_models()
    env = FastCandidateTetrisEnv(max_pieces=100)
    try:
        observation, _ = env.reset(seed=730902)
        mask = env.action_masks()
        ids = mask.nonzero()[0]
        row = {"state": observation["state"].tolist(), "legal_ids": ids.tolist(),
               "legal_features": observation["candidate"][ids].tolist(),
               "action": int(ids[-1])}
        obs, legal, actions = batch([row], "cpu")
        reference = models["ppo"].policy
        learner = models["bc"].policy
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        critic_before = {name: value.clone() for name, value in learner.value_net.state_dict().items()}
        loss = -learner.get_distribution(obs, action_masks=legal).log_prob(actions).mean()
        loss = loss + masked_kl(reference, learner, obs, legal).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.score_head.parameters())
        assert all(p.grad is None for p in reference.parameters())
        assert all(torch.equal(value, critic_before[name]) for name, value in learner.value_net.state_dict().items())
    finally:
        env.close()


def test_counterfactual_snapshot_is_reproducible(monkeypatch):
    models = load_models()
    monkeypatch.setattr(data, "HORIZON", 20)
    env = FastCandidateTetrisEnv(max_pieces=data.CAP)
    try:
        obs, _ = env.reset(seed=730903)
        for _ in range(3):
            action = int(models["ppo"].predict(obs, action_masks=env.action_masks(),
                                                deterministic=True)[0])
            obs, _, _, _, _ = env.step(action)
        state = env.get_state()
        action = int(models["ppo"].predict(obs, action_masks=env.action_masks(),
                                            deterministic=True)[0])
        first = data._branch(models, state, action)
        second = data._branch(models, state, action)
        assert first == second
        assert first["pieces"] == 20
    finally:
        env.close()


def test_sampler_keeps_every_candidate_until_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "RUN", tmp_path)
    monkeypatch.setattr(data, "CAP", 200)
    result = data.capture(730904, load_models())
    assert result["risky_candidates"] == result["risk_disagreements_seen"]
    assert result["normal_samples"] == result["normal_candidates_seen"]


def test_branch_selection_prioritizes_observed_predeath_states():
    rows = {i * 100: {"step": i * 100, "category": "candidate",
                      "death_distance": None, "ppo_risk": {"height": 19, "central_top4": 2}}
            for i in range(13)}
    rows[5000] = {"step": 5000, "category": "candidate", "death_distance": 500,
                  "ppo_risk": {"height": 12, "central_top4": 0}}
    chosen = data._branch_targets(rows)
    assert len(chosen) == 12
    assert 5000 in chosen


def test_selected_state_replay_matches_public_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "RUN", tmp_path)
    monkeypatch.setattr(data, "CAP", 200)
    models = load_models()
    meta = data.capture(730904, models)
    shard, _, _ = data._paths(730904)
    import gzip
    import json
    with gzip.open(shard, "rt", encoding="utf8") as stream:
        rows = {row["step"]: row for row in map(json.loads, stream)}
    targets = data._branch_targets(rows)
    snapshots = data._replay_snapshots(730904, rows, targets, models, meta["shard_sha256"])
    assert sorted(snapshots) == targets


def test_rapid_growth_requires_branch_and_score_evidence():
    row = {"ppo_risk": {"lock_out": False, "central_top4": 0, "height": 8,
                        "height_growth": 3, "new_holes": 0, "bumpiness": 19},
           "bc_risk": {"lock_out": False, "central_top4": 0, "height": 4,
                       "height_growth": 0, "new_holes": 0, "bumpiness": 9}}
    evidence = {"continuation_ok": True, "common_score_ratio": 1.0,
                "common_four_clears": {"ppo": 10, "bc": 10},
                "branches": {"ppo": {"pieces": 500, "game_over": True},
                             "bc": {"pieces": 1000, "game_over": False}}}
    assert data._grade(row, evidence)["category"] == "A"
    evidence["common_score_ratio"] = .7
    assert data._grade(row, evidence)["category"] == "B"
    evidence["common_score_ratio"] = 1.0
    row["ppo_risk"]["height_growth"] = 2
    assert data._grade(row, evidence)["category"] == "B"


def test_development_gate_preserves_four_clear_strategy():
    ppo = {"mean_pieces": 70000, "p10_pieces": 20000,
           "mean_score": 10_000_000_000, "four_fraction": .93}
    candidate = {"mean_pieces": 72000, "p10_pieces": 25000,
                 "mean_score": 10_000_000_000, "four_fraction": .89}
    assert _healthy(candidate, ppo)
    assert not _promising(candidate, ppo)
    candidate["four_fraction"] = .87
    assert not _healthy(candidate, ppo)


def test_rng_state_loaded_on_cuda_can_restore_cpu_generator():
    if not torch.cuda.is_available():
        return
    state = torch.get_rng_state()
    output = io.BytesIO()
    torch.save({"torch_rng": state}, output)
    output.seek(0)
    loaded = torch.load(output, map_location="cuda", weights_only=False)
    torch.set_rng_state(loaded["torch_rng"].cpu())
    assert torch.equal(torch.get_rng_state(), state)
