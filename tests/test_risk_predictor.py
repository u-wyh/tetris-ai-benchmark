import numpy as np
import torch

from training.risk_predictor import RiskPredictor, labels, probabilities


def test_right_censor_and_death():
    assert labels(900, 1000, True, 100) == 1
    assert labels(899, 1000, True, 100) == 0
    assert labels(950, 1000, False, 100) is None
    assert labels(900, 1000, False, 100) == 0


def test_probability_shape_and_range():
    model = RiskPredictor().eval()
    result = probabilities(model, np.zeros(237), np.zeros((5, 16)))
    assert result.shape == (5, 2)
    assert np.isfinite(result).all()
    assert ((0 <= result) & (result <= 1)).all()


def test_episode_split_and_main_checkpoint():
    from training.local_correction_data import TRAIN, VALIDATION
    from training.evaluation.hybrid_queue import HASHES, PPO, digest
    assert set(TRAIN).isdisjoint(VALIDATION)
    assert digest(PPO) == HASHES["ppo"]


def test_auc_with_tied_baseline_scores():
    from training.train_risk_predictor import metrics
    scores = np.zeros((4, 2), dtype=np.float32)
    target = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], np.float32)
    mask = np.ones((4, 2), dtype=bool)
    result = metrics(scores, target, mask)["1000"]
    assert result["pr_auc"] == 0.5
    assert result["roc_auc"] == 0.5


def test_checkpoint_contains_restorable_optimizer_and_rng(tmp_path, monkeypatch):
    from training import train_risk_predictor as trainer
    from training.train_ppo import atomic_json
    monkeypatch.setattr(trainer, "RUN", tmp_path)
    atomic_json(tmp_path / "protocol.json", {"test": True})
    model = RiskPredictor()
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.zeros((1, 237)), torch.zeros((1, 16))).sum()
    loss.backward(); optimizer.step()
    trainer.checkpoint(tmp_path / "checkpoint.pt", model, optimizer, 3, 0.2, 1)
    record = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    saved_rng = torch.get_rng_state()
    restored = RiskPredictor(); restored.load_state_dict(record["model"])
    resumed_optimizer = torch.optim.AdamW(restored.parameters())
    resumed_optimizer.load_state_dict(record["optimizer"])
    assert record["epoch"] == 3 and record["protocol_sha256"] == trainer.digest(tmp_path / "protocol.json")
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    assert resumed_optimizer.state_dict()["state"]
    assert torch.equal(record["torch_rng"], saved_rng)
