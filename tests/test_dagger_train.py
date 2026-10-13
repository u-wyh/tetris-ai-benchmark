"""DAgger actor optimizer and RNG resume exactly after a committed epoch."""
import pytest
import torch
from sb3_contrib import MaskablePPO

import training.dagger_train as dagger_train
from training.dagger_dataset import CHAMPION, capture
from training.imitation_bc import records as original_records
from training.imitation_dataset import digest


def test_dagger_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch):
    if not CHAMPION.exists():
        pytest.skip("Frozen BC checkpoint unavailable")
    model = MaskablePPO.load(str(CHAMPION), device="cpu")
    training, validation = tmp_path / "data", tmp_path / "validation"
    capture(1247777, training, model, limit=4)
    capture(1257777, validation, model, limit=4)
    original = list(original_records(dagger_train.ORIGINAL, [600000], 4))
    monkeypatch.setattr(dagger_train, "sample_original", lambda limit, seed: original[:limit])
    base_records = dagger_train.records

    def small_validation(dataset, seeds, limit=None):
        if dataset == dagger_train.ORIGINAL and list(seeds) == list(range(600100, 600108)):
            yield from base_records(dataset, [600100], 16)
        else:
            yield from base_records(dataset, seeds, limit)

    monkeypatch.setattr(dagger_train, "records", small_validation)
    interrupted, continuous = tmp_path / "interrupted", tmp_path / "continuous"
    first = dagger_train.train(training, validation, interrupted, epochs=2,
                               samples_per_source=4, max_new_epochs=1)
    assert len(first) == 2
    resumed = dagger_train.train(training, validation, interrupted, epochs=2,
                                  samples_per_source=4)
    whole = dagger_train.train(training, validation, continuous, epochs=2,
                                samples_per_source=4)
    assert len(resumed) == len(whole) == 3
    assert [r["dagger_validation"]["top1"] for r in resumed] == [
        r["dagger_validation"]["top1"] for r in whole]
    a = MaskablePPO.load(str(interrupted / "epoch_02.zip"), device="cpu")
    b = MaskablePPO.load(str(continuous / "epoch_02.zip"), device="cpu")
    assert all(torch.equal(value, b.policy.state_dict()[key])
               for key, value in a.policy.state_dict().items())
    saved = torch.load(interrupted / "training_state.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2
    assert digest(interrupted / "epoch_02.zip") == saved["checkpoint_sha256"]
    assert dagger_train.train(training, validation, interrupted, epochs=2,
                              samples_per_source=4) == resumed
    with (interrupted / "epoch_02.zip").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity"):
        dagger_train.train(training, validation, interrupted, epochs=2,
                            samples_per_source=4)
