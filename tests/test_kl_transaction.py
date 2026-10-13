"""KL-PPO preserves the existing task commit/restart contract."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
from sb3_contrib import MaskablePPO

from training.dagger_dataset import CHAMPION
from training.imitation_dataset import digest
from training.kl_maskable_ppo import KLMaskablePPO
from training.train_ppo import ROOT


def test_kl_transaction_resume_and_reference_integrity(tmp_path):
    if not CHAMPION.exists():
        pytest.skip("Frozen BC checkpoint unavailable")
    preset = json.loads((ROOT / "configs/ppo_candidate_raw_2m_seed42.json").read_text())
    reference_hash = digest(CHAMPION)
    preset.update({"target_total_steps": 8192, "device": "cpu", "n_envs": 1,
                   "n_steps": 4096, "base_seed": 1231,
                   "validation_interval": 1000000, "milestone_interval": 1000000,
                   "step0_baseline": False,
                   "initial_model": {"path": str(CHAMPION), "sha256": reference_hash},
                   "kl_reference": {"path": str(CHAMPION), "sha256": reference_hash,
                                    "beta": 0.02}})
    config = tmp_path / "preset.json"
    config.write_text(json.dumps(preset))
    run = tmp_path / "kl_run"
    base = [sys.executable, "-m", "training.train_vector_transaction", "--run-dir", str(run)]
    first = subprocess.run(base + ["--config-file", str(config), "--max-tasks", "1"],
                           cwd=ROOT, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert json.loads((run / "transaction_state.json").read_text())["committed_steps"] == 4096
    second = subprocess.run(base + ["--resume", "--config-file", str(config)],
                            cwd=ROOT, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    state = json.loads((run / "transaction_state.json").read_text())
    assert state["status"] == "completed" and state["committed_steps"] == 8192
    rows = list(csv.DictReader((run / "training_metrics.csv").open()))
    assert len(rows) == 2 and all(row["bc_kl"] for row in rows)
    assert all(row["explained_variance"] for row in rows)
    model = KLMaskablePPO.load(str(run / "final/model.zip"), device="cpu")
    assert model.kl_beta == 0.02 and model.policy.optimizer.state
    assert model.reference_policy is None
    assert digest(CHAMPION) == reference_hash
