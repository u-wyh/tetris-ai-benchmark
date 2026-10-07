"""Step-0 validation and resume use the same initial model as uninterrupted PPO."""

import csv
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from training.train_ppo import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def run_trainer(run_dir, *args):
    result = subprocess.run([sys.executable, "-m", "training.train_vector_transaction",
                             "--run-dir", str(run_dir), *args], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def assert_equal_committed(left, right):
    model_a = MaskablePPO.load(str(left / "model.zip"), device="cpu")
    model_b = MaskablePPO.load(str(right / "model.zip"), device="cpu")
    assert model_a.num_timesteps == model_b.num_timesteps == 4096
    for key, value in model_a.policy.state_dict().items():
        assert torch.equal(value, model_b.policy.state_dict()[key]), key
    opt_a = model_a.policy.optimizer.state_dict()
    opt_b = model_b.policy.optimizer.state_dict()
    assert opt_a["param_groups"] == opt_b["param_groups"]
    assert opt_a["state"].keys() == opt_b["state"].keys()
    for key, state in opt_a["state"].items():
        for field, value in state.items():
            other = opt_b["state"][key][field]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert (left / "vector_env_state.pkl").read_bytes() == (right / "vector_env_state.pkl").read_bytes()
    rng_a = pickle.loads((left / "trainer_state.pkl").read_bytes())
    rng_b = pickle.loads((right / "trainer_state.pkl").read_bytes())
    assert rng_a["python_random"] == rng_b["python_random"]
    assert np.array_equal(rng_a["numpy_random"][1], rng_b["numpy_random"][1])
    assert torch.equal(rng_a["torch_cpu_random"], rng_b["torch_cpu_random"])
    assert all(torch.equal(a, b) for a, b in zip(rng_a["torch_cuda_random"],
                                                rng_b["torch_cuda_random"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Step-0 trajectory test")
def test_step0_baseline_resume_matches_uninterrupted_training(tmp_path):
    config = {"target_total_steps": 4096, "device": "cuda", "n_envs": 8,
              "n_steps": 512, "task_steps": 4096, "base_seed": 42,
              "transaction_retention": 3, "milestone_interval": 8192,
              "validation_interval": 8192, "periodic_validation_seeds": 1,
              "periodic_validation_max_pieces": 3,
              "milestone_validation_seeds": 1, "milestone_validation_max_pieces": 3,
              "step0_baseline": True}
    config_path = tmp_path / "with_baseline.json"
    atomic_json(config_path, config)
    run_dir = tmp_path / "baseline"
    run_trainer(run_dir, "--config-file", str(config_path), "--max-tasks", "0")
    state = json.loads((run_dir / "transaction_state.json").read_text())
    assert state["committed_task"] == state["committed_steps"] == 0
    assert state["status"] == "awaiting_resume"
    baseline = run_dir / "baseline"
    model = MaskablePPO.load(str(baseline / "model.zip"), device="cpu")
    assert model.num_timesteps == 0
    assert not model.policy.optimizer.state
    baseline_meta = json.loads((baseline / "metadata.json").read_text())
    assert baseline_meta["training_steps"] == 0 and baseline_meta["seed"] == 42
    assert baseline_meta["git_commit"]
    assert len(baseline_meta["initial_worker_digests"]) == 8
    result = json.loads((baseline / "evaluation.json").read_text())
    assert result["evaluation_type"] == "baseline"
    assert result["training_steps"] == result["committed_steps"] == 0
    assert result["num_seeds"] == 1 and result["max_pieces"] == 3
    assert len(result["per_seed"]) == 1
    with (run_dir / "evaluations" / "validation_summary.csv").open() as file:
        assert [int(row["committed_steps"]) for row in csv.DictReader(file)] == [0]
    assert (run_dir / "best" / "model.zip").exists()
    # Simulate a shutdown after the durable model save but before validation publish.
    (baseline / "evaluation.json").unlink()
    (run_dir / "evaluations" / "validation" / "step_000000000_periodic.json").unlink()
    (run_dir / "best").unlink()
    shutil.rmtree(run_dir / "best_versions")
    run_trainer(run_dir, "--resume")
    assert (run_dir / "committed" / "task_000001").exists()
    assert (baseline / "model.zip").exists()
    assert (baseline / "evaluation.json").exists()

    config["step0_baseline"] = False
    control_path = tmp_path / "control.json"
    atomic_json(control_path, config)
    control = tmp_path / "control"
    run_trainer(control, "--config-file", str(control_path))
    assert_equal_committed(run_dir / "committed" / "task_000001",
                           control / "committed" / "task_000001")
