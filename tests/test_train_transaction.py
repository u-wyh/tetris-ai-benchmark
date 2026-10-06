"""Real short Task/Commit/rollback test with model, environment and RNG recovery."""

import csv
import json
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.env import TetrisEnv
from training.train_transaction import config_for_tasks, load_committed, run

ROOT = Path(__file__).resolve().parents[1]


def invoke(run_dir, *arguments):
    subprocess.run([sys.executable, "-m", "training.train_transaction", "--run-dir", str(run_dir),
                    *arguments], cwd=ROOT, capture_output=True, text=True, check=True)


def test_default_task_boundaries():
    config = config_for_tasks()
    assert config["task_steps"] == 4096
    assert config["target_total_steps"] == 32768
    assert config["ppo"]["n_steps"] == 1024
    assert config["ppo"]["n_envs"] == 1
    assert config["device"] == "cpu"


def test_resume_rejects_device_change_before_loading_checkpoint(tmp_path):
    run_dir = tmp_path / "recorded_cpu"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config_for_tasks(device="cpu")))
    (run_dir / "metadata.json").write_text(json.dumps({"device": "cpu"}))
    (run_dir / "transaction_state.json").write_text(json.dumps({"status": "running"}))
    import pytest
    with pytest.raises(ValueError, match="created for device=cpu"):
        run(run_dir, resume=True, device="cuda")


def test_resume_never_falls_back_from_recorded_cuda(tmp_path, monkeypatch):
    run_dir = tmp_path / "recorded_cuda"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config_for_tasks(device="cuda")))
    (run_dir / "metadata.json").write_text(json.dumps({"device": "cuda"}))
    (run_dir / "transaction_state.json").write_text(json.dumps({"status": "running", "committed_task": 0}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import pytest
    with pytest.raises(RuntimeError, match="Recorded CUDA device is unavailable"):
        run(run_dir, resume=True)


def test_rollback_restores_last_commit_and_reexecutes_task(tmp_path):
    resumed = tmp_path / "resumed"
    baseline = tmp_path / "baseline"
    invoke(resumed, "--task-steps", "1024", "--target-steps", "3072", "--max-tasks", "2")
    pointer = json.loads((resumed / "transaction_state.json").read_text())
    assert pointer["committed_task"] == 2 and pointer["committed_steps"] == 2048
    task2 = resumed / "committed" / "task_000002"
    assert all((task2 / name).exists() for name in
               ("model.zip", "trainer_state.pkl", "env_state.pkl", "metrics.json", "task_meta.json"))
    assert json.loads((task2 / "task_meta.json").read_text())["device"] == "cpu"
    assert json.loads((resumed / "metadata.json").read_text())["device"] == "cpu"
    env = TetrisEnv()
    model, trainer, saved_env = load_committed(MaskablePPO, resumed, 2, env, "cpu")
    assert model.num_timesteps == 2048 and len(model.policy.optimizer.state) > 0
    assert env.core.board == saved_env["core"].board
    assert env.core.current == saved_env["core"].current
    assert env.core.queue == saved_env["core"].queue
    assert env.core.rng.state == saved_env["core"].rng.state
    assert env.episode_seed == saved_env["episode_seed"]
    assert env.pieces == saved_env["pieces"]
    assert random.getstate() == trainer["python_random"]
    assert np.array_equal(np.random.get_state()[1], trainer["numpy_random"][1])
    assert torch.equal(torch.get_rng_state(), trainer["torch_cpu_random"])
    env.close()

    # Simulate an interrupted Task 3 and the narrow rename-before-pointer crash window.
    working = resumed / "working" / "task_000003.tmp"
    working.mkdir()
    (working / "metrics.json").write_text('{"task_id": 3, "end_step": 999999, "fake": true}')
    (working / "progress.json").write_text('{"working_steps": 2500}')
    shutil.copy2(task2 / "model.zip", working / "model.zip")
    shutil.copytree(task2, resumed / "committed" / "task_000003")
    invoke(resumed, "--resume")
    final_pointer = json.loads((resumed / "transaction_state.json").read_text())
    assert final_pointer["status"] == "completed"
    assert final_pointer["committed_task"] == 3
    assert final_pointer["committed_steps"] == 3072
    assert len(list((resumed / "abandoned").iterdir())) == 2
    with (resumed / "training_metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["end_step"]) for row in rows] == [1024, 2048, 3072]
    assert rows[2]["start_step"] == "2048" and rows[2]["task_id"] == "3"
    assert (resumed / "final" / "model.zip").exists()
    assert "ABANDONED task=task_000003.tmp" in (resumed / "events.log").read_text()
    assert "RESUMED committed_task=2 committed_steps=2048" in (resumed / "events.log").read_text()

    # CPU continuation is bit-identical to an uninterrupted three-Task run.
    invoke(baseline, "--task-steps", "1024", "--target-steps", "3072")
    continued = MaskablePPO.load(str(resumed / "committed" / "task_000003" / "model.zip"), device="cpu")
    uninterrupted = MaskablePPO.load(str(baseline / "committed" / "task_000003" / "model.zip"), device="cpu")
    assert all(torch.equal(value, uninterrupted.policy.state_dict()[key])
               for key, value in continued.policy.state_dict().items())
    assert len(continued.policy.optimizer.state) > 0
    assert pickle.loads((resumed / "committed" / "task_000003" / "trainer_state.pkl").read_bytes())["num_timesteps"] == 3072
