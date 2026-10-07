"""CUDA vector transaction recovery, integrity and deterministic continuation."""

import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from training.env import TetrisEnv
from training.train_vector_transaction import TransactionTetrisEnv, restore_task, vector_config
from training.vector_env import make_vector_env

ROOT = Path(__file__).resolve().parents[1]


def invoke(run_dir, *arguments, succeeds=True):
    result = subprocess.run(
        [sys.executable, "-m", "training.train_vector_transaction", "--run-dir", str(run_dir),
         *arguments], cwd=ROOT, capture_output=True, text=True)
    if succeeds:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def dead_workers(pids_file):
    for pid in json.loads(pids_file.read_text())["workers"]:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_vector_config_and_full_environment_state_roundtrip():
    config = vector_config()
    assert config["device"] == "cuda"
    assert config["ppo"]["n_envs"] == 8
    assert config["ppo"]["n_steps"] == 512
    assert config["task_steps"] == 4096
    assert config["worker_seeds"] == list(range(42, 50))
    first, second = TetrisEnv(), TetrisEnv()
    try:
        first.reset(seed=42)
        for _ in range(3):
            first.step(int(np.argmax(first.action_masks())))
        state = first.get_state()
        observation, mask = first.get_observation(), first.action_masks()
        second.reset(seed=999)
        second.set_state(state)
        assert np.array_equal(second.get_observation(), observation)
        assert np.array_equal(second.action_masks(), mask)
        assert second.core.board == first.core.board
        assert second.core.current == first.core.current
        assert second.core.queue == first.core.queue
        assert second.core.bag == first.core.bag
        assert second.core.rng.state == first.core.rng.state
        assert second.episode_count == first.episode_count
        assert second.pieces == first.pieces
        action = int(np.argmax(mask))
        observation_a, reward_a, terminated_a, truncated_a, info_a = first.step(action)
        observation_b, reward_b, terminated_b, truncated_b, info_b = second.step(action)
        assert np.array_equal(observation_a, observation_b)
        assert (reward_a, terminated_a, truncated_a, info_a) == (
            reward_b, terminated_b, truncated_b, info_b)
    finally:
        first.close()
        second.close()


def test_interrupted_vector_task_reexecutes_identically(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is needed for the intended transaction comparison")
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    invoke(baseline, "--target-steps", "12288")
    dead_workers(baseline / "logs" / "worker_pids.json")
    interrupted = invoke(resumed, "--target-steps", "12288", "--crash-during-task", "2",
                         succeeds=False)
    assert "Injected interruption" in interrupted.stderr
    crashed_pids = json.loads((resumed / "logs" / "worker_pids.json").read_text())["workers"]
    dead_workers(resumed / "logs" / "worker_pids.json")
    pointer = json.loads((resumed / "transaction_state.json").read_text())
    assert pointer["committed_task"] == 1 and pointer["committed_steps"] == 4096
    working = resumed / "working" / "task_000002.tmp"
    assert working.exists()
    config = json.loads((resumed / "config.json").read_text())
    env = make_vector_env(8, 42, env_class=TransactionTetrisEnv)
    try:
        model, trainer, workers = restore_task(resumed, 1, config, env)
        assert model.num_timesteps == 4096 and len(workers) == 8
        assert len(model.policy.optimizer.state) > 0
        assert trainer["torch_cuda_random"] is not None
    finally:
        env.close()
    # Case B: all files exist, but neither working nor an orphaned renamed Task is official.
    task1 = resumed / "committed" / "task_000001"
    shutil.copytree(task1, working, dirs_exist_ok=True)
    shutil.copytree(task1, resumed / "committed" / "task_000002")
    invoke(resumed, "--resume")
    dead_workers(resumed / "logs" / "worker_pids.json")
    for pid in crashed_pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    pointer = json.loads((resumed / "transaction_state.json").read_text())
    assert pointer["status"] == "completed"
    assert pointer["committed_task"] == 3 and pointer["committed_steps"] == 12288
    assert len(list((resumed / "abandoned").iterdir())) == 2
    with (resumed / "training_metrics.csv").open() as file:
        metrics = list(csv.DictReader(file))
    assert [int(row["task_id"]) for row in metrics] == [1, 2, 3]
    assert [int(row["end_step"]) for row in metrics] == [4096, 8192, 12288]
    events = (resumed / "events.log").read_text()
    assert "ABANDONED task=task_000002.tmp" in events
    assert "ABANDONED task=task_000002." in events
    assert "RESUMED committed_task=1 committed_steps=4096" in events
    assert (resumed / "final" / "model.zip").exists()

    first = MaskablePPO.load(str(baseline / "final" / "model.zip"), device="cuda")
    second = MaskablePPO.load(str(resumed / "final" / "model.zip"), device="cuda")
    assert first.num_timesteps == second.num_timesteps == 12288
    for name, value in first.policy.state_dict().items():
        assert torch.equal(value, second.policy.state_dict()[name]), name
    assert first.policy.optimizer.state_dict()["param_groups"] == second.policy.optimizer.state_dict()["param_groups"]
    for key, state in first.policy.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            other = second.policy.optimizer.state_dict()["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    trainer_a = pickle.loads((baseline / "committed/task_000003/trainer_state.pkl").read_bytes())
    trainer_b = pickle.loads((resumed / "committed/task_000003/trainer_state.pkl").read_bytes())
    assert trainer_a["python_random"] == trainer_b["python_random"]
    assert np.array_equal(trainer_a["numpy_random"][1], trainer_b["numpy_random"][1])
    assert torch.equal(trainer_a["torch_cpu_random"], trainer_b["torch_cpu_random"])
    assert all(torch.equal(a, b) for a, b in zip(trainer_a["torch_cuda_random"],
                                                trainer_b["torch_cuda_random"]))
    assert (baseline / "committed/task_000003/vector_env_state.pkl").read_bytes() == (
        resumed / "committed/task_000003/vector_env_state.pkl").read_bytes()


def test_corrupt_committed_task_is_rejected(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is needed for the intended transaction check")
    run_dir = tmp_path / "corrupt"
    invoke(run_dir, "--target-steps", "12288", "--max-tasks", "2")
    checkpoint = run_dir / "committed/task_000002/model.zip"
    with checkpoint.open("ab") as file:
        file.write(b"corrupt")
    result = invoke(run_dir, "--resume", succeeds=False)
    assert "checksum failed" in result.stderr
    pointer = json.loads((run_dir / "transaction_state.json").read_text())
    assert pointer["committed_task"] == 2 and pointer["committed_steps"] == 8192
    assert not (run_dir / "committed/task_000003").exists()
    dead_workers(run_dir / "logs" / "worker_pids.json")
