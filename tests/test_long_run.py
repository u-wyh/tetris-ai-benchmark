"""Long-run retention, schedule, evaluation recovery and trajectory isolation."""

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

from training.evaluation.evaluate import aggregate, best_key
import training.long_run as long_run
from training.long_run import (create_milestone, crossings, evaluate_atomic, evaluation_path,
                               post_commit, prune_transactions, sync_best)
from training.train_ppo import atomic_json
from training.train_transaction import task_name
from training.train_vector_transaction import vector_config

ROOT = Path(__file__).resolve().parents[1]


def test_target_rounding_and_seed_sets():
    preset = json.loads((ROOT / "configs" / "ppo_raw_10m_seed42.json").read_text())
    assert preset["target_total_steps"] == 10_000_000
    assert preset["step0_baseline"] is True
    assert (preset["device"], preset["n_envs"], preset["n_steps"], preset["task_steps"]) == (
        "cuda", 8, 512, 4096)
    assert (preset["transaction_retention"], preset["milestone_interval"],
            preset["validation_interval"]) == (3, 1_000_000, 250_000)
    config = vector_config(target_steps=10_000_000)
    assert config["requested_target_steps"] == 10_000_000
    assert config["actual_target_committed_steps"] == 10_002_432
    assert config["ppo"]["n_steps"] == 512
    assert len(config["validation_seed_set"]) == 32
    assert len(config["final_test_seed_set"]) == 100
    assert not set(config["worker_seeds"]) & set(config["validation_seed_set"])
    assert not set(config["validation_seed_set"]) & set(config["final_test_seed_set"])


def test_threshold_crossing_once():
    assert list(crossings(999424, 1003520, 1000000)) == [1000000]
    assert list(crossings(1003520, 1007616, 1000000)) == []
    assert list(crossings(249856, 253952, 250000)) == [250000]
    assert list(crossings(253952, 258048, 250000)) == []


def test_inherited_checkpoint_longlife_runs_separately(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(long_run, "evaluate_atomic",
                        lambda *args, **kwargs: calls.append(args[3]) or {})
    monkeypatch.setattr(long_run, "create_milestone", lambda *args: None)
    monkeypatch.setattr(long_run, "sync_best", lambda *args: None)
    monkeypatch.setattr(long_run, "rebuild_validation_summary", lambda *args: None)
    monkeypatch.setattr(long_run, "prune_transactions", lambda *args: None)
    config = {"task_steps": 4096, "validation_interval": 4096,
              "milestone_interval": 4096, "pinned_tasks": [489],
              "longlife_validation": {"seeds": 16, "max_pieces": 20000}}
    for task in (489, 490):
        state = {"committed_task": task, "committed_steps": task * 4096}
        post_commit(tmp_path, state, config, {}, lambda *args: None)
    assert calls == ["periodic", "milestone", "periodic", "milestone", "longlife"]


def test_non_aligned_milestone_is_independent(tmp_path):
    source = tmp_path / "committed" / "task_000245"
    source.mkdir(parents=True)
    (source / "model.zip").write_bytes(b"committed model")
    config = {"task_steps": 4096, "device": "cuda", "ppo": {"n_envs": 8, "n_steps": 512},
              "run_seed": 42}
    milestone = create_milestone(tmp_path, source, 1_000_000, 1_003_520, config,
                                 {"git_commit": "test"}, fake_result(1_003_520, 1, 0, 0))
    shutil.rmtree(source)
    metadata = json.loads((milestone / "metadata.json").read_text())
    assert metadata["requested_milestone"] == 1_000_000
    assert metadata["actual_committed_steps"] == 1_003_520
    assert metadata["task_id"] == 245
    assert (milestone / "model.zip").read_bytes() == b"committed model"


def test_retention_after_ten_commits_preserves_milestone_and_best(tmp_path):
    run_dir = tmp_path
    config = {"transaction_retention": 3}
    (run_dir / "committed").mkdir()
    (run_dir / "milestones" / "step_000008192").mkdir(parents=True)
    (run_dir / "milestones" / "step_000008192" / "model.zip").write_bytes(b"milestone")
    (run_dir / "best_versions" / "step_000008192").mkdir(parents=True)
    (run_dir / "best_versions" / "step_000008192" / "model.zip").write_bytes(b"best")
    (run_dir / "best").symlink_to("best_versions/step_000008192", target_is_directory=True)
    for number in range(1, 11):
        directory = run_dir / "committed" / task_name(number)
        directory.mkdir()
        (directory / "model.zip").write_bytes(str(number).encode())
        atomic_json(run_dir / "transaction_state.json",
                    {"committed_task": number, "committed_steps": number * 4096})
        prune_transactions(run_dir, {"committed_task": number}, config,
                           lambda path, task, _: (path / "model.zip").read_bytes() == str(task).encode()
                           or pytest.fail("Invalid retained checkpoint"))
        assert (directory / "model.zip").read_bytes() == str(number).encode()
    assert sorted(path.name for path in (run_dir / "committed").iterdir()) == [
        "task_000008", "task_000009", "task_000010"]
    assert (run_dir / "milestones" / "step_000008192" / "model.zip").read_bytes() == b"milestone"
    assert (run_dir / "best" / "model.zip").read_bytes() == b"best"


def fake_result(step, pieces, lines, score, protocol="periodic"):
    return {"protocol": protocol, "committed_steps": step,
            "model_checkpoint": f"committed/{task_name(step // 4096)}/model.zip",
            "num_seeds": 1, "max_pieces": 5, "seeds": [100000],
            "per_seed": [{"seed": 100000, "pieces_survived": pieces, "lines": lines,
                          "score": score, "episode_reward": 0.0,
                          "game_over": True, "survived_cap": False}],
            "aggregate": {"mean_pieces": pieces, "median_pieces": pieces,
                          "mean_lines": lines, "median_lines": lines,
                          "mean_score": score, "median_score": score,
                          "mean_reward": 0, "survival_cap_rate": 0,
                          "game_over_rate": 1, "p90_pieces": pieces,
                          "best_pieces": pieces, "evaluation_saturated": False,
                          "metric_censored_by_cap": False}, "wall_seconds": 0.1}


def test_best_lexicographic_and_atomic_pointer(tmp_path):
    run_dir = tmp_path
    (run_dir / "committed").mkdir()
    (run_dir / "evaluations" / "validation").mkdir(parents=True)
    atomic_json(run_dir / "config.json", {"task_steps": 4096})
    cases = [(4096, 100, 2, 100), (8192, 120, 0, 0), (12288, 110, 99, 99),
             (16384, 120, 1, 0), (20480, 120, 1, 99)]
    for step, pieces, lines, score in cases:
        source = run_dir / "committed" / task_name(step // 4096)
        source.mkdir()
        (source / "model.zip").write_bytes(str(step).encode())
        atomic_json(evaluation_path(run_dir, step, "periodic"),
                    fake_result(step, pieces, lines, score))
        sync_best(run_dir)
        expected = max(cases[:step // 4096], key=lambda row: (row[1], row[2], row[3]))[0]
        assert (run_dir / "best" / "model.zip").read_bytes() == str(expected).encode()
    assert best_key(fake_result(0, 120, 1, 99)) == (120, 1, 99)
    assert len(list((run_dir / "best_versions").iterdir())) == 1


def test_failed_best_pointer_swap_preserves_previous_model(tmp_path, monkeypatch):
    run_dir = tmp_path
    (run_dir / "committed").mkdir()
    (run_dir / "evaluations" / "validation").mkdir(parents=True)
    atomic_json(run_dir / "config.json", {"task_steps": 4096})
    for step, pieces in ((4096, 100), (8192, 120)):
        source = run_dir / "committed" / task_name(step // 4096)
        source.mkdir()
        (source / "model.zip").write_bytes(str(step).encode())
        atomic_json(evaluation_path(run_dir, step, "periodic"),
                    fake_result(step, pieces, 0, 0))
        if step == 4096:
            sync_best(run_dir)
    replace = long_run.os.replace

    def interrupt(source, destination):
        if str(destination) == str(run_dir / "best"):
            raise OSError("injected pointer interruption")
        return replace(source, destination)

    monkeypatch.setattr(long_run.os, "replace", interrupt)
    with pytest.raises(OSError, match="pointer interruption"):
        sync_best(run_dir)
    assert (run_dir / "best" / "model.zip").read_bytes() == b"4096"
    monkeypatch.setattr(long_run.os, "replace", replace)
    sync_best(run_dir)
    assert (run_dir / "best" / "model.zip").read_bytes() == b"8192"


def test_best_uses_one_comparable_validation_protocol(tmp_path):
    run_dir = tmp_path
    (run_dir / "committed").mkdir()
    (run_dir / "evaluations" / "validation").mkdir(parents=True)
    atomic_json(run_dir / "config.json", {"task_steps": 4096})
    for step, pieces in ((4096, 100), (8192, 110)):
        source = run_dir / "committed" / task_name(step // 4096)
        source.mkdir()
        (source / "model.zip").write_bytes(str(step).encode())
        atomic_json(evaluation_path(run_dir, step, "periodic"),
                    fake_result(step, pieces, 0, 0))
    atomic_json(evaluation_path(run_dir, 8192, "milestone"),
                fake_result(8192, 10000, 0, 0, protocol="milestone"))
    sync_best(run_dir)
    assert (run_dir / "best" / "model.zip").read_bytes() == b"8192"
    assert json.loads((run_dir / "best" / "metadata.json").read_text())[
        "evaluation_path"].endswith("_periodic.json")


def test_evaluation_cap_and_interruption(tmp_path, monkeypatch):
    rows = [{"seed": 1, "pieces_survived": 5, "lines": 2, "score": 10,
             "episode_reward": 1.0, "game_over": False, "survived_cap": True},
            {"seed": 2, "pieces_survived": 3, "lines": 0, "score": 0,
             "episode_reward": -1.0, "game_over": True, "survived_cap": False}]
    stats = aggregate(rows, 5)
    assert stats["evaluation_saturated"] and stats["metric_censored_by_cap"]
    assert stats["survival_cap_rate"] == stats["game_over_rate"] == 0.5
    run_dir = tmp_path
    model = run_dir / "committed" / "task_000001" / "model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake")
    calls = 0

    def flaky(command, **kwargs):
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--output") + 1])
        if calls == 1:
            output.write_text("incomplete")
            raise subprocess.CalledProcessError(1, command)
        result = fake_result(4096, 5, 2, 10)
        result["per_seed"][0].update({"episode_reward": 1.0,
                                      "game_over": False, "survived_cap": True})
        result["aggregate"] = aggregate(result["per_seed"], 5)
        atomic_json(output, result)

    monkeypatch.setattr(subprocess, "run", flaky)
    with pytest.raises(subprocess.CalledProcessError):
        evaluate_atomic(run_dir, model, 4096, "periodic", 1, 5, {})
    assert not evaluation_path(run_dir, 4096, "periodic").exists()
    assert model.read_bytes() == b"fake"
    result = evaluate_atomic(run_dir, model, 4096, "periodic", 1, 5, {})
    assert calls == 2 and result["num_seeds"] == 1
    assert evaluation_path(run_dir, 4096, "periodic").exists()


def test_resume_retries_missing_evaluation_before_retention(tmp_path, monkeypatch):
    run_dir = tmp_path
    config = {"task_steps": 4096, "transaction_retention": 3,
              "validation_interval": 4096, "milestone_interval": 1000000,
              "periodic_validation_seeds": 1, "periodic_validation_max_pieces": 5}
    atomic_json(run_dir / "config.json", config)
    state = {"committed_task": 4, "committed_steps": 16384}
    atomic_json(run_dir / "transaction_state.json", state)
    for number in range(1, 5):
        directory = run_dir / "committed" / task_name(number)
        directory.mkdir(parents=True)
        (directory / "model.zip").write_bytes(str(number).encode())
    calls = 0

    def flaky(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, command)
        output = Path(command[command.index("--output") + 1])
        atomic_json(output, fake_result(16384, 5, 1, 10))

    monkeypatch.setattr(subprocess, "run", flaky)
    verify = lambda path, number, _: (path / "model.zip").is_file() or pytest.fail("Missing model")
    with pytest.raises(subprocess.CalledProcessError):
        post_commit(run_dir, state, config, {}, verify)
    assert (run_dir / "committed" / "task_000001").exists()
    assert json.loads((run_dir / "transaction_state.json").read_text()) == state
    assert not evaluation_path(run_dir, 16384, "periodic").exists()
    post_commit(run_dir, state, config, {}, verify)
    assert calls == 2
    assert sorted(path.name for path in (run_dir / "committed").iterdir()) == [
        "task_000002", "task_000003", "task_000004"]
    assert (run_dir / "best" / "model.zip").read_bytes() == b"4"
    result = json.loads(evaluation_path(run_dir, 16384, "periodic").read_text())
    assert result["requested_validation_step"] == 16384
    assert result["actual_committed_steps"] == 16384


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA trajectory comparison")
def test_validation_does_not_change_training_trajectory(tmp_path):
    def launch(name, interval):
        preset = tmp_path / f"{name}.json"
        atomic_json(preset, {"target_total_steps": 12288, "device": "cuda", "n_envs": 8,
                             "n_steps": 512, "task_steps": 4096, "base_seed": 42,
                             "transaction_retention": 3, "milestone_interval": 1000000,
                             "validation_interval": interval,
                             "periodic_validation_seeds": 1,
                             "periodic_validation_max_pieces": 3,
                             "milestone_validation_seeds": 1,
                             "milestone_validation_max_pieces": 3})
        run_dir = tmp_path / name
        result = subprocess.run([sys.executable, "-m", "training.train_vector_transaction",
                                 "--run-dir", str(run_dir), "--config-file", str(preset)],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return run_dir

    baseline = launch("without_validation", 250000)
    validated = launch("with_validation", 4096)
    assert len(list((validated / "evaluations" / "validation").glob("*.json"))) == 3
    final_a = baseline / "committed" / "task_000003"
    final_b = validated / "committed" / "task_000003"
    model_a = MaskablePPO.load(str(final_a / "model.zip"), device="cuda")
    model_b = MaskablePPO.load(str(final_b / "model.zip"), device="cuda")
    assert model_a.num_timesteps == model_b.num_timesteps == 12288
    for key, value in model_a.policy.state_dict().items():
        assert torch.equal(value, model_b.policy.state_dict()[key]), key
    state_a = model_a.policy.optimizer.state_dict()
    state_b = model_b.policy.optimizer.state_dict()
    assert state_a["param_groups"] == state_b["param_groups"]
    for key, state in state_a["state"].items():
        for field, value in state.items():
            other = state_b["state"][key][field]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    trainer_a = pickle.loads((final_a / "trainer_state.pkl").read_bytes())
    trainer_b = pickle.loads((final_b / "trainer_state.pkl").read_bytes())
    assert trainer_a["python_random"] == trainer_b["python_random"]
    assert np.array_equal(trainer_a["numpy_random"][1], trainer_b["numpy_random"][1])
    assert torch.equal(trainer_a["torch_cpu_random"], trainer_b["torch_cpu_random"])
    assert all(torch.equal(a, b) for a, b in zip(trainer_a["torch_cuda_random"],
                                                trainer_b["torch_cuda_random"]))
    assert (final_a / "vector_env_state.pkl").read_bytes() == (
        final_b / "vector_env_state.pkl").read_bytes()
