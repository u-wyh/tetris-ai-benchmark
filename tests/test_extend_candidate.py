"""A completed Candidate checkpoint continues without resetting PPO or workers."""

import json
import hashlib
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from training.extend_candidate import prepare
from training.train_vector_transaction import verify_task


def train(run_dir, preset, *flags, succeeds=True):
    command = [sys.executable, "-m", "training.train_vector_transaction",
               "--run-dir", str(run_dir)]
    if preset:
        command += ["--config-file", str(preset)]
    result = subprocess.run(command + list(flags), capture_output=True, text=True)
    assert (result.returncode == 0) == succeeds, result.stderr
    return result


def assert_same(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif hasattr(left, "__dict__"):
        assert_same(vars(left), vars(right))
    else:
        assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA continuation contract")
@pytest.mark.parametrize("full_context", [False, True])
def test_completed_candidate_extension_and_interrupted_resume(tmp_path, full_context):
    source = tmp_path / "source"
    extended = tmp_path / "extended"
    control = tmp_path / "control"
    preset = {"device": "cuda", "n_envs": 8, "n_steps": 512, "task_steps": 4096,
              "base_seed": 42, "candidate_policy": True, "step0_baseline": False}
    if full_context:
        # Exercise the same explicit LR transition as the current experiment,
        # then extend its completed run without restoring the original again.
        origin = tmp_path / "origin"
        origin_preset = tmp_path / "origin.json"
        preset.update(candidate_feature_backend="bitboard", fast_candidate_env=True)
        origin_preset.write_text(json.dumps(dict(preset, target_total_steps=4096)))
        train(origin, origin_preset)
        files = ["config.json", "metadata.json", "committed/task_000001/manifest.json"]
        preset.update(ppo_overrides={"learning_rate": 0.00003}, initial_context={
            "run_dir": str(origin), "task": 1,
            "files_sha256": {name: hashlib.sha256((origin / name).read_bytes()).hexdigest()
                             for name in files},
            "learning_rate_after_restore": 0.00003,
        })
    short = tmp_path / "short.json"
    full = tmp_path / "full.json"
    short.write_text(json.dumps(dict(preset, target_total_steps=4096)))
    full.write_text(json.dumps(dict(preset, target_total_steps=12288)))
    train(source, short)
    train(control, full)
    original = (source / "transaction_state.json").read_bytes()
    prepare(source, extended, 12288, evaluate_baseline=False)
    assert (source / "transaction_state.json").read_bytes() == original
    config = json.loads((extended / "config.json").read_text())
    assert config["pinned_tasks"] == [1]
    verify_task(extended / "committed/task_000001", 1, config)
    train(extended, None, "--resume", "--crash-during-task", "2", succeeds=False)
    state = json.loads((extended / "transaction_state.json").read_text())
    assert state["committed_steps"] == 4096
    train(extended, None, "--resume")
    assert (extended / "committed/task_000001").exists()
    assert (extended / "continuation.json").exists()
    assert json.loads((extended / "transaction_state.json").read_text())["committed_steps"] == 12288
    a = MaskablePPO.load(str(extended / "committed/task_000003/model.zip"), device="cuda")
    b = MaskablePPO.load(str(control / "committed/task_000003/model.zip"), device="cuda")
    assert a.num_timesteps == b.num_timesteps == 12288
    assert a._n_updates == b._n_updates
    assert a.policy.optimizer.param_groups[0]["lr"] == b.policy.optimizer.param_groups[0]["lr"]
    if full_context:
        assert config["source_steps_offset"] == 4096
        task = json.loads((extended / "committed/task_000003/task_meta.json").read_text())
        assert task["source_steps_offset"] == 4096
        assert task["lifetime_steps"] == 16384
    for name, tensor in a.policy.state_dict().items():
        assert torch.equal(tensor, b.policy.state_dict()[name]), name
    for file in ("trainer_state.pkl", "vector_env_state.pkl"):
        assert_same(pickle.loads((extended / "committed/task_000003" / file).read_bytes()),
                    pickle.loads((control / "committed/task_000003" / file).read_bytes()))
    assert len(a.policy.optimizer.state) > 0
    for parameter_a, parameter_b in zip(a.policy.parameters(), b.policy.parameters()):
        for key, value in a.policy.optimizer.state[parameter_a].items():
            other = b.policy.optimizer.state[parameter_b][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    config["ppo"]["learning_rate"] *= 2
    (extended / "config.json").write_text(json.dumps(config))
    rejected = train(extended, None, "--resume", succeeds=False)
    assert "Continuation settings" in rejected.stderr
