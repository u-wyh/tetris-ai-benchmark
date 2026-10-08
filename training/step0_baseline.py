"""Durable Step-0 baseline from the exact model that will enter PPO training."""

import copy
import hashlib
import json
import os
import pickle
import shutil

from training.long_run import evaluate_atomic, rebuild_validation_summary, sync_best
from training.train_ppo import atomic_bytes, atomic_json, now
from training.train_ppo import ROOT
from training.train_transaction import (fsync_dir, fsync_tree, global_rng_state,
                                        log_event, restore_global_rng)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_digests(env):
    return env.env_method("initial_state_digest")


def compare_initial_parameters(model, reference, label):
    """Verify that a reward-only run starts from the same random policy."""
    import torch
    from sb3_contrib import MaskablePPO

    if not reference.is_file():
        raise FileNotFoundError(f"{label} Step-0 reference is missing: {reference}")
    baseline = MaskablePPO.load(str(reference), device="cpu")
    actual, expected = model.policy.state_dict(), baseline.policy.state_dict()
    if (model.num_timesteps != 0 or baseline.num_timesteps != 0
            or actual.keys() != expected.keys()
            or any(not torch.equal(value.detach().cpu(), expected[key])
                   for key, value in actual.items())):
        raise RuntimeError(f"Shaped Step-0 parameters differ from {label} Step-0")
    return file_sha256(reference)


def compare_raw_initial_parameters(model, config):
    if not config.get("hole_penalty_coef", 0.0):
        return None
    return compare_initial_parameters(
        model, ROOT / "runs/ppo_raw_10m_seed42/baseline/model.zip", "Raw")


def compare_v1_initial_parameters(model, config):
    if config.get("hole_penalty_coef") != 0.02:
        return None
    return compare_initial_parameters(
        model, ROOT / "runs/ppo_hole_v1_2m_seed42_lambda010/baseline/model.zip",
        "Shaped V1")


def model_snapshot(model, env):
    return {"parameters": {key: value.detach().cpu().clone()
                           for key, value in model.policy.state_dict().items()},
            "optimizer": copy.deepcopy(model.policy.optimizer.state_dict()),
            "rng": global_rng_state(model), "workers": worker_digests(env),
            "pending_worker_seeds": list(env._seeds)}


def assert_unchanged(before, model, env):
    import numpy as np
    import torch

    after = model_snapshot(model, env)
    if (before["workers"] != after["workers"]
            or before["pending_worker_seeds"] != after["pending_worker_seeds"]):
        raise RuntimeError("Step-0 validation changed training worker state")
    if before["rng"]["num_timesteps"] != after["rng"]["num_timesteps"] or model.num_timesteps != 0:
        raise RuntimeError("Step-0 validation changed training timesteps")
    if any(not torch.equal(value, after["parameters"][key])
           for key, value in before["parameters"].items()):
        raise RuntimeError("Step-0 validation changed policy parameters")
    first_opt, second_opt = before["optimizer"], after["optimizer"]
    if first_opt["param_groups"] != second_opt["param_groups"] or first_opt["state"].keys() != second_opt["state"].keys():
        raise RuntimeError("Step-0 validation changed optimizer")
    for key, state in first_opt["state"].items():
        for field, value in state.items():
            other = second_opt["state"][key][field]
            if (not torch.equal(value, other) if isinstance(value, torch.Tensor) else value != other):
                raise RuntimeError("Step-0 validation changed optimizer")
    first, second = before["rng"], after["rng"]
    if (first["python_random"] != second["python_random"]
            or first["numpy_random"][0] != second["numpy_random"][0]
            or not np.array_equal(first["numpy_random"][1], second["numpy_random"][1])
            or first["numpy_random"][2:] != second["numpy_random"][2:]
            or not torch.equal(first["torch_cpu_random"], second["torch_cpu_random"])
            or (first["torch_cuda_random"] is None) != (second["torch_cuda_random"] is None)
            or (first["torch_cuda_random"] is not None and
                len(first["torch_cuda_random"]) != len(second["torch_cuda_random"]))
            or (first["torch_cuda_random"] is not None and
                any(not torch.equal(a, b) for a, b in zip(first["torch_cuda_random"], second["torch_cuda_random"])))):
        raise RuntimeError("Step-0 validation changed main RNG state")
    for field in ("sb3_last_obs", "sb3_last_episode_starts"):
        initial, current = first[field], second[field]
        if (initial is None) != (current is None) or (
                initial is not None and not np.array_equal(initial, current)):
            raise RuntimeError("Step-0 validation changed PPO rollout state")


def save_baseline(run_dir, model, env, config, run_metadata):
    if model.num_timesteps != 0:
        raise RuntimeError("Baseline model must have zero training timesteps")
    if list(env._seeds) != config["worker_seeds"]:
        raise RuntimeError("Initial worker seeds differ from formal config")
    reference_sha = compare_raw_initial_parameters(model, config)
    v1_reference_sha = compare_v1_initial_parameters(model, config)
    baseline = run_dir / "baseline"
    if baseline.exists():
        return baseline
    pending = run_dir / "baseline.tmp"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    model.save(str(pending / "model.zip"))
    rng = global_rng_state(model)
    atomic_bytes(pending / "trainer_state.pkl", pickle.dumps(rng, protocol=5))
    atomic_json(pending / "metadata.json", {
        "training_steps": 0, "seed": config["run_seed"],
        "git_commit": run_metadata["git_commit"], "ppo_config": config["ppo"],
        "policy": config["policy"], "network": config["network"],
        "observation_version": config["observation_version"],
        "action_space_version": config["action_space_version"],
        "reward_version": config["reward_version"], "reward_definition": config["reward"],
        "hole_penalty_coef": config.get("hole_penalty_coef", 0.0),
        "height_penalty_coef": config.get("height_penalty_coef", 0.0),
        "raw_step0_model_sha256": reference_sha,
        "v1_step0_model_sha256": v1_reference_sha,
        "initial_worker_digests": worker_digests(env),
        "pending_worker_seeds": list(env._seeds),
        "model_sha256": file_sha256(pending / "model.zip"),
        "trainer_state_sha256": file_sha256(pending / "trainer_state.pkl"),
        "created_at": now()})
    fsync_tree(pending)
    os.replace(pending, baseline)
    fsync_dir(run_dir)
    log_event(run_dir, "BASELINE_SAVED", steps=0)
    return baseline


def load_baseline(run_dir, env, config):
    from sb3_contrib import MaskablePPO

    baseline = run_dir / "baseline"
    meta = json.loads((baseline / "metadata.json").read_text())
    if (meta["seed"] != config["run_seed"] or meta["training_steps"] != 0
            or meta.get("reward_version") != config["reward_version"]
            or meta.get("hole_penalty_coef", 0.0) != config.get("hole_penalty_coef", 0.0)
            or meta.get("height_penalty_coef", 0.0) != config.get("height_penalty_coef", 0.0)
            or file_sha256(baseline / "model.zip") != meta["model_sha256"]
            or file_sha256(baseline / "trainer_state.pkl") != meta["trainer_state_sha256"]):
        raise RuntimeError("Step-0 baseline integrity failed")
    model = MaskablePPO.load(str(baseline / "model.zip"), env=env,
                             device=config["device"], force_reset=False)
    trainer = pickle.loads((baseline / "trainer_state.pkl").read_bytes())
    if trainer["num_timesteps"] != 0:
        raise RuntimeError("Step-0 baseline trainer state has training updates")
    restore_global_rng(trainer)
    if (model.num_timesteps != 0 or worker_digests(env) != meta["initial_worker_digests"]
            or list(env._seeds) != meta["pending_worker_seeds"]):
        raise RuntimeError("Restored Step-0 model or workers differ from baseline")
    return model


def evaluate_baseline(run_dir, model, env, config, run_metadata):
    baseline = save_baseline(run_dir, model, env, config, run_metadata)
    before = model_snapshot(model, env)
    result = evaluate_atomic(
        run_dir, baseline / "model.zip", 0, "periodic",
        config["periodic_validation_seeds"], config["periodic_validation_max_pieces"],
        config, extra={"evaluation_type": "baseline", "training_steps": 0,
                       "requested_validation_step": 0, "actual_committed_steps": 0})
    assert_unchanged(before, model, env)
    atomic_json(baseline / "evaluation.json", result)
    fsync_dir(baseline)
    sync_best(run_dir)
    rebuild_validation_summary(run_dir)
    log_event(run_dir, "BASELINE_VALIDATED", mean_pieces=result["aggregate"]["mean_pieces"])
    return result
