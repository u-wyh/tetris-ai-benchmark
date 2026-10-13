"""Actor-only mixed BC and learner-state DAgger supervised experiment."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.imitation_bc import batch, metrics, records
from training.imitation_dataset import digest, TEACHER_HASH
from training.train_ppo import ROOT, atomic_json, now

CHAMPION = ROOT / "runs/candidate_bc_next_stage/champion/model.zip"
ORIGINAL = ROOT / "runs/imitation_v2_hold_20m/data"


def seed_list(directory):
    return sorted(int(path.stem.split("_")[1]) for path in Path(directory).glob("seed_*.json"))


def sample_original(limit, seed):
    """Reservoir across whole training episodes; never use BC validation seeds."""
    rng = random.Random(seed)
    sample = []
    for index, row in enumerate(records(ORIGINAL, range(600000, 600100))):
        if index < limit:
            sample.append(row)
        else:
            place = rng.randrange(index + 1)
            if place < limit:
                sample[place] = row
    return sample


def unique_dagger(directory, seeds):
    seen = set()
    result = []
    for row in records(Path(directory), seeds):
        if row.get("source") != "dagger_bc_visited" or row["learner_sha256"] != digest(CHAMPION):
            raise RuntimeError("DAgger source model identity mismatch")
        if row["state_hash"] not in seen:
            seen.add(row["state_hash"])
            result.append(row)
    return result


def train(dagger_data, dagger_validation, output, epochs=2, samples_per_source=20000,
          device="cpu", seed=20261013, max_new_epochs=None):
    output = Path(output)
    train_seeds = seed_list(dagger_data)
    validation_seeds = seed_list(dagger_validation)
    if not train_seeds or not validation_seeds or set(train_seeds) & set(validation_seeds):
        raise ValueError("DAgger training and validation require disjoint completed episodes")
    if set(train_seeds + validation_seeds) & set(range(1234000, 1234030)):
        raise ValueError("Reserved fresh test seeds may not be used for DAgger")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    original = sample_original(samples_per_source, seed)
    dagger = unique_dagger(dagger_data, train_seeds)
    dagger_val = unique_dagger(dagger_validation, validation_seeds)
    original_val = list(records(ORIGINAL, range(600100, 600108), 2048))
    if len(dagger) < samples_per_source or not dagger_val:
        raise RuntimeError("Insufficient unique DAgger training or validation states")
    dagger = dagger[:samples_per_source]
    if set(r["state_hash"] for r in dagger) & set(r["state_hash"] for r in dagger_val):
        raise RuntimeError("DAgger training/validation state overlap")
    config = {"label": "20M PPO + BC + DAgger v1", "source_model": str(CHAMPION),
              "source_sha256": digest(CHAMPION), "original_dataset": str(ORIGINAL),
              "original_train_seeds": [600000, 600099],
              "original_validation_seeds": [600100, 600107],
              "dagger_dataset": str(Path(dagger_data).resolve()),
              "dagger_train_seeds": train_seeds,
              "dagger_validation_dataset": str(Path(dagger_validation).resolve()),
              "dagger_validation_seeds": validation_seeds,
              "samples_per_source": samples_per_source, "mix_ratio": "1:1",
              "epochs": epochs, "device": device, "random_seed": seed,
              "teacher_hash": TEACHER_HASH, "ppo_environment_steps_added": 0}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("DAgger training config changed")
    if not config_path.exists():
        atomic_json(config_path, config)
    model = MaskablePPO.load(str(CHAMPION), device=device)
    critic_before = {k: v.detach().clone() for k, v in model.policy.value_net.state_dict().items()}
    training_state = output / "training_state.pt"
    if training_state.exists():
        saved = torch.load(training_state, map_location="cpu", weights_only=False)
        checkpoint = output / f"epoch_{saved['epoch']:02d}.zip"
        if (saved["source_sha256"] != config["source_sha256"] or
                digest(checkpoint) != saved["checkpoint_sha256"] or
                not 0 < saved["epoch"] <= epochs):
            raise RuntimeError("DAgger training transaction integrity failed")
        model = MaskablePPO.load(str(checkpoint), device=device)
        history = saved["history"]
        start_epoch = saved["epoch"]
    else:
        if (output / "history.json").exists():
            raise RuntimeError("DAgger history exists without an optimizer/RNG transaction")
        history = [{"epoch": 0, "original_validation": metrics(model.policy, original_val, device),
                    "dagger_validation": metrics(model.policy, dagger_val, device), "timestamp": now()}]
        start_epoch = 0
    policy = model.policy
    actor = (list(policy.state_encoder.parameters()) +
             list(policy.candidate_encoder.parameters()) +
             list(policy.score_head.parameters()))
    if any(not torch.equal(v, critic_before[k]) for k, v in policy.value_net.state_dict().items()):
        raise RuntimeError("DAgger checkpoint changed the frozen Critic")
    optimizer = torch.optim.Adam(actor, lr=5e-5)
    if training_state.exists():
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_random"])
        np.random.set_state(saved["numpy_random"])
        torch.set_rng_state(saved["torch_cpu_random"])
        if saved["torch_cuda_random"] is not None:
            torch.cuda.set_rng_state_all(saved["torch_cuda_random"])
        atomic_json(output / "history.json", history)
    for epoch in range(start_epoch + 1, epochs + 1):
        policy.train()
        mixed = original + dagger
        random.shuffle(mixed)
        for start in range(0, len(mixed), 16):
            obs, mask, actions = batch(mixed[start:start + 16], device)
            loss = -policy.get_distribution(obs, action_masks=mask).log_prob(actions).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0)
            optimizer.step()
        if any(not torch.equal(v, critic_before[k]) for k, v in policy.value_net.state_dict().items()):
            raise RuntimeError("DAgger supervised training modified Critic")
        row = {"epoch": epoch, "original_validation": metrics(policy, original_val, device),
               "dagger_validation": metrics(policy, dagger_val, device), "timestamp": now()}
        history.append(row)
        checkpoint = output / f"epoch_{epoch:02d}.zip"
        temporary_model = output / f".epoch_{epoch:02d}.tmp.zip"
        model.save(str(temporary_model))
        os.replace(temporary_model, checkpoint)
        transaction = {"epoch": epoch, "history": history,
                       "source_sha256": config["source_sha256"],
                       "checkpoint_sha256": digest(checkpoint),
                       "optimizer": optimizer.state_dict(),
                       "python_random": random.getstate(),
                       "numpy_random": np.random.get_state(),
                       "torch_cpu_random": torch.get_rng_state(),
                       "torch_cuda_random": (torch.cuda.get_rng_state_all()
                                             if device == "cuda" else None)}
        temporary_state = output / ".training_state.tmp.pt"
        torch.save(transaction, temporary_state)
        os.replace(temporary_state, training_state)
        atomic_json(output / "history.json", history)
        print(row, flush=True)
        if max_new_epochs is not None and epoch - start_epoch >= max_new_epochs:
            return history
    loaded = MaskablePPO.load(str(output / f"epoch_{epochs:02d}.zip"), device=device)
    obs, mask, _ = batch(dagger_val[:1], device)
    with torch.no_grad():
        before = policy.get_distribution(obs, action_masks=mask).distribution.logits
        after = loaded.policy.get_distribution(obs, action_masks=mask).distribution.logits
    if not torch.allclose(before, after, atol=1e-6):
        raise RuntimeError("DAgger save/load changed actor output")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger-data", type=Path, required=True)
    parser.add_argument("--dagger-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples-per-source", type=int, default=20000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.dagger_data, args.dagger_validation, args.output, args.epochs,
          args.samples_per_source, args.device)


if __name__ == "__main__":
    main()
