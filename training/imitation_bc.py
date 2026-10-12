"""Actor-only behavior cloning through the official masked Candidate distribution."""
import argparse
import csv
import gzip
import json
import random
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.tetris_core import ACTION_COUNT
from training.train_ppo import atomic_json, now
from training.imitation_dataset import digest, TEACHER_HASH


def records(dataset, seeds, limit=None):
    count = 0
    for seed in seeds:
        shard = dataset / f"seed_{seed}.jsonl.gz"
        meta = json.loads((dataset / f"seed_{seed}.json").read_text())
        if digest(shard) != meta["sha256"] or meta["teacher_hash"] != TEACHER_HASH:
            raise RuntimeError(f"Shard mismatch: {shard}")
        with gzip.open(shard, "rt", encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["seed"] != seed or row["teacher_hash"] != TEACHER_HASH:
                    raise RuntimeError("Sample identity mismatch")
                yield row
                count += 1
                if limit is not None and count >= limit:
                    return


def batch(rows, device):
    state = np.asarray([r["state"] for r in rows], dtype=np.float32)
    candidates = np.zeros((len(rows), ACTION_COUNT, 16), dtype=np.uint8)
    mask = np.zeros((len(rows), ACTION_COUNT), dtype=np.bool_)
    actions = np.asarray([r["action"] for r in rows], dtype=np.int64)
    for index, row in enumerate(rows):
        ids = np.asarray(row["legal_ids"], dtype=np.intp)
        candidates[index, ids] = np.asarray(row["legal_features"], dtype=np.uint8)
        mask[index, ids] = True
        if not mask[index, actions[index]]:
            raise RuntimeError("Illegal teacher label")
    obs = {"state": torch.as_tensor(state, device=device),
           "candidate": torch.as_tensor(candidates, device=device)}
    return obs, torch.as_tensor(mask, device=device), torch.as_tensor(actions, device=device)


def metrics(policy, rows, device, batch_size=16):
    policy.eval()
    total = correct = 0
    logp_sum = prob_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            obs, mask, actions = batch(rows[start:start+batch_size], device)
            dist = policy.get_distribution(obs, action_masks=mask)
            logp = dist.log_prob(actions)
            predicted = dist.get_actions(deterministic=True)
            correct += int((predicted == actions).sum())
            total += len(actions)
            logp_sum += float(logp.sum())
            prob_sum += float(logp.exp().sum())
            if not bool(mask.gather(1, predicted[:, None]).all()):
                raise RuntimeError("BC selected illegal action")
    return {"loss": -logp_sum / total, "top1": correct / total,
            "teacher_probability": prob_sum / total, "samples": total}


def train(dataset, output, source, train_seeds, val_seeds, epochs=3,
          max_train=None, max_val=None, device="cpu", seed=20261012):
    if set(train_seeds) & set(val_seeds):
        raise ValueError("Training and validation episodes overlap")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    config = {"label": "20M PPO + BC", "source_model": str(source.resolve()),
              "source_sha256": digest(source), "dataset": str(dataset.resolve()),
              "train_seeds": train_seeds, "validation_seeds": val_seeds,
              "epochs": epochs, "max_train": max_train, "max_val": max_val,
              "device": device, "random_seed": seed, "teacher_hash": TEACHER_HASH,
              "ppo_environment_steps_added": 0}
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("BC config changed")
    if not path.exists():
        atomic_json(path, config)
    model = MaskablePPO.load(str(source), device=device)
    policy = model.policy
    actor = list(policy.state_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.score_head.parameters())
    critic_before = {name: value.detach().clone() for name, value in policy.value_net.state_dict().items()}
    optimizer = torch.optim.Adam(actor, lr=1e-4)
    train_rows = list(records(dataset, train_seeds, max_train))
    val_rows = list(records(dataset, val_seeds, max_val))
    if not train_rows or not val_rows:
        raise ValueError("Both episode partitions need samples")
    history = []
    history.append({"epoch": 0, "train": metrics(policy, train_rows, device),
                    "validation": metrics(policy, val_rows, device), "timestamp": now()})
    for epoch in range(1, epochs + 1):
        policy.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for start in range(0, len(order), 16):
            rows = [train_rows[i] for i in order[start:start+16]]
            obs, mask, actions = batch(rows, device)
            loss = -policy.get_distribution(obs, action_masks=mask).log_prob(actions).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0)
            optimizer.step()
        if any(not torch.equal(value, critic_before[name]) for name, value in policy.value_net.state_dict().items()):
            raise RuntimeError("BC modified Critic")
        row = {"epoch": epoch, "train": metrics(policy, train_rows, device),
               "validation": metrics(policy, val_rows, device), "timestamp": now()}
        history.append(row)
        model.save(str(output / f"epoch_{epoch:02d}.zip"))
        atomic_json(output / "history.json", history)
        print(row, flush=True)
    loaded = MaskablePPO.load(str(output / f"epoch_{epochs:02d}.zip"), device=device)
    probe = batch(val_rows[:1], device)
    with torch.no_grad():
        before = policy.get_distribution(probe[0], action_masks=probe[1]).distribution.logits
        after = loaded.policy.get_distribution(probe[0], action_masks=probe[1]).distribution.logits
    if not torch.allclose(before, after, atol=1e-6):
        raise RuntimeError("BC save/load changed policy output")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--val-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.dataset, args.output, args.source, args.train_seeds, args.val_seeds,
          args.epochs, args.max_train, args.max_val, args.device)


if __name__ == "__main__":
    main()
