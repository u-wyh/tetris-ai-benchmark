"""Actor-only local BC on verified PPO30M danger; masked PPO preservation elsewhere."""
import argparse
import gzip
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.evaluation.hybrid_queue import HASHES, PPO, digest, verify_models
from training.imitation_bc import batch
from training.local_correction_data import RUN, TRAIN, VALIDATION, _paths
from training.train_ppo import atomic_bytes, atomic_json, now


def dataset(seeds):
    correct, keep = [], []
    for seed in seeds:
        shard, _, meta_path = _paths(seed)
        meta = json.loads(meta_path.read_text())
        if digest(shard) != meta["shard_sha256"]:
            raise RuntimeError("Training source shard changed")
        label_path = RUN / "labels" / f"seed_{seed}.json"
        labels = json.loads(label_path.read_text())
        if labels["source_sha256"] != meta["shard_sha256"]:
            raise RuntimeError("Training label source changed")
        with gzip.open(shard, "rt", encoding="utf8") as stream:
            for row in map(json.loads, stream):
                if row["category"] == "normal":
                    row["action"] = row["ppo_action"]
                    keep.append(row)
                elif labels["labels"].get(str(row["step"]), {}).get("category") == "A":
                    row["action"] = row["bc_action"]
                    evidence = labels["labels"][str(row["step"])]
                    row["weight"] = min(2.0, max(1.0, evidence["branches"]["bc"]["pieces"] /
                                                   max(1, evidence["branches"]["ppo"]["pieces"])))
                    correct.append(row)
    return correct, keep


def masked_kl(reference, current, observations, mask):
    with torch.no_grad():
        ref = reference.get_distribution(observations, action_masks=mask).distribution
        ref_probs, ref_logits = ref.probs.detach(), ref.logits.detach()
    new_logits = current.get_distribution(observations, action_masks=mask).distribution.logits
    terms = torch.where(mask, ref_probs * (ref_logits - new_logits), 0.0)
    result = terms.sum(dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite masked KL")
    return result


def metrics(policy, reference, correct, keep, device):
    policy.eval()
    result = {}
    with torch.no_grad():
        for name, rows in (("danger", correct), ("normal", keep)):
            count = matched = ref_matched = 0
            kl_total = 0.0
            for start in range(0, len(rows), 16):
                obs, mask, actions = batch(rows[start:start + 16], device)
                current = policy.get_distribution(obs, action_masks=mask)
                selected = current.get_actions(deterministic=True)
                if not bool(mask.gather(1, selected[:, None]).all()):
                    raise RuntimeError("Local model selected illegal action")
                matched += int((selected == actions).sum())
                count += len(actions)
                if name == "normal":
                    original = reference.get_distribution(obs, action_masks=mask).get_actions(deterministic=True)
                    ref_matched += int((selected == original).sum())
                    kl_total += float(masked_kl(reference, policy, obs, mask).sum())
            result[name] = {"samples": count, "label_top1": matched / count if count else None}
            if name == "normal":
                result[name].update({"ppo_agreement": ref_matched / count if count else None,
                                     "masked_kl": kl_total / count if count else None})
    return result


def _atomic_model(model, path):
    temporary = path.with_name(f".{path.stem}.tmp.zip")
    model.save(str(temporary))
    os.replace(temporary, path)


def train(variant="V1", epochs=3, keep_weight=1.0, device="cuda", seed=20261014):
    if variant not in ("V1", "V2") or epochs < 1:
        raise ValueError("Invalid local training configuration")
    verify_models()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    correct, keep = dataset(TRAIN)
    val_correct, val_keep = dataset(VALIDATION)
    if len(correct) < 8 or len(val_correct) < 2 or not keep or not val_keep:
        raise RuntimeError(f"Insufficient verified A samples: train={len(correct)}, validation={len(val_correct)}")
    folder = RUN / "training" / variant
    folder.mkdir(parents=True, exist_ok=True)
    sources = {str(seed): {"shard_sha256": json.loads((RUN / "labels" / f"seed_{seed}.json").read_text())["source_sha256"],
                           "labels_sha256": digest(RUN / "labels" / f"seed_{seed}.json")}
               for seed in TRAIN + VALIDATION}
    config = {"variant": variant, "epochs": epochs, "keep_weight": keep_weight,
              "device": device, "seed": seed, "source_model_sha256": HASHES["ppo"],
              "sources": sources, "algorithm": "actor_only_masked_CE_plus_forward_KL_v1",
              "ppo_environment_steps_added": 0}
    path = folder / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Training configuration changed on resume")
    if not path.exists():
        atomic_json(path, config)
    reference = MaskablePPO.load(str(PPO), device=device).policy
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    history_path = folder / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    completed = len(history)
    model_path = folder / f"epoch_{completed:02d}.zip"
    model = MaskablePPO.load(str(model_path if completed else PPO), device=device)
    policy = model.policy
    actor = list(policy.state_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.score_head.parameters())
    for parameter in policy.value_net.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(actor, lr=5e-5)
    if completed:
        state = torch.load(folder / f"epoch_{completed:02d}.state.pt", map_location=device,
                           weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    critic = {key: value.detach().clone() for key, value in policy.value_net.state_dict().items()}
    if not history:
        atomic_json(folder / "baseline.json", {"train": metrics(policy, reference, correct, keep, device),
                                               "validation": metrics(policy, reference, val_correct, val_keep, device),
                                               "timestamp": now()})
    batches = max(8, min(128, (len(correct) + 7) // 8))
    for epoch in range(completed + 1, epochs + 1):
        policy.train()
        sums = {"correct": 0.0, "keep": 0.0}
        for _ in range(batches):
            risky = rng.choices(correct, k=8)
            safe = rng.choices(keep, k=8)
            dangerous_obs, dangerous_mask, actions = batch(risky, device)
            safe_obs, safe_mask, _ = batch(safe, device)
            logp = policy.get_distribution(dangerous_obs, action_masks=dangerous_mask).log_prob(actions)
            weights = torch.as_tensor([row["weight"] if variant == "V2" else 1.0 for row in risky], device=device)
            correct_loss = -(weights * logp).mean()
            keep_loss = masked_kl(reference, policy, safe_obs, safe_mask).mean()
            loss = correct_loss + keep_weight * keep_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local imitation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0)
            optimizer.step()
            sums["correct"] += float(correct_loss.detach())
            sums["keep"] += float(keep_loss.detach())
        if any(not torch.equal(value, critic[key]) for key, value in policy.value_net.state_dict().items()):
            raise RuntimeError("Local imitation changed the Critic")
        row = {"epoch": epoch, "loss_correct": sums["correct"] / batches,
               "loss_keep": sums["keep"] / batches,
               "train": metrics(policy, reference, correct, keep, device),
               "validation": metrics(policy, reference, val_correct, val_keep, device),
               "timestamp": now()}
        checkpoint = folder / f"epoch_{epoch:02d}.zip"
        _atomic_model(model, checkpoint)
        state = {"optimizer": optimizer.state_dict(), "python_rng": rng.getstate(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}
        buffer = io.BytesIO()
        torch.save(state, buffer)
        atomic_bytes(folder / f"epoch_{epoch:02d}.state.pt", buffer.getvalue())
        loaded = MaskablePPO.load(str(checkpoint), device=device)
        probe, mask, _ = batch(val_keep[:1], device)
        with torch.no_grad():
            lhs = policy.get_distribution(probe, action_masks=mask).distribution.logits
            rhs = loaded.policy.get_distribution(probe, action_masks=mask).distribution.logits
        if not torch.allclose(lhs, rhs, atol=1e-6):
            raise RuntimeError("Local checkpoint changed masked logits")
        history.append(row)
        atomic_json(history_path, history)
        print(row, flush=True)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("V1", "V2"), default="V1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.variant, args.epochs, args.keep_weight, args.device)
