"""Read-only V1 label audit plus isolated gradient and memorization probes."""
import json
import math
import random

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.hybrid_queue import PPO, HASHES, verify_models
from training.imitation_bc import batch
from training.hybrid_policy import F
from training.local_correction_data import RUN, TRAIN, VALIDATION, _paths
from training.train_local_correction import dataset, masked_kl
from training.train_ppo import atomic_json, now

OUT = RUN / "active_v2" / "diagnosis.json"


def _actor(policy):
    return list(policy.state_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.score_head.parameters())


def _action_metrics(policy, row, device):
    obs, mask, action = batch([row], device)
    with torch.no_grad():
        dist = policy.get_distribution(obs, action_masks=mask).distribution
        logp = dist.logits[0, action[0]]
        legal = dist.logits[0, mask[0]]
        rank = int((legal > logp).sum()) + 1
        selected = int(dist.probs[0].argmax())
        return {"expert_probability": float(logp.exp()), "expert_nll": -float(logp),
                "rank": rank, "top1": rank <= 1, "top3": rank <= 3,
                "top5": rank <= 5, "selected_action": selected,
                "selected_hold": bool(row["legal_features"][row["legal_ids"].index(selected)][F["hold"]])}


def _grad_vector(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
                      for p, g in zip(parameters, grads)])


def _gradient_audit(reference, model, correct, keep, device):
    policy = model.policy
    actor = _actor(policy)
    before = [p.detach().clone() for p in actor]
    rng = random.Random(8181)
    risky = rng.choices(correct, k=8)
    safe = rng.choices(keep, k=8)
    dobs, dmask, dactions = batch(risky, device)
    sobs, smask, _ = batch(safe, device)
    correct_loss = -policy.get_distribution(dobs, action_masks=dmask).log_prob(dactions).mean()
    keep_loss = masked_kl(reference, policy, sobs, smask).mean()
    ga = _grad_vector(correct_loss, actor)
    gc = _grad_vector(keep_loss, actor)
    na, nc = float(ga.norm()), float(gc.norm())
    cos = float(torch.dot(ga, gc) / (ga.norm() * gc.norm())) if na and nc else None
    optimizer = torch.optim.Adam(actor, lr=5e-5)
    optimizer.zero_grad(set_to_none=True)
    (correct_loss + keep_loss).backward()
    preclip = float(torch.nn.utils.clip_grad_norm_(actor, 1.0))
    optimizer.step()
    update = math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(actor, before)))
    gradient_present = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and
                           float(p.grad.abs().sum()) > 0 for p in actor)
    return {"A_loss": float(correct_loss), "C_loss": float(keep_loss),
            "A_gradient_norm": na, "C_gradient_norm": nc,
            "gradient_cosine": cos, "combined_gradient_preclip_norm": preclip,
            "actor_update_l2": update, "actor_gradient_present": gradient_present,
            "reference_frozen": all(not p.requires_grad for p in reference.parameters()),
            "optimizer_actor_parameter_count": sum(p.numel() for p in actor),
            "coefficients": {str(k): {"combined_norm": float((ga + k * gc).norm()),
                                     "C_to_A_norm_ratio": k * nc / na if na else None}
                             for k in (0, .1, 1, 3, 10)}}


def _overfit(correct, device):
    torch.manual_seed(8132)
    model = MaskablePPO.load(str(PPO), device=device)
    policy = model.policy
    actor = _actor(policy)
    optimizer = torch.optim.Adam(actor, lr=1e-4)
    rng = random.Random(8132)
    history = []
    for step in range(201):
        if step % 20 == 0:
            values = [_action_metrics(policy, row, device) for row in correct]
            history.append({"updates": step, "top1": sum(v["top1"] for v in values),
                            "mean_probability": sum(v["expert_probability"] for v in values) / len(values),
                            "mean_nll": sum(v["expert_nll"] for v in values) / len(values)})
            if history[-1]["top1"] == len(correct):
                break
        if step == 200:
            break
        sampled = rng.choices(correct, k=8)
        obs, mask, actions = batch(sampled, device)
        loss = -policy.get_distribution(obs, action_masks=mask).log_prob(actions).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Overfit loss nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor, 1.0)
        optimizer.step()
    return history


def _labels():
    import gzip
    data = []
    rejected = []
    for seed in TRAIN + VALIDATION:
        shard, _, _ = _paths(seed)
        with gzip.open(shard, "rt", encoding="utf8") as stream:
            rows = {r["step"]: r for r in map(json.loads, stream)}
        labels = json.loads((RUN / "labels" / f"seed_{seed}.json").read_text())["labels"]
        for step, evidence in labels.items():
            row = rows[int(step)]
            item = {"seed": seed, "step": int(step), "set": "train" if seed in TRAIN else "validation",
                    "hold_changed": evidence["hold_changed"], "ppo_risk": row["ppo_risk"],
                    "bc_risk": row["bc_risk"], "evidence": evidence,
                    "death_distance": row["death_distance"], "score_before": row["score"]}
            if evidence["category"] == "A":
                data.append(item)
            else:
                rejected.append(item)
    return data, rejected


def main():
    verify_models()
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    correct, keep = dataset(TRAIN)
    val_correct, _ = dataset(VALIDATION)
    reference = MaskablePPO.load(str(PPO), device=device).policy
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    models = [MaskablePPO.load(str(PPO if epoch == 0 else RUN / "training/V1" / f"epoch_{epoch:02d}.zip"),
                               device=device) for epoch in range(4)]
    trusted, rejected = _labels()
    details = []
    for row in correct + val_correct:
        sample = next(x for x in trusted if x["seed"] == row["seed"] and x["step"] == row["step"])
        detail = {"seed": row["seed"], "step": row["step"], "set": sample["set"],
                  "hold_changed": sample["hold_changed"], "expert_action": row["action"],
                  "ppo_action": row["ppo_action"], "epochs": [
                      _action_metrics(model.policy, row, device) for model in models]}
        details.append(detail)
    result = {"version": "diagnosis_v1", "models": HASHES, "device": device,
              "completed_at": now(), "trusted": trusted, "rejected": rejected,
              "action_details": details,
              "gradient_baseline": _gradient_audit(reference, models[0], correct, keep, device),
              "gradient_epoch3": _gradient_audit(reference, models[3], correct, keep, device),
              "overfit": _overfit(correct, device)}
    if OUT.exists():
        raise RuntimeError("Diagnosis already saved; refusing to overwrite immutable result")
    atomic_json(OUT, result)
    return result


if __name__ == "__main__":
    main()
