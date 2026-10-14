"""In-memory 24-update KL coefficient probe, excluded from formal model selection."""
import gc
import json
import random

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.hybrid_queue import PPO, verify_models
from training.evaluation.local_correction_active_v2 import V2
from training.imitation_bc import batch
from training.local_correction_data import TRAIN, VALIDATION
from training.train_local_correction import dataset, masked_kl, metrics
from training.train_ppo import atomic_json


def main():
    path = V2 / "loss_sensitivity.json"
    if path.exists():
        return json.loads(path.read_text())
    verify_models()
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    danger, keep = dataset(TRAIN)
    val_danger, val_keep = dataset(VALIDATION)
    reference = MaskablePPO.load(str(PPO), device=device).policy
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    rng = random.Random(102414)
    batches = [(rng.choices(danger, k=8), rng.choices(keep, k=8)) for _ in range(24)]
    results = []
    for coefficient in (0, 1, 3, 10):
        torch.manual_seed(1014)
        model = MaskablePPO.load(str(PPO), device=device)
        policy = model.policy
        actor = list(policy.state_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.score_head.parameters())
        for p in policy.value_net.parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.Adam(actor, lr=5e-5)
        for risk_rows, normal_rows in batches:
            aobs, amask, actions = batch(risk_rows, device)
            cobs, cmask, _ = batch(normal_rows, device)
            a = -policy.get_distribution(aobs, action_masks=amask).log_prob(actions).mean()
            c = masked_kl(reference, policy, cobs, cmask).mean()
            loss = a + coefficient * c
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0)
            optimizer.step()
        measured = metrics(policy, reference, danger, keep, device)
        validation = metrics(policy, reference, val_danger, val_keep, device)
        results.append({"keep_weight": coefficient, "updates": 24,
                        "train": measured, "validation": validation})
        del model, policy, optimizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    result = {"version": "controlled_24_update_KL_sensitivity_v1", "device": device,
              "same_fixed_batches": True, "learning_rate": 5e-5,
              "not_formal_checkpoint": True, "results": results}
    atomic_json(path, result)
    return result


if __name__ == "__main__":
    main()
