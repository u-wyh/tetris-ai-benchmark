"""Fixed-batch A/C gradient interaction after V1 epoch 3; no optimizer steps."""
import json
import random
import statistics

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.hybrid_queue import PPO, verify_models
from training.evaluation.local_correction_active_v2 import V2
from training.imitation_bc import batch
from training.local_correction_data import RUN, TRAIN
from training.train_local_correction import dataset, masked_kl
from training.train_ppo import atomic_json


def main():
    path = V2 / "gradient_sweep.json"
    if path.exists():
        return json.loads(path.read_text())
    verify_models()
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    correct, keep = dataset(TRAIN)
    reference = MaskablePPO.load(str(PPO), device=device).policy
    policy = MaskablePPO.load(str(RUN / "training/V1/epoch_03.zip"), device=device).policy
    reference.eval()
    policy.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    actor = list(policy.state_encoder.parameters()) + list(policy.candidate_encoder.parameters()) + list(policy.score_head.parameters())
    rng = random.Random(62014)
    rows = []
    for index in range(16):
        dangerous = rng.choices(correct, k=8)
        normal = rng.choices(keep, k=8)
        aobs, amask, actions = batch(dangerous, device)
        cobs, cmask, _ = batch(normal, device)
        a = -policy.get_distribution(aobs, action_masks=amask).log_prob(actions).mean()
        c = masked_kl(reference, policy, cobs, cmask).mean()
        def vector(loss):
            grads = torch.autograd.grad(loss, actor, retain_graph=True, allow_unused=True)
            return torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
                              for p, g in zip(actor, grads)])
        ga, gc = vector(a), vector(c)
        na, nc = float(ga.norm()), float(gc.norm())
        rows.append({"batch": index, "A_gradient_norm": na, "C_gradient_norm": nc,
                     "cosine": float(torch.dot(ga, gc) / (ga.norm() * gc.norm())) if na and nc else None,
                     "C_to_A_ratio": nc / na if na else None,
                     "A_loss": float(a.detach()), "C_loss": float(c.detach())})
    result = {"version": "epoch3_16_fixed_batches_v1", "device": device,
              "batch_size_each": 8, "rows": rows,
              "median_C_to_A_ratio": statistics.median(x["C_to_A_ratio"] for x in rows),
              "negative_cosine_batches": sum(x["cosine"] < 0 for x in rows),
              "median_cosine": statistics.median(x["cosine"] for x in rows)}
    atomic_json(path, result)
    return result


if __name__ == "__main__":
    main()
