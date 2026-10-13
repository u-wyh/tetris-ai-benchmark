"""Same-state masked V2-action agreement for all next-stage Candidate models."""
import json
from pathlib import Path
import random

import torch
from sb3_contrib import MaskablePPO

from training.imitation_bc import batch, records
from training.imitation_dataset import TEACHER_HASH, digest
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/candidate_bc_next_stage"
DATA = ROOT / "runs/imitation_v2_hold_20m/data"
SEEDS = list(range(600100, 600108))
SAMPLES_PER_SEED = 256


def probe_rows():
    rows = []
    for seed in SEEDS:
        rng = random.Random(seed + 20261013)
        sampled = []
        for index, row in enumerate(records(DATA, [seed])):
            if index < SAMPLES_PER_SEED:
                sampled.append(row)
            else:
                place = rng.randrange(index + 1)
                if place < SAMPLES_PER_SEED:
                    sampled[place] = row
        if len(sampled) != SAMPLES_PER_SEED:
            raise RuntimeError(f"Insufficient BC validation states for seed {seed}")
        rows.extend(sampled)
    return rows


def agreement(models):
    torch.set_num_threads(1)
    rows = probe_rows()
    result = {"protocol": {"source": "held-out V2 Hold expert validation states",
                            "seeds": SEEDS, "samples_per_seed": SAMPLES_PER_SEED,
                            "teacher_hash": TEACHER_HASH, "samples": len(rows),
                            "deterministic": True, "action_mask": True,
                            "not_on_policy_survival": True},
              "models": {}, "completed_at": now()}
    for label, path in models.items():
        path = Path(path)
        model = MaskablePPO.load(str(path), device="cpu")
        correct = legal = total = 0
        with torch.no_grad():
            for start in range(0, len(rows), 16):
                obs, mask, teacher_actions = batch(rows[start:start + 16], "cpu")
                actions = model.policy.get_distribution(obs, action_masks=mask).get_actions(
                    deterministic=True)
                correct += int((actions == teacher_actions).sum())
                legal += int(mask.gather(1, actions[:, None]).sum())
                total += len(actions)
        if legal != total:
            raise RuntimeError(f"Model {label} selected an illegal action")
        result["models"][label] = {"v2_action_agreement": correct / total,
                                    "correct": correct, "samples": total,
                                    "model_sha256": digest(path)}
    atomic_json(RUN / "agreement_summary.json", result)
    return result


if __name__ == "__main__":
    selected = json.loads((ROOT / "runs/candidate_bc_dagger_v1/selected.json").read_text())["epoch"]
    print(json.dumps(agreement({
        "pure_30m": ROOT / "runs/ppo_candidate_raw_30m_seed42/final/model.zip",
        "bc": RUN / "champion/model.zip",
        "bc_ppo_2m": ROOT / "runs/imitation_v2_hold_20m/ppo_finetune_2m/final/model.zip",
        "kl_ppo_2m": ROOT / "runs/candidate_bc_klppo_2m/final/model.zip",
        "dagger_selected": ROOT / "runs/candidate_bc_dagger_v1/bc" / f"epoch_{selected:02d}.zip",
    }), indent=2))
