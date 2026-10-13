"""Reproduce the frozen BC versus vanilla PPO policy-drift analysis."""
import csv
import json
from pathlib import Path
import statistics

import torch
from sb3_contrib import MaskablePPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training.imitation_bc import batch, records
from training.imitation_dataset import digest
from training.kl_maskable_ppo import masked_policy_kl
from training.train_ppo import ROOT, atomic_json

BASE = ROOT / "runs/imitation_v2_hold_20m"
PPO = BASE / "ppo_finetune_2m"
MODELS = {
    "bc": BASE / "bc/epoch_03.zip",
    "ppo_1m": PPO / "milestones/step_001003520/model.zip",
    "ppo_2m": PPO / "milestones/step_002002944/model.zip",
}


def analyze(limit=512):
    torch.set_num_threads(1)
    rows = list(records(BASE / "data", [600100, 600101], limit))
    models = {key: MaskablePPO.load(str(path), device="cpu") for key, path in MODELS.items()}
    reference = models["bc"].policy
    actor_names = [name for name in reference.state_dict()
                   if name.startswith(("state_encoder.", "candidate_encoder.", "score_head."))]
    comparison = {}
    for name, model in models.items():
        correct = equal = total = 0
        kls = []
        with torch.no_grad():
            for start in range(0, len(rows), 16):
                obs, mask, labels = batch(rows[start:start+16], "cpu")
                ref_actions = reference.get_distribution(obs, action_masks=mask).get_actions(deterministic=True)
                actions = model.policy.get_distribution(obs, action_masks=mask).get_actions(deterministic=True)
                correct += int((actions == labels).sum())
                equal += int((actions == ref_actions).sum())
                total += len(labels)
                kls.append(float(masked_policy_kl(model.policy, reference, obs, mask)))
        squared = sum(float((model.policy.state_dict()[key] - reference.state_dict()[key]).square().sum())
                      for key in actor_names)
        reference_squared = sum(float(reference.state_dict()[key].square().sum()) for key in actor_names)
        comparison[name] = {"teacher_accuracy": correct / total, "bc_action_agreement": equal / total,
                            "mean_bc_kl": statistics.mean(kls), "actor_relative_l2":
                            (squared / reference_squared) ** 0.5, "samples": total,
                            "checkpoint_sha256": digest(MODELS[name])}
    events = {}
    for event in PPO.glob("committed/task_*/tensorboard/events.out.tfevents.*"):
        acc = EventAccumulator(str(event))
        acc.Reload()
        scalars = acc.Tags()["scalars"]
        for key in ("train/explained_variance",):
            if key in scalars:
                for item in acc.Scalars(key):
                    events[item.step] = item.value
    metrics = list(csv.DictReader((PPO / "training_metrics.csv").open()))
    columns = {}
    for key in ("value_loss", "approx_kl", "clip_fraction", "entropy"):
        values = [float(r[key]) for r in metrics if r[key]]
        columns[key] = {"mean": statistics.mean(values), "max": max(values),
                        "first_50_tasks": statistics.mean(values[:50]),
                        "last_50_tasks": statistics.mean(values[-50:])}
    vals = list(events.values())
    columns["explained_variance"] = ({"samples": len(vals), "mean": statistics.mean(vals),
                                       "last": events[max(events)]} if vals else {"samples": 0})
    result = {"models": comparison, "training_metrics": columns,
              "validation_curve": list(csv.DictReader((PPO / "evaluations/validation_summary.csv").open())),
              "probe_seeds": [600100, 600101], "probe_limit": limit,
              "note": "Probe data is BC validation data, never the fresh 30-seed final test."}
    atomic_json(ROOT / "runs/candidate_bc_next_stage/degradation.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
