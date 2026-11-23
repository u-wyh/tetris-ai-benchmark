"""Compare reference-anchored Lookahead C distillation with the first pilot."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import random
import sys
from zoneinfo import ZoneInfo

import torch
from sb3_contrib import MaskablePPO

import train_lookahead_distillation as pilot


KL_WEIGHT = 4
REPLAY_SEEDS = [3256112330, 3256112331, 3256112332, 3256112333]


def fit(source, batch, training, validation, output, atomic_json):
    torch.manual_seed(pilot.RANDOM_SEED)
    random_generator = random.Random(pilot.RANDOM_SEED)
    model = MaskablePPO.load(source, device="cuda")
    reference = MaskablePPO.load(source, device="cuda").policy
    reference.eval()
    actor = tuple(
        parameter
        for layer in (model.policy.state_encoder, model.policy.candidate_encoder, model.policy.score_head)
        for parameter in layer.parameters()
    )
    critic = {name: p.detach().clone() for name, p in model.policy.value_net.state_dict().items()}
    optimizer = torch.optim.Adam(actor, lr=pilot.LEARNING_RATE)
    history = [{"epoch": 0, "validation": pilot.metrics(model.policy, batch, validation, "cuda")}]
    order = list(range(len(training)))
    for epoch in range(1, pilot.EPOCHS + 1):
        model.policy.train()
        random_generator.shuffle(order)
        total_loss = total_weight = 0.0
        for start in range(0, len(order), pilot.BATCH_SIZE):
            rows = [training[index] for index in order[start:start + pilot.BATCH_SIZE]]
            observation, legal, actions = batch(rows, "cuda")
            correction = torch.tensor([row["correction"] for row in rows], device="cuda")
            distribution = model.policy.get_distribution(observation, action_masks=legal)
            with torch.no_grad():
                old = reference.get_distribution(observation, action_masks=legal).distribution.logits
            new = distribution.distribution.logits
            kl = torch.where(legal, old.exp() * (old - new), 0).sum(1)
            weights = torch.where(correction, pilot.CORRECTION_WEIGHT, KL_WEIGHT)
            loss_sum = (weights * torch.where(correction, -distribution.log_prob(actions), kl)).sum()
            weight = weights.sum()
            optimizer.zero_grad(set_to_none=True)
            (loss_sum / weight).backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss_sum.detach())
            total_weight += float(weight)
        if any(not torch.equal(p, critic[name]) for name, p in model.policy.value_net.state_dict().items()):
            raise RuntimeError("Distillation changed the value network")
        history.append({
            "epoch": epoch, "train_weighted_loss": total_loss / total_weight,
            "validation": pilot.metrics(model.policy, batch, validation, "cuda"),
        })
        checkpoint = pilot.save_model(model, output, epoch)
        atomic_json(output / "history.json", history)
        print("fit", history[-1], flush=True)
    restored = MaskablePPO.load(checkpoint, device="cuda").policy.state_dict()
    if any(not torch.equal(p, restored[name]) for name, p in model.policy.state_dict().items()):
        raise RuntimeError("Reloaded checkpoint differs from trained state")
    return checkpoint, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Worktree containing the frozen evaluator")
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, captures, output = (p.resolve() for p in (args.checkout, args.pilot_output, args.output))
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.imitation_bc import batch
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.freeze()
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        pilot_result = captures / "result.json"
        previous = json.loads(pilot_result.read_text())
        pilot_checkpoint = captures / previous["checkpoint"]
        if (study.digest(study.STUDENT) != previous["protocol"]["initial_student_sha256"]
                or study.digest(pilot_checkpoint) != previous["checkpoint_sha256"]):
            raise RuntimeError("Pilot or source checkpoint identity differs")
        for row in previous["captures"]:
            if study.digest(captures / f"seed_{row['seed']}.jsonl.gz") != row["sha256"]:
                raise RuntimeError(f"Pilot capture identity differs for seed {row['seed']}")
        protocol = {
            "pilot_result_sha256": study.digest(pilot_result),
            "pilot_source_sha256": study.digest(Path(pilot.__file__)),
            "script_sha256": study.digest(Path(__file__)),
            "source_sha256": study.digest(study.STUDENT),
            "epochs": pilot.EPOCHS, "batch_size": pilot.BATCH_SIZE, "lr": pilot.LEARNING_RATE,
            "correction_weight": pilot.CORRECTION_WEIGHT, "ordinary_kl_weight": KL_WEIGHT,
            "train_seed": pilot.RANDOM_SEED, "device": "cuda", "replay_seeds": REPLAY_SEEDS,
            "replay_cap": pilot.REPLAY_CAP,
        }
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Existing run uses a different anchor protocol")
        else:
            atomic_json(protocol_path, protocol)
        result_path = output / "result.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if study.digest(output / existing["checkpoint"]) != existing["checkpoint_sha256"]:
                raise RuntimeError("Existing anchored checkpoint hash differs")
            print("completed run already verified", flush=True)
            return

        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        training, validation = (pilot.read_rows(captures, pilot.SEEDS[group]) for group in ("train", "validation"))
        checkpoint, history = fit(study.STUDENT, batch, training, validation, output, atomic_json)
        models = {
            name: MaskablePPO.load(path, device="cpu")
            for name, path in (("student", study.STUDENT), ("pilot", pilot_checkpoint), ("anchored", checkpoint))
        }
        pairs = []
        for seed in REPLAY_SEEDS:
            rows = {name: pilot.replay(study, model, seed) for name, model in models.items()}
            common = min(len(scores) for scores, _ in rows.values()) - 1
            pairs.append({
                "seed": seed, "common_pieces": common,
                "results": {
                    name: {"pieces": len(scores) - 1, "game_over": over, "common_score": scores[common]}
                    for name, (scores, over) in rows.items()
                },
            })
        atomic_json(result_path, {
            "started": started, "ended": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "protocol": protocol, "history": history, "replay_pairs": pairs,
            "checkpoint": checkpoint.name, "checkpoint_sha256": study.digest(checkpoint),
            "checkpoint_use": "supervised_policy", "ppo_updates": 0,
        })
        print("completed", result_path, flush=True)


if __name__ == "__main__":
    main()
