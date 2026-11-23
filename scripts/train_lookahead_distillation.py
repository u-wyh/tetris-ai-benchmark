"""Fit an actor on public Lookahead C actions from student-visited states."""

import argparse
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
from sb3_contrib import MaskablePPO


SEEDS = {
    "train": [3256112300, 3256112301, 3256112302, 3256112303],
    "validation": [3256112310, 3256112311],
    "replay": [3256112320, 3256112321, 3256112322, 3256112323],
}
CAP = 1536
REPLAY_CAP = 1024
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
CORRECTION_WEIGHT = 32
RANDOM_SEED = 20261123


def capture(study, teacher, seed, output, identity, atomic_json):
    manifest = output / f"seed_{seed}.json"
    shard = manifest.with_suffix(".jsonl.gz")
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if (saved["seed"], saved["protocol_sha256"], saved["sha256"]) != (
            seed, identity, study.digest(shard),
        ):
            raise RuntimeError(f"Existing capture differs for seed {seed}")
        return saved

    env = study.LongCandidateTetrisEnv(max_pieces=CAP)
    env.reset(seed=seed)
    temporary = shard.with_suffix(".tmp")
    samples = corrections = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as stream:
            for index in range(CAP):
                observation = env.get_observation()
                mask = env.action_masks()
                public = env.core.get_public_observation()
                action, trace = teacher.decide(observation, mask, public)
                student_action = trace["student_action"]
                if not (mask[action] and mask[student_action]):
                    raise RuntimeError(f"Illegal teacher or student action at seed {seed}, step {index}")
                ids = np.flatnonzero(mask)
                label = int(action)
                correction = label != int(student_action)
                row = {
                    "seed": seed, "step": index, "state": observation["state"].tolist(),
                    "legal_ids": ids.tolist(),
                    "legal_features": observation["candidate"][ids].tolist(),
                    "action": label, "correction": correction,
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                _, _, terminated, truncated, info = env.step(int(student_action))
                samples += 1
                corrections += int(correction)
                if terminated or truncated:
                    break
    finally:
        env.close()

    temporary.replace(shard)
    saved = {
        "seed": seed, "samples": samples, "teacher_corrections": corrections,
        "game_over": bool(terminated), "student_score": int(info["score"]),
        "protocol_sha256": identity, "sha256": study.digest(shard),
    }
    atomic_json(manifest, saved)
    print("capture", saved, flush=True)
    return saved


def read_rows(output, seeds):
    rows = []
    for seed in seeds:
        with gzip.open(output / f"seed_{seed}.jsonl.gz", "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if (row["seed"], row["step"]) != (seed, index):
                    raise RuntimeError(f"Capture row identity differs for seed {seed}")
                rows.append(row)
    return rows


@torch.no_grad()
def metrics(policy, batch, rows, device):
    policy.eval()
    loss_sum = 0.0
    correct = changed = corrected = retained = 0
    for start in range(0, len(rows), BATCH_SIZE):
        part = rows[start:start + BATCH_SIZE]
        observation, legal, labels = batch(part, device)
        distribution = policy.get_distribution(observation, action_masks=legal)
        predicted = distribution.get_actions(deterministic=True)
        if not legal.gather(1, predicted[:, None]).all().item():
            raise RuntimeError("Actor selected an illegal action during validation")
        matches = (predicted == labels).tolist()
        loss_sum -= float(distribution.log_prob(labels).sum())
        correct += sum(matches)
        changed += sum(row["correction"] for row in part)
        corrected += sum(match and row["correction"] for row, match in zip(part, matches))
        retained += sum(match and not row["correction"] for row, match in zip(part, matches))
    return dict(
        samples=len(rows), teacher_corrections=changed, corrected=corrected,
        top1=correct / len(rows), loss=loss_sum / len(rows),
        correction_top1=corrected / changed if changed else None,
        retention_top1=retained / (len(rows) - changed) if changed < len(rows) else None,
    )


def save_model(model, output, epoch):
    path = output / f"epoch_{epoch:02d}.zip"
    temporary = output / f".epoch_{epoch:02d}.tmp.zip"
    model.save(str(temporary))
    temporary.replace(path)
    return path


def fit(source, batch, training, validation, device, output, atomic_json):
    random_generator = random.Random(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    model = MaskablePPO.load(source, device=device)
    actor = tuple(
        parameter
        for module in (model.policy.state_encoder, model.policy.candidate_encoder, model.policy.score_head)
        for parameter in module.parameters()
    )
    critic_before = {name: p.detach().clone() for name, p in model.policy.value_net.state_dict().items()}
    optimizer = torch.optim.Adam(actor, lr=LEARNING_RATE)
    history = [{"epoch": 0, "validation": metrics(model.policy, batch, validation, device)}]
    order = list(range(len(training)))
    for epoch in range(1, EPOCHS + 1):
        model.policy.train()
        random_generator.shuffle(order)
        weighted_loss = weight_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            part = [training[index] for index in order[start:start + BATCH_SIZE]]
            observation, legal, actions = batch(part, device)
            weights = torch.tensor(
                [CORRECTION_WEIGHT if row["correction"] else 1 for row in part], device=device,
            )
            distribution = model.policy.get_distribution(observation, action_masks=legal)
            loss_sum = -(distribution.log_prob(actions) * weights).sum()
            weight = weights.sum()
            loss = loss_sum / weight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0, error_if_nonfinite=True)
            optimizer.step()
            weighted_loss += float(loss_sum.detach())
            weight_sum += float(weight)
        if not math.isfinite(weighted_loss):
            raise RuntimeError("Non-finite supervised objective")
        if any(not torch.equal(p, critic_before[name]) for name, p in model.policy.value_net.state_dict().items()):
            raise RuntimeError("Distillation unexpectedly changed the value network")
        row = {
            "epoch": epoch, "train_weighted_loss": weighted_loss / weight_sum,
            "validation": metrics(model.policy, batch, validation, device),
        }
        history.append(row)
        checkpoint = save_model(model, output, epoch)
        atomic_json(output / "history.json", history)
        print("fit", row, flush=True)

    loaded = MaskablePPO.load(checkpoint, device=device)
    if any(not torch.equal(p, loaded.policy.state_dict()[name]) for name, p in model.policy.state_dict().items()):
        raise RuntimeError("Reloaded checkpoint differs from trained state")
    return checkpoint, history


def replay(study, model, seed):
    env = study.LongCandidateTetrisEnv(max_pieces=REPLAY_CAP)
    env.reset(seed=seed)
    try:
        scores = [0]
        for _ in range(REPLAY_CAP):
            observation, legal = env.get_observation(), env.action_masks()
            action = study.fast_action(model, observation, legal)
            if not legal[action]:
                raise RuntimeError("Replay selected an illegal action")
            _, _, over, truncated, info = env.step(int(action))
            scores.append(int(info["score"]))
            if over or truncated:
                break
        return scores, bool(over)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Worktree that contains the frozen teacher")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    checkout, output = args.checkout.resolve(), args.output.resolve()
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.imitation_bc import batch
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = study.freeze()
        torch.set_num_threads(1)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        protocol = {
            "initial_student_sha256": study.digest(study.STUDENT),
            "teacher_bc_sha256": study.digest(study.BC),
            "teacher_frozen_git": frozen["git"], "teacher_variant": "C",
            "teacher_protocol_sha256": study.digest(study.RUN / "protocol.json"),
            "script_sha256": study.digest(Path(__file__).resolve()),
            "seeds": SEEDS, "capture_policy": "frozen_student", "episode_cap": CAP,
            "replay_cap": REPLAY_CAP, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "correction_weight": CORRECTION_WEIGHT,
            "random_seed": RANDOM_SEED, "device": args.device,
        }
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Existing distillation run uses a different protocol")
        else:
            atomic_json(protocol_path, protocol)
        result_path = output / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if study.digest(output / result["checkpoint"]) != result["checkpoint_sha256"]:
                raise RuntimeError("Existing result checkpoint hash differs")
            print("completed run already verified", flush=True)
            return

        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        student = MaskablePPO.load(study.STUDENT, device="cpu")
        bc = MaskablePPO.load(study.BC, device="cpu")
        teacher = study.AblationController(student, bc, "C")
        captured = [
            capture(study, teacher, seed, output, study.digest(protocol_path), atomic_json)
            for group in ("train", "validation") for seed in SEEDS[group]
        ]
        training, validation = (read_rows(output, SEEDS[group]) for group in ("train", "validation"))
        checkpoint, history = fit(study.STUDENT, batch, training, validation, args.device, output, atomic_json)
        trained = MaskablePPO.load(checkpoint, device="cpu")
        pairs = []
        for seed in SEEDS["replay"]:
            a, over_a = replay(study, student, seed)
            b, over_b = replay(study, trained, seed)
            common = min(len(a), len(b)) - 1
            pairs.append({
                "seed": seed, "student_pieces": len(a) - 1, "trained_pieces": len(b) - 1,
                "student_game_over": over_a, "trained_game_over": over_b,
                "common_pieces": common, "student_score": a[common], "trained_score": b[common],
            })
        result = {
            "started": started, "ended": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "protocol": protocol, "captures": captured, "history": history, "replay_pairs": pairs,
            "checkpoint": checkpoint.name, "checkpoint_sha256": study.digest(checkpoint),
            "checkpoint_use": "supervised_policy", "ppo_updates": 0,
        }
        atomic_json(result_path, result)
        print("completed", result_path, flush=True)


if __name__ == "__main__":
    main()
