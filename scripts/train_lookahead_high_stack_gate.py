"""Collect natural high stacks from PPO and fit a public-action switch."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from capture_lookahead_gate import gate_features
from enrich_lookahead_gate_next3 import next3_features
from train_lookahead_gate import BATCH, EPOCHS, calibration, load_partition, model, train


TRAIN_SEEDS = list(range(3256112440, 3256112446))
VALIDATION_SEEDS = list(range(3256112450, 3256112453))
CAP = 4096
MIN_HEIGHT = 10
MIN_TRAIN_PER_CLASS = 20
MIN_VALIDATION_PER_CLASS = 10


def capture(study, rollout, teacher, feature_names, scale, output, seed, identity, atomic_json, cap=CAP):
    manifest = output / f"seed_{seed}.json"
    shard = manifest.with_suffix(".npz")
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        if (metadata["seed"], metadata["protocol_sha256"], metadata["sha256"]) != (
            seed, identity, study.digest(shard),
        ):
            raise RuntimeError(f"Existing high-stack capture differs for seed {seed}")
        return

    env = study.LongCandidateTetrisEnv(max_pieces=cap)
    env.reset(seed=seed)
    height_column = feature_names.index("max_height_before")
    features, labels, steps = [], [], []
    high_steps = severe_steps = severe_disagreements = 0
    try:
        for step in range(cap):
            observation, legal = env.get_observation(), env.action_masks()
            rollout_action = study.fast_action(rollout, observation, legal)
            if not legal[rollout_action]:
                raise RuntimeError(f"Illegal rollout action at seed {seed}, step {step}")
            height = int(observation["candidate"][rollout_action, height_column])
            if height >= MIN_HEIGHT:
                high_steps += 1
                severe_steps += int(height >= 12)
                a = study.fast_action(teacher.student, observation, legal)
                b = study.fast_action(teacher.bc, observation, legal)
                if not (legal[a] and legal[b]):
                    raise RuntimeError(f"Illegal teacher proposal at seed {seed}, step {step}")
                if a != b:
                    public = env.core.get_public_observation()
                    selected, trace = teacher.decide(observation, legal, public)
                    if (trace["student_action"], trace["bc_action"]) != (a, b) or selected not in (a, b):
                        raise RuntimeError(f"Frozen teacher proposals differ at seed {seed}, step {step}")
                    features.append(np.concatenate((gate_features(observation, a, b, scale),
                                                    next3_features(public))))
                    steps.append(step)
                    labels.append(selected == b)
                    severe_disagreements += int(height >= 12)
            _, _, game_over, truncated, info = env.step(int(rollout_action))
            if game_over or truncated:
                break
    finally:
        env.close()

    temporary = shard.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, features=np.asarray(features, dtype=np.float32).reshape(-1, 2 * len(scale) + 21),
            labels=np.asarray(labels, dtype=np.bool_), steps=np.asarray(steps, dtype=np.int32),
        )
    temporary.replace(shard)
    row = {
        "seed": seed, "pieces": step + 1, "score": int(info["score"]), "game_over": bool(game_over),
        "height_ge_10": high_steps, "height_ge_12": severe_steps,
        "disagreements_ge_12": severe_disagreements, "disagreements": len(labels), "switch_labels": sum(labels),
        "protocol_sha256": identity, "sha256": study.digest(shard),
    }
    atomic_json(manifest, row)
    print("high-stack capture", row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=VALIDATION_SEEDS)
    args = parser.parse_args()
    seeds = args.train_seeds + args.validation_seeds
    if args.cap < 1:
        parser.error("--cap must be positive")
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error("Training and validation seeds must be distinct unsigned 32-bit numbers")
    checkout, output = (p.resolve() for p in (args.checkout, args.output))
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.candidate_features import FEATURE_NAMES, FEATURE_SCALE
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = study.freeze()
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        protocol = {
            "rollout": "frozen_ppo_30m", "teacher": "Lookahead_C", "minimum_stack_height": MIN_HEIGHT,
            "train_seeds": args.train_seeds, "validation_seeds": args.validation_seeds, "cap": args.cap,
            "minimum_train_examples_per_class": MIN_TRAIN_PER_CLASS,
            "minimum_validation_examples_per_class": MIN_VALIDATION_PER_CLASS,
            "fit_epochs": EPOCHS, "batch_size": BATCH, "teacher_git": frozen["git"],
            "teacher_protocol_sha256": study.digest(study.RUN / "protocol.json"),
            "sources": {key: study.digest(path) for key, path in (
                ("rollout_ppo", study.PPO), ("student", study.STUDENT), ("bc", study.BC),
                ("script", Path(__file__)), ("fit", Path(__file__).with_name("train_lookahead_gate.py")),
                ("pair_features", Path(__file__).with_name("capture_lookahead_gate.py")),
                ("next3", Path(__file__).with_name("enrich_lookahead_gate_next3.py")),
            )},
        }
        protocol_path = output / "protocol.json"
        result_path = output / "result.json"
        checkpoint = output / "gate.pt"
        if protocol_path.is_file():
            previous = json.loads(protocol_path.read_text())
            if result_path.is_file():
                saved = json.loads(result_path.read_text())
                if (saved["protocol"] != previous
                        or any(previous.get(key) != value for key, value in protocol.items() if key != "sources")
                        or any(previous["sources"].get(key) != value for key, value in protocol["sources"].items()
                               if key != "script")
                        or (saved["checkpoint_sha256"] is not None
                            and saved["checkpoint_sha256"] != study.digest(checkpoint))):
                    raise RuntimeError("Existing high-stack result differs")
                print("completed high-stack run already verified", flush=True)
                return
            if previous != protocol:
                raise RuntimeError("Existing high-stack protocol differs")
        else:
            atomic_json(protocol_path, protocol)
        identity = study.digest(protocol_path)

        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        data = output / "capture"
        data.mkdir(exist_ok=True)
        rollout = MaskablePPO.load(study.PPO, device="cpu")
        student = MaskablePPO.load(study.STUDENT, device="cpu")
        bc = MaskablePPO.load(study.BC, device="cpu")
        teacher = study.AblationController(student, bc, "C")
        for seed in seeds:
            capture(study, rollout, teacher, FEATURE_NAMES, FEATURE_SCALE, data, seed, identity, atomic_json, args.cap)

        width = 2 * len(FEATURE_NAMES) + 21
        x, y, captures = load_partition(data, args.train_seeds, identity, study.digest, width)
        xv, yv, validation = load_partition(data, args.validation_seeds, identity, study.digest, width)
        train_counts = (int(y.sum()), int((~y).sum()))
        validation_counts = (int(yv.sum()), int((~yv).sum()))
        result = {
            "protocol": protocol, "started": started, "captures": captures, "validation": validation,
            "train_switch_retain": train_counts, "validation_switch_retain": validation_counts,
            "status": "insufficient_label_coverage", "checkpoint_sha256": None,
        }
        if min(train_counts) >= MIN_TRAIN_PER_CLASS and min(validation_counts) >= MIN_VALIDATION_PER_CLASS:
            gate, mean, std, history, weight = train(x, y)
            with torch.no_grad():
                probability = gate(torch.from_numpy((xv - mean) / std)).sigmoid().numpy()
            choice = calibration(probability, yv)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                "model": gate.state_dict(), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                "threshold": choice["threshold"],
            }, temporary)
            temporary.replace(checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            restored = model(width).eval()
            restored.load_state_dict(saved["model"])
            with torch.no_grad():
                score = restored(torch.from_numpy((xv - saved["mean"].numpy()) / saved["std"].numpy()))
                if not np.array_equal(probability, score.sigmoid().numpy()):
                    raise RuntimeError("Reload changed high-stack validation predictions")
            result.update(status="trained", train_loss=history, positive_class_weight=weight,
                          calibration=choice, checkpoint_sha256=study.digest(checkpoint))
        result["ended"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        atomic_json(result_path, result)
        print(result["status"], "train classes", train_counts, "validation classes", validation_counts, flush=True)


if __name__ == "__main__":
    main()
