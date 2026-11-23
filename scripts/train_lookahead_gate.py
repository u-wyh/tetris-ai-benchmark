"""Train a CUDA switch between two frozen public-information policies."""

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch import nn

from capture_lookahead_gate import gate_features


SEEDS = list(range(3256112380, 3256112386))
CAP = 4096
EPOCHS = 16
BATCH = 128
RANDOM_SEED = 20261124
MIN_PRECISION = 0.6
MIN_TRUE_POSITIVES = 5


def model(input_width):
    return nn.Sequential(nn.Linear(input_width, 32), nn.Tanh(), nn.Linear(32, 1), nn.Flatten(0))


def load_partition(directory, seeds, identity, digest, feature_width):
    features, labels, records = [], [], []
    for seed in seeds:
        manifest = directory / f"seed_{seed}.json"
        shard = manifest.with_suffix(".npz")
        row = json.loads(manifest.read_text())
        if (row["seed"], row["protocol_sha256"], row["sha256"]) != (seed, identity, digest(shard)):
            raise RuntimeError(f"Capture shard identity differs for seed {seed}")
        with np.load(shard, allow_pickle=False) as archive:
            x, y, steps = (archive[key] for key in ("features", "labels", "steps"))
        if (x.shape != (len(y), feature_width) or steps.shape != y.shape or len(y) != row["disagreements"]
                or int(y.sum()) != row["switch_labels"] or not np.isfinite(x).all()):
            raise RuntimeError(f"Capture shard content differs for seed {seed}")
        features.append(x)
        labels.append(y)
        records.append(row)
    return np.concatenate(features), np.concatenate(labels), records


def calibration(probabilities, labels):
    order = np.argsort(-probabilities, kind="stable")
    score = probabilities[order]
    sorted_labels = labels[order]
    positive = int(labels.sum())
    if not positive or not len(score):
        raise ValueError("Validation has no positive switch labels")
    endpoints = np.flatnonzero(np.r_[score[:-1] != score[1:], True])
    cumulative = np.cumsum(sorted_labels)[endpoints]
    predictions = endpoints + 1
    precision = cumulative / predictions
    recall = cumulative / positive
    average_precision = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    eligible = (precision >= MIN_PRECISION) & (cumulative >= MIN_TRUE_POSITIVES)
    rows = np.flatnonzero(eligible)
    if not len(rows):
        rows = np.flatnonzero(cumulative >= MIN_TRUE_POSITIVES)
        key = lambda i: (precision[i], cumulative[i], score[endpoints[i]])
    else:
        key = lambda i: (cumulative[i], precision[i], score[endpoints[i]])
    if not len(rows):
        raise ValueError("Validation has fewer positive switch labels than the fixed calibration minimum")
    index = max(rows, key=key)
    return {
        "samples": len(labels), "positives": positive, "average_precision": average_precision,
        "threshold": float(score[endpoints[index]]), "precision": float(precision[index]),
        "recall": float(recall[index]), "true_positives": int(cumulative[index]),
        "predicted_switches": int(predictions[index]), "meets_precision_target": bool(eligible[index]),
    }


def train(x, y):
    torch.manual_seed(RANDOM_SEED)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-5] = 1
    normalized = torch.as_tensor((x - mean) / std, device="cuda")
    label = torch.as_tensor(y, dtype=torch.float32, device="cuda")
    positives = int(y.sum())
    if not 0 < positives < len(y):
        raise ValueError("Training requires both switch and retain labels")
    positive_weight = min(8.0, math.sqrt((len(y) - positives) / positives))
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device="cuda"))
    network = model(x.shape[1]).to("cuda")
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-3)
    history = []
    for epoch in range(EPOCHS):
        total = 0.0
        order = torch.randperm(len(x), generator=generator)
        for rows in order.split(BATCH):
            ids = rows.to("cuda")
            loss = objective(network(normalized[ids]), label[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        history.append(total / len(x))
        print("train", epoch + 1, history[-1], flush=True)
    return network.cpu().eval(), mean, std, history, positive_weight


@torch.no_grad()
def evaluate(study, student, bc, scale, seed, gate=None, normalization=None, threshold=None):
    env = study.LongCandidateTetrisEnv(max_pieces=CAP)
    scores, game_over, switches, disagreements, durations = [0], False, 0, 0, []
    env.reset(seed=seed)
    try:
        for _ in range(CAP):
            start = time.perf_counter()
            observation, legal = env.get_observation(), env.action_masks()
            action = study.fast_action(student, observation, legal)
            if gate is not None:
                alternate = study.fast_action(bc, observation, legal)
                if alternate != action:
                    disagreements += 1
                    features = gate_features(observation, action, alternate, scale)
                    mean, std = normalization
                    probability = gate(torch.from_numpy((features - mean) / std)[None]).sigmoid().item()
                    if probability >= threshold:
                        action = alternate
                        switches += 1
            durations.append(time.perf_counter() - start)
            if not legal[action]:
                raise RuntimeError("Evaluation selected an illegal action")
            _, _, game_over, truncated, info = env.step(int(action))
            scores.append(int(info["score"]))
            if game_over or truncated:
                break
        metrics = {
            "pieces": len(scores) - 1, "game_over": bool(game_over),
            "switches": switches, "disagreements": disagreements,
            "median_decision_ms": float(1000 * np.median(durations)),
        }
        return scores, metrics
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, captures, output = (p.resolve() for p in (args.checkout, args.capture, args.output))
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.candidate_features import FEATURE_SCALE
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.freeze()
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        capture_protocol = captures / "protocol.json"
        identity = study.digest(capture_protocol)
        capture = json.loads(capture_protocol.read_text())
        if (capture["student_sha256"], capture["bc_sha256"], capture["teacher_protocol_sha256"]) != (
            study.digest(study.STUDENT), study.digest(study.BC), study.digest(study.RUN / "protocol.json"),
        ):
            raise RuntimeError("Frozen capture sources differ")
        train_seeds, val_seeds = capture["train_seeds"], capture["validation_seeds"]
        completed = json.loads((captures / "completed.json").read_text())
        if (completed["protocol_sha256"], completed["seeds"]) != (identity, train_seeds + val_seeds):
            raise RuntimeError("Capture completion identity differs")
        if capture["feature_scale"] != FEATURE_SCALE.tolist():
            raise RuntimeError("Capture feature normalization differs")
        distinct = set(train_seeds) | set(val_seeds) | set(SEEDS)
        if len(distinct) != len(train_seeds) + len(val_seeds) + len(SEEDS):
            raise RuntimeError("Training, calibration, and replay seeds overlap")
        protocol = {
            "capture_protocol_sha256": identity, "script_sha256": study.digest(Path(__file__)),
            "train_seeds": train_seeds, "calibration_seeds": val_seeds, "replay_seeds": SEEDS,
            "replay_cap": CAP, "epochs": EPOCHS, "batch": BATCH, "optimization_seed": RANDOM_SEED,
            "minimum_precision": MIN_PRECISION, "minimum_true_positives": MIN_TRUE_POSITIVES,
        }
        protocol_path = output / "protocol.json"
        if protocol_path.is_file():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Training protocol changed")
        else:
            atomic_json(protocol_path, protocol)
        result_path = output / "result.json"
        checkpoint = output / "gate.pt"
        if result_path.is_file():
            saved = json.loads(result_path.read_text())
            if study.digest(checkpoint) != saved["checkpoint_sha256"]:
                raise RuntimeError("Completed checkpoint hash differs")
            print("completed run already verified", flush=True)
            return

        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        width = 2 * len(FEATURE_SCALE)
        x, y, training = load_partition(captures, train_seeds, identity, study.digest, width)
        xv, yv, validation = load_partition(captures, val_seeds, identity, study.digest, width)
        gate, mean, std, history, weight = train(x, y)
        with torch.no_grad():
            probabilities = gate(torch.from_numpy((xv - mean) / std)).sigmoid().numpy()
        choice = calibration(probabilities, yv)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({
            "model": gate.state_dict(), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
            "threshold": choice["threshold"],
        }, temporary)
        temporary.replace(checkpoint)
        saved = torch.load(checkpoint, weights_only=True)
        restored = model(width).eval()
        restored.load_state_dict(saved["model"])
        mean, std = saved["mean"].numpy(), saved["std"].numpy()
        with torch.no_grad():
            restored_probabilities = restored(torch.from_numpy((xv - mean) / std)).sigmoid().numpy()
        if not np.array_equal(probabilities, restored_probabilities):
            raise RuntimeError("Checkpoint reload changed gate predictions")
        print("calibration", choice, flush=True)

        student = MaskablePPO.load(study.STUDENT, device="cpu")
        bc = MaskablePPO.load(study.BC, device="cpu")
        pairs = []
        for seed in SEEDS:
            original, original_metrics = evaluate(study, student, bc, FEATURE_SCALE, seed)
            gated, gated_metrics = evaluate(
                study, student, bc, FEATURE_SCALE, seed, restored, (mean, std), saved["threshold"],
            )
            common = min(len(original), len(gated)) - 1
            row = {
                "seed": seed, "common_pieces": common, "student_score": original[common],
                "gate_score": gated[common], "student": original_metrics, "gate": gated_metrics,
            }
            pairs.append(row)
            print("replay", row, flush=True)
        atomic_json(result_path, {
            "started": started, "ended": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "protocol": protocol, "capture_training": training, "capture_validation": validation,
            "train_loss": history, "positive_class_weight": weight, "calibration": choice,
            "pairs": pairs, "checkpoint_sha256": study.digest(checkpoint), "ppo_updates": 0,
        })


if __name__ == "__main__":
    main()
