"""Collect fresh public disagreement states for a selective Lookahead switch."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sb3_contrib import MaskablePPO


TRAIN_SEEDS = list(range(3256112360, 3256112368))
VALIDATION_SEEDS = list(range(3256112370, 3256112374))
CAP = 4096


def gate_features(observation, student_action, bc_action, scale):
    """Encode the two visible landing candidates in their fixed policy order."""
    rows = observation["candidate"][[student_action, bc_action]]
    return (rows.astype(np.float32) / scale).reshape(-1)


def collect(study, teacher, scale, directory, seed, identity, atomic_json):
    manifest, shard = (directory / f"seed_{seed}{suffix}" for suffix in (".json", ".npz"))
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        expected = (seed, identity, study.digest(shard))
        if (metadata["seed"], metadata["protocol_sha256"], metadata["sha256"]) != expected:
            raise RuntimeError(f"Stored capture identity differs for seed {seed}")
        print("verified capture", metadata, flush=True)
        return

    env = study.LongCandidateTetrisEnv(max_pieces=CAP)
    env.reset(seed=seed)
    features, labels, steps = [], [], []
    try:
        for step in range(CAP):
            observation, legal = env.get_observation(), env.action_masks()
            student_action = study.fast_action(teacher.student, observation, legal)
            bc_action = study.fast_action(teacher.bc, observation, legal)
            if not (legal[student_action] and legal[bc_action]):
                raise RuntimeError(f"Illegal policy action at seed {seed}, step {step}")
            if student_action != bc_action:
                public = env.core.get_public_observation()
                selected, trace = teacher.decide(observation, legal, public)
                if (trace["student_action"], trace["bc_action"]) != (student_action, bc_action):
                    raise RuntimeError(f"Frozen teacher proposal mismatch at seed {seed}, step {step}")
                if selected not in (student_action, bc_action):
                    raise RuntimeError("Two-root teacher chose an unexpected action")
                features.append(gate_features(observation, student_action, bc_action, scale))
                labels.append(selected == bc_action)
                steps.append(step)
            _, _, terminated, truncated, info = env.step(int(student_action))
            if terminated or truncated:
                break
    finally:
        env.close()

    temporary = shard.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, features=np.asarray(features, dtype=np.float32).reshape(-1, 2 * len(scale)),
            labels=np.asarray(labels, dtype=np.bool_), steps=np.asarray(steps, dtype=np.int32),
        )
    temporary.replace(shard)
    metadata = {
        "seed": seed, "pieces": step + 1, "score": int(info["score"]), "game_over": bool(terminated),
        "disagreements": len(labels), "switch_labels": sum(labels),
        "protocol_sha256": identity, "sha256": study.digest(shard),
    }
    atomic_json(manifest, metadata)
    print("capture", metadata, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Worktree containing the frozen teacher")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, output = args.checkout.resolve(), args.output.resolve()
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.candidate_features import FEATURE_NAMES, FEATURE_SCALE
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = study.freeze()
        torch.set_num_threads(1)
        protocol = {
            "train_seeds": TRAIN_SEEDS, "validation_seeds": VALIDATION_SEEDS, "cap": CAP,
            "teacher_git": frozen["git"], "teacher": "C",
            "teacher_protocol_sha256": study.digest(study.RUN / "protocol.json"),
            "student_sha256": study.digest(study.STUDENT), "bc_sha256": study.digest(study.BC),
            "capture_script_sha256": study.digest(Path(__file__)),
            "feature_names_per_candidate": list(FEATURE_NAMES), "feature_scale": FEATURE_SCALE.tolist(),
        }
        protocol_path = output / "protocol.json"
        if protocol_path.is_file():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Capture protocol changed")
        else:
            atomic_json(protocol_path, protocol)
        identity = study.digest(protocol_path)
        student = MaskablePPO.load(study.STUDENT, device="cpu")
        bc = MaskablePPO.load(study.BC, device="cpu")
        teacher = study.AblationController(student, bc, "C")
        for seed in TRAIN_SEEDS + VALIDATION_SEEDS:
            collect(study, teacher, FEATURE_SCALE, output, seed, identity, atomic_json)
        atomic_json(output / "completed.json", {"protocol_sha256": identity, "seeds": TRAIN_SEEDS + VALIDATION_SEEDS})


if __name__ == "__main__":
    main()
