"""Replay the frozen student path to add the public three-piece preview."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from capture_lookahead_gate import gate_features


PIECE_INDEX = {piece: i for i, piece in enumerate("IJLOSTZ")}
ONE_HOT = np.eye(len(PIECE_INDEX), dtype=np.float32)


def next3_features(public):
    pieces = public["next"]
    if len(pieces) != 3:
        raise ValueError("Expected exactly three public preview pieces")
    return ONE_HOT[[PIECE_INDEX[piece] for piece in pieces]].reshape(-1)


def enrich(study, student, bc, scale, previous, output, seed, source_id, identity, atomic_json):
    source = previous / f"seed_{seed}.json"
    original = json.loads(source.read_text())
    data = source.with_suffix(".npz")
    if (original["seed"], original["protocol_sha256"], original["sha256"]) != (
        seed, source_id, study.digest(data),
    ):
        raise RuntimeError(f"Original capture differs for seed {seed}")
    manifest = output / source.name
    shard = manifest.with_suffix(".npz")
    if manifest.is_file():
        row = json.loads(manifest.read_text())
        if (row["protocol_sha256"], row["sha256"]) != (identity, study.digest(shard)):
            raise RuntimeError(f"Enriched capture differs for seed {seed}")
        return row

    with np.load(data, allow_pickle=False) as archive:
        features, labels, steps = (archive[key] for key in ("features", "labels", "steps"))
    previews = np.empty((len(steps), 3 * len(PIECE_INDEX)), dtype=np.float32)
    env = study.LongCandidateTetrisEnv(max_pieces=original["pieces"])
    env.reset(seed=seed)
    try:
        index = 0
        for step in range(original["pieces"]):
            observation, legal = env.get_observation(), env.action_masks()
            student_action = study.fast_action(student, observation, legal)
            if index < len(steps) and step == steps[index]:
                bc_action = study.fast_action(bc, observation, legal)
                pair = gate_features(observation, student_action, bc_action, scale)
                if student_action == bc_action or not np.array_equal(pair, features[index]):
                    raise RuntimeError(f"Student trajectory changed at seed {seed}, step {step}")
                previews[index] = next3_features(env.core.get_public_observation())
                index += 1
            _, _, game_over, truncated, info = env.step(int(student_action))
            if game_over or truncated:
                break
        if (index, step + 1, int(info["score"]), bool(game_over)) != (
            len(steps), original["pieces"], original["score"], original["game_over"],
        ):
            raise RuntimeError(f"Student trajectory endpoint differs for seed {seed}")
    finally:
        env.close()

    temporary = shard.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=np.concatenate((features, previews), axis=1), labels=labels, steps=steps)
    temporary.replace(shard)
    row = {**original, "parent_sha256": original["sha256"],
           "protocol_sha256": identity, "sha256": study.digest(shard)}
    atomic_json(manifest, row)
    print("enriched", seed, "states", len(steps), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, previous, output = (p.resolve() for p in (args.checkout, args.capture, args.output))
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.candidate_features import FEATURE_SCALE
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.freeze()
        torch.set_num_threads(1)
        previous_protocol = previous / "protocol.json"
        source_id = study.digest(previous_protocol)
        base = json.loads(previous_protocol.read_text())
        seeds = base["train_seeds"] + base["validation_seeds"]
        status = json.loads((previous / "completed.json").read_text())
        if (status["protocol_sha256"], status["seeds"]) != (source_id, seeds):
            raise RuntimeError("Parent capture is incomplete")
        if (base["student_sha256"], base["bc_sha256"], base["teacher_protocol_sha256"]) != (
            study.digest(study.STUDENT), study.digest(study.BC), study.digest(study.RUN / "protocol.json"),
        ):
            raise RuntimeError("Frozen parent capture sources differ")
        protocol = {**base, "parent_capture_sha256": source_id, "public_next3": True,
                    "enrichment_script_sha256": study.digest(Path(__file__))}
        protocol_path = output / "protocol.json"
        if protocol_path.is_file():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Enrichment protocol changed")
        else:
            atomic_json(protocol_path, protocol)
        identity = study.digest(protocol_path)
        student = MaskablePPO.load(study.STUDENT, device="cpu")
        bc = MaskablePPO.load(study.BC, device="cpu")
        for seed in seeds:
            enrich(study, student, bc, FEATURE_SCALE, previous, output, seed, source_id, identity, atomic_json)
        atomic_json(output / "completed.json", {"protocol_sha256": identity, "seeds": seeds})


if __name__ == "__main__":
    main()
