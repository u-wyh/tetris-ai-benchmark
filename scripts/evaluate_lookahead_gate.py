"""Run a longer paired check of the already trained Next3 gate."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import torch
from sb3_contrib import MaskablePPO

from train_lookahead_gate import evaluate, model


SEEDS = list(range(3256112400, 3256112406))
CAP = 12000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, fit, output = (p.resolve() for p in (args.checkout, args.fit, args.output))
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.candidate_features import FEATURE_SCALE
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "replay.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = study.freeze()
        torch.set_num_threads(1)
        fit_result_path, checkpoint = fit / "result.json", fit / "gate.pt"
        fit_result = json.loads(fit_result_path.read_text())
        if (not fit_result["protocol"]["public_next3"] or not fit_result["calibration"]["meets_precision_target"]
                or study.digest(checkpoint) != fit_result["checkpoint_sha256"]):
            raise RuntimeError("Next3 gate does not match the completed fit")
        previous_seeds = {
            seed for field in ("train_seeds", "calibration_seeds", "replay_seeds")
            for seed in fit_result["protocol"][field]
        }
        if previous_seeds.intersection(SEEDS):
            raise RuntimeError("Longer replay seeds overlap previous samples")
        protocol = {
            "seeds": SEEDS, "cap": CAP, "checkpoint_sha256": fit_result["checkpoint_sha256"],
            "fit_result_sha256": study.digest(fit_result_path), "frozen_teacher_git": frozen["git"],
            "script_sha256": study.digest(Path(__file__)),
            "replay_helper_sha256": study.digest(Path(__file__).with_name("train_lookahead_gate.py")),
        }
        protocol_path = output / "protocol.json"
        if protocol_path.is_file():
            if json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError("Existing replay protocol differs")
        else:
            atomic_json(protocol_path, protocol)
        identity = study.digest(protocol_path)
        result_path = output / "result.json"
        if result_path.is_file():
            saved = json.loads(result_path.read_text())
            if saved["protocol"] != protocol:
                raise RuntimeError("Existing replay summary differs")
            print("completed replay already verified", flush=True)
            return
        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        student = MaskablePPO.load(study.STUDENT, device="cpu")
        bc = MaskablePPO.load(study.BC, device="cpu")
        saved = torch.load(checkpoint, weights_only=True)
        mean, std = saved["mean"].numpy(), saved["std"].numpy()
        gate = model(len(mean)).eval()
        gate.load_state_dict(saved["model"])
        pairs = []
        for seed in SEEDS:
            shard = output / f"seed_{seed}.json"
            if shard.is_file():
                row = json.loads(shard.read_text())
                if (row["seed"], row["protocol_sha256"]) != (seed, identity):
                    raise RuntimeError(f"Existing replay differs for seed {seed}")
            else:
                original, original_metrics = evaluate(study, student, bc, FEATURE_SCALE, seed, cap=CAP)
                candidate, candidate_metrics = evaluate(
                    study, student, bc, FEATURE_SCALE, seed, gate, (mean, std), saved["threshold"],
                    preview=True, cap=CAP,
                )
                common = min(len(original), len(candidate)) - 1
                row = {
                    "seed": seed, "protocol_sha256": identity, "common_pieces": common,
                    "student_score": original[common], "gate_score": candidate[common],
                    "student": original_metrics, "gate": candidate_metrics,
                }
                atomic_json(shard, row)
            pairs.append(row)
            print("paired replay", row, flush=True)
        atomic_json(result_path, {
            "started": started, "ended": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "protocol": protocol, "pairs": pairs,
        })


if __name__ == "__main__":
    main()
