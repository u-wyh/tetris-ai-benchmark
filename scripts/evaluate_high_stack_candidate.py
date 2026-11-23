"""Evaluate a learned high-stack switch on paired forks and natural games."""

import argparse
from datetime import datetime
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from capture_lookahead_gate import gate_features
from enrich_lookahead_gate_next3 import next3_features
from train_lookahead_gate import model


STRESS_SEEDS = list(range(3256112540, 3256112556))
NATURAL_SEEDS = list(range(3256112570, 3256112578))
STRESS_SEEK_CAP = 12000
STRESS_HORIZON = 2048
STRESS_HEIGHT = 12
NATURAL_CAP = 100000


class SwitchPolicy:
    def __init__(self, study, student, bc, scale, height_column, saved, height):
        self.study = study
        self.student = student
        self.bc = bc
        self.scale = scale
        self.height_column = height_column
        self.mean = saved["mean"].numpy()
        self.std = saved["std"].numpy()
        self.gate = model(len(self.mean)).eval()
        self.gate.load_state_dict(saved["model"])
        self.threshold = saved["threshold"]
        self.minimum_height = height

    def height(self, observation, action):
        return int(observation["candidate"][action, self.height_column])

    @torch.no_grad()
    def choose(self, env, observation, legal, original):
        alternate = self.study.fast_action(self.bc, observation, legal)
        if alternate == original:
            return original, None
        features = np.concatenate((gate_features(observation, original, alternate, self.scale),
                                   next3_features(env.core.get_public_observation())))
        probability = self.gate(torch.from_numpy((features - self.mean) / self.std)[None]).sigmoid().item()
        return (alternate if probability >= self.threshold else original), probability


def advance(env, action, legal):
    if not legal[action]:
        raise RuntimeError(f"Policy returned an illegal action: {action}")
    _, _, over, truncated, info = env.step(int(action))
    return bool(over), bool(truncated), int(info["score"])


def fork(study, snapshot, first, student):
    # Each branch receives its own copy of the identical Core and RNG state.
    start = snapshot["pieces"]
    cap = start + STRESS_HORIZON
    env = study.LongCandidateTetrisEnv(max_pieces=cap)
    env.set_state({**snapshot, "max_pieces": cap})
    initial_score = int(env.core.score)
    scores = [0]
    try:
        for step in range(STRESS_HORIZON):
            legal = env.action_masks()
            action = (first if step == 0 else study.fast_action(student, env.get_observation(), legal))
            over, truncated, score = advance(env, action, legal)
            scores.append(score - initial_score)
            if over or truncated:
                break
        return scores, over
    finally:
        env.close()


def stress_pair(study, switch, rollout, seed):
    env = study.LongCandidateTetrisEnv(max_pieces=STRESS_SEEK_CAP)
    env.reset(seed=seed)
    eligible = 0
    try:
        for step in range(STRESS_SEEK_CAP):
            observation, legal = env.get_observation(), env.action_masks()
            proposal = study.fast_action(rollout, observation, legal)
            height = switch.height(observation, proposal)
            if height >= STRESS_HEIGHT:
                eligible += 1
                original = study.fast_action(switch.student, observation, legal)
                alternate, probability = switch.choose(env, observation, legal, original)
                if alternate != original:
                    snapshot = env.get_state()
                    fingerprint = sha256(pickle.dumps(snapshot, protocol=5)).hexdigest()
                    a, a_over = fork(study, snapshot, original, switch.student)
                    b, b_over = fork(study, snapshot, alternate, switch.student)
                    if sha256(pickle.dumps(snapshot, protocol=5)).hexdigest() != fingerprint:
                        raise RuntimeError("A branch mutated the shared initial snapshot")
                    common = min(len(a), len(b)) - 1
                    return {
                        "seed": seed, "found": True, "seek_pieces": step, "eligible_high_boards": eligible,
                        "height": height, "probability": probability, "snapshot_sha256": fingerprint,
                        "student_first": int(original), "gate_first": int(alternate), "common_pieces": common,
                        "student": {"pieces": len(a) - 1, "game_over": a_over, "common_score": a[common]},
                        "gate": {"pieces": len(b) - 1, "game_over": b_over, "common_score": b[common]},
                    }
            over, truncated, _ = advance(env, proposal, legal)
            if over or truncated:
                break
        return {"seed": seed, "found": False, "seek_pieces": step + 1,
                "eligible_high_boards": eligible, "rollout_game_over": over}
    finally:
        env.close()


def natural_game(study, switch, seed, enabled):
    env = study.LongCandidateTetrisEnv(max_pieces=NATURAL_CAP)
    env.reset(seed=seed)
    scores = [0]
    high_boards = switches = 0
    try:
        for _ in range(NATURAL_CAP):
            observation, legal = env.get_observation(), env.action_masks()
            original = study.fast_action(switch.student, observation, legal)
            action = original
            if switch.height(observation, original) >= switch.minimum_height:
                high_boards += 1
                if enabled:
                    action, _ = switch.choose(env, observation, legal, original)
                    switches += int(action != original)
            over, truncated, score = advance(env, action, legal)
            scores.append(score)
            if over or truncated:
                break
        return scores, {"pieces": len(scores) - 1, "game_over": over,
                        "high_boards": high_boards, "switches": switches}
    finally:
        env.close()


def natural_pair(study, switch, seed):
    a, original = natural_game(study, switch, seed, False)
    b, candidate = natural_game(study, switch, seed, True)
    common = min(len(a), len(b)) - 1
    return {"seed": seed, "common_pieces": common,
            "student": {**original, "common_score": a[common]},
            "gate": {**candidate, "common_score": b[common]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("stress", "natural"), required=True)
    args = parser.parse_args()
    checkout, fit, output = (p.resolve() for p in (args.checkout, args.fit, args.output))
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from training import lookahead_long_validation as study
    from training.candidate_features import FEATURE_NAMES, FEATURE_SCALE
    from training.train_ppo import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    with (output / "replay.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = study.freeze()
        torch.set_num_threads(1)
        fit_path, checkpoint = fit / "result.json", fit / "gate.pt"
        trained = json.loads(fit_path.read_text())
        sources = trained["protocol"]["sources"]
        if (not trained["calibration"]["meets_precision_target"]
                or study.digest(checkpoint) != trained["checkpoint_sha256"]
                or (sources["student"], sources["bc"], sources["rollout_ppo"]) != (
                    study.digest(study.STUDENT), study.digest(study.BC), study.digest(study.PPO),
                )):
            raise RuntimeError("High-stack fit identity or admission differs")
        seeds = STRESS_SEEDS if args.mode == "stress" else NATURAL_SEEDS
        used = trained["protocol"]["train_seeds"] + trained["protocol"]["validation_seeds"]
        if set(seeds).intersection(used):
            raise RuntimeError("Evaluation overlaps training or calibration seeds")
        specification = {
            "mode": args.mode, "seeds": seeds, "stress_seek_cap": STRESS_SEEK_CAP,
            "stress_horizon": STRESS_HORIZON, "stress_height": STRESS_HEIGHT, "natural_cap": NATURAL_CAP,
            "switch_height": trained["protocol"]["minimum_stack_height"],
            "teacher_git": frozen["git"], "checkpoint_sha256": trained["checkpoint_sha256"],
            "fit_result_sha256": study.digest(fit_path), "script_sha256": study.digest(Path(__file__)),
        }
        protocol, result = output / "protocol.json", output / "result.json"
        if protocol.is_file():
            if json.loads(protocol.read_text()) != specification:
                raise RuntimeError("Existing evaluation protocol differs")
        else:
            atomic_json(protocol, specification)
        identity = study.digest(protocol)
        if result.is_file():
            if json.loads(result.read_text())["protocol"] != specification:
                raise RuntimeError("Existing summary differs")
            print("completed evaluation already verified", flush=True)
            return

        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        student = MaskablePPO.load(study.STUDENT, device="cpu")
        bc = MaskablePPO.load(study.BC, device="cpu")
        saved = torch.load(checkpoint, weights_only=True)
        if saved["threshold"] != trained["calibration"]["threshold"]:
            raise RuntimeError("Published threshold differs from the checkpoint")
        switch = SwitchPolicy(study, student, bc, FEATURE_SCALE, FEATURE_NAMES.index("max_height_before"),
                              saved, specification["switch_height"])
        rollout = MaskablePPO.load(study.PPO, device="cpu") if args.mode == "stress" else None
        rows = []
        for seed in seeds:
            shard = output / f"seed_{seed}.json"
            if shard.is_file():
                row = json.loads(shard.read_text())
                if (row["seed"], row["protocol_sha256"]) != (seed, identity):
                    raise RuntimeError(f"Existing record differs for seed {seed}")
            else:
                row = (stress_pair(study, switch, rollout, seed) if args.mode == "stress"
                       else natural_pair(study, switch, seed))
                row["protocol_sha256"] = identity
                atomic_json(shard, row)
            rows.append(row)
            print(args.mode, row, flush=True)
        atomic_json(result, {
            "protocol": specification, "started": started,
            "ended": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "rows": rows,
        })


if __name__ == "__main__":
    main()
