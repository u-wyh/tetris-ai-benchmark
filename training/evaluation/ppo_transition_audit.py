"""Independent structural and public-state audit of prospective PPO snapshots."""

import gzip
import hashlib
import json
import pickle

from training.candidate_features import board_features
from training.evaluation.hybrid_queue import PPO, digest
from training.evaluation.ppo_transition_capture import (CAP, PPO_SHA256, RUN, SEEDS,
                                                          episode_path, public_hash,
                                                          snapshot_path)
from training.train_ppo import atomic_json, now


def corrected_episode_path(seed):
    return RUN / "episodes_corrected" / f"seed_{seed}.json"


def main():
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO checkpoint changed")
    files = {}
    counts = {"rapid_stack": 0, "high_control": 0, "routine_control": 0}
    rapid_seeds = set()
    corrections = []
    (RUN / "episodes_corrected").mkdir(exist_ok=True)
    for seed in SEEDS:
        path = episode_path(seed)
        raw_row = json.loads(path.read_text())
        row = {**raw_row, "states": [dict(item) for item in raw_row["states"]]}
        for item in row["states"]:
            if item["step"] == 5000 and item["stratum"] != "routine_control":
                corrections.append({"seed": seed, "step": 5000,
                                    "from": item["stratum"], "to": "routine_control",
                                    "reason": "fixed routine sample overlapped risk trigger"})
                item["stratum"] = "routine_control"
                item["inclusion_probability"] = 1.0
        if (row["seed"] != seed or row["ppo_sha256"] != PPO_SHA256 or row["cap"] != CAP
                or row["pieces"] > CAP or row["game_over"] == row["capped"]
                or len(row["states"]) > 5 or len({x["step"] for x in row["states"]}) != len(row["states"])):
            raise RuntimeError(f"Episode identity, censoring, or sample cap invalid: {seed}")
        files[str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
        strata = {name: 0 for name in counts}
        for item in row["states"]:
            kind, step = item["stratum"], item["step"]
            if kind not in counts or not 0 <= step < row["pieces"]:
                raise RuntimeError(f"Selected state identity invalid: {seed}:{step}")
            strata[kind] += 1
            counts[kind] += 1
            if kind == "rapid_stack":
                rapid_seeds.add(seed)
            snap_path = snapshot_path(seed, step)
            raw = snap_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["snapshot_sha256"]:
                raise RuntimeError(f"Snapshot checksum changed: {seed}:{step}")
            state = pickle.loads(gzip.decompress(raw))  # locally captured and checksum verified
            public = state["core"].get_public_observation()
            holes, _, height, _, _ = board_features(public["board"])
            if (state["episode_seed"] != seed or state["pieces"] != step or
                    state["max_pieces"] != CAP or public_hash(public) != item["public_sha256"] or
                    height != item["height"] or holes != item["holes"]):
                raise RuntimeError(f"Snapshot/public feature mismatch: {seed}:{step}")
            expected_probability = (min(1, 2 / row["eligible_counts"][kind])
                                    if kind != "routine_control" else 1.0)
            if item["inclusion_probability"] != expected_probability:
                raise RuntimeError(f"Reservoir probability mismatch: {seed}:{step}")
            files[str(snap_path.relative_to(RUN))] = hashlib.sha256(raw).hexdigest()
        if strata["rapid_stack"] > 2 or strata["high_control"] > 2 or strata["routine_control"] > 1:
            raise RuntimeError(f"Per-seed stratum cap exceeded: {seed}")
        corrected_path = corrected_episode_path(seed)
        if corrected_path.exists():
            if json.loads(corrected_path.read_text()) != row:
                raise RuntimeError(f"Corrected episode changed: {seed}")
        else:
            atomic_json(corrected_path, row)
        files[str(corrected_path.relative_to(RUN))] = hashlib.sha256(corrected_path.read_bytes()).hexdigest()
    for seed in range(1531000, 1531016):
        if any(RUN.rglob(f"seed_{seed}*")):
            raise RuntimeError(f"Reserved confirmation seed used: {seed}")
    atomic_json(RUN / "capture_integrity.json", {"audited_at": now(), "episodes": len(SEEDS),
                                                 "states": sum(counts.values()),
                                                 "rapid_seeds": len(rapid_seeds),
                                                 "strata": counts, "corrections": corrections,
                                                 "raw_capture_preserved": True,
                                                 "file_sha256": files})


if __name__ == "__main__":
    main()
