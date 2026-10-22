"""Exploratory paired risk traces for the frozen survival-balanced development games."""

import hashlib
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.candidate_features import board_features
from training.evaluation.balanced_reward_queue import MODEL, PPO, RUN, digest
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.train_ppo import atomic_json, now

SEEDS = (1521000, 1521003, 1521004, 1521007, 1521009, 1521012, 1521013)
CAP = 20000
OUT = RUN / "failure_diagnostic_v2"
WINDOWS = (100, 500, 1000, 5000)


def windows(samples):
    result = {}
    for size in WINDOWS:
        chunk = samples[-size:]
        if not chunk:
            continue
        heights = [row[0] for row in chunk]
        result[str(size)] = {"observed": len(chunk), "height_start": heights[0],
                             "height_end": heights[-1], "height_max": max(heights),
                             "height_mean": sum(heights) / len(heights),
                             "holes_start": chunk[0][1], "holes_end": chunk[-1][1],
                             "four_clears": sum(row[2] for row in chunk)}
    return result


def replay(seed, hashes):
    torch.set_num_threads(1)
    rows = {}
    models = {name: MaskablePPO.load(path, device="cpu")
              for name, path in (("ppo", PPO), ("candidate", MODEL))}
    for name, model in models.items():
        env = LongCandidateTetrisEnv(max_pieces=CAP)
        try:
            obs, _ = env.reset(seed=seed)
            clears = [0] * 5
            samples = deque(maxlen=5000)
            first_15 = first_18 = None
            high_15_steps = high_18_steps = 0
            longest_no_four = no_four = 0
            while True:
                mask = env.action_masks()
                action = fast_action(model, obs, mask)
                if not mask[action]:
                    raise RuntimeError("Diagnostic selected an illegal action")
                obs, _, terminated, truncated, info = env.step(action)
                holes, _, height, _, _ = board_features(env.core.board)
                cleared = info["cleared_lines"]
                clears[cleared] += 1
                if height >= 15:
                    high_15_steps += 1
                    if first_15 is None:
                        first_15 = env.pieces
                if height >= 18:
                    high_18_steps += 1
                    if first_18 is None:
                        first_18 = env.pieces
                no_four = 0 if cleared == 4 else no_four + 1
                longest_no_four = max(longest_no_four, no_four)
                samples.append((height, holes, int(cleared == 4)))
                if terminated or truncated:
                    break
            original = json.loads((RUN / "evaluation/dev_20000" / name /
                                   f"seed_{seed}.json").read_text())
            if (original["seed"] != seed or original["variant"] != name or original["cap"] != CAP
                    or original["model_sha256"] != hashes or original["pieces"] != env.pieces
                    or original["score"] != env.core.score or original["capped"] != truncated
                    or original["game_over"] != terminated or original["clears_0_to_4"] != clears):
                raise RuntimeError(f"Deterministic replay differs from formal evaluation: {name} {seed}")
            rows[name] = {"pieces": env.pieces, "score": env.core.score, "capped": truncated,
                          "first_height_15": first_15, "first_height_18": first_18,
                          "steps_height_15_plus": high_15_steps,
                          "steps_height_18_plus": high_18_steps,
                          "longest_non_four_gap": longest_no_four,
                          "death_or_cap_windows": windows(list(samples))}
        finally:
            env.close()
    return {"seed": seed, "model_sha256": hashes, "cap": CAP, "replay_exact": True,
            "variants": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {"ppo": digest(PPO), "candidate": digest(MODEL)}
    if hashes["ppo"] != "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34":
        raise RuntimeError("PPO checkpoint changed")
    protocol = {"selected_after_development": True, "seeds": list(SEEDS), "cap": CAP,
                "model_sha256": hashes, "windows": list(WINDOWS),
                "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "analysis_use": "exploratory_only; not a confirmation set"}
    frozen = OUT / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Diagnostic protocol changed")
    else:
        atomic_json(frozen, protocol)
    pending = []
    for seed in SEEDS:
        path = OUT / f"seed_{seed}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["seed"] != seed or row["cap"] != CAP or row["model_sha256"] != hashes or not row["replay_exact"]:
                raise RuntimeError(f"Saved diagnostic identity changed: {path}")
        else:
            pending.append(seed)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(replay, seed, hashes): seed for seed in pending}
        for future in as_completed(futures):
            seed = futures[future]
            atomic_json(OUT / f"seed_{seed}.json", future.result())
            atomic_json(OUT / "status.json", {"stage": "running", "at": now(),
                                              "pid": os.getpid(), "completed":
                                              len(SEEDS) - len(pending) + sum(f.done() and not f.exception()
                                                                              for f in futures),
                                              "total": len(SEEDS)})
    files = {f"seed_{seed}.json": hashlib.sha256((OUT / f"seed_{seed}.json").read_bytes()).hexdigest()
             for seed in SEEDS}
    atomic_json(OUT / "integrity.json", {"count": len(files), "sha256": files})
    atomic_json(OUT / "status.json", {"stage": "completed", "at": now(),
                                      "completed": len(SEEDS), "total": len(SEEDS)})


if __name__ == "__main__":
    main()
