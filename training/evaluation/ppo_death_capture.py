"""Frozen PPO30M long-game trajectories with sparse, replayable Core checkpoints."""
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
from sb3_contrib import MaskablePPO

from training.candidate_features import board_features
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import PPO, HASHES, digest, verify_models
from training.evaluation.ppo_lookahead_v3_audit import policy_probs
from training.train_ppo import ROOT, atomic_json, now
from training.traditional.v2 import features

RUN = ROOT / "runs/ppo30m_death_analysis"
VERSION = "ppo30m_death_trace_v1"
SEEDS = list(range(992000, 992032))
EXTRA = list(range(992032, 992064))
CAP = 100000
CHECKPOINT_EVERY = 1000
WORKER_MODEL = None


def public_hash(public):
    return hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_gzip(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name+".",
                                     suffix=".tmp", delete=False) as file:
        temporary = file.name
        with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=3) as zipped:
            zipped.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def protocol():
    verify_models()
    if digest(PPO) != HASHES["ppo"]:
        raise RuntimeError("Frozen PPO SHA256 changed")
    existing = []
    names = {f"seed_{seed}" for seed in SEEDS+EXTRA}
    for base in (ROOT / "runs", ROOT / "reports"):
        for path in base.rglob("seed_*.*"):
            if path.stem.split(".")[0] in names and not path.is_relative_to(RUN):
                existing.append(str(path))
    if existing:
        raise RuntimeError(f"Diagnostic seed already used: {existing[:5]}")
    spec = {"version": VERSION, "ppo_sha256": HASHES["ppo"], "bc_sha256": HASHES["bc"],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "initial_seeds": SEEDS, "reserved_extra_seeds": EXTRA, "cap": CAP,
            "workers": 4, "checkpoint_every": CHECKPOINT_EVERY,
            "extension_rule": "Only if initial batch has fewer than 12 true deaths; add enough reserved seeds to reach 12 deaths, max 64 games",
            "trace": "one public pre-action state and actual post-action outcome per piece; compressed JSONL",
            "outcome": "100K cap is right-censored"}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Frozen death-analysis protocol changed")
    if not path.exists():
        atomic_json(path, spec)
    return spec


def load_ppo():
    import torch
    torch.set_num_threads(1)
    return MaskablePPO.load(str(PPO), device="cpu")


def worker_init():
    global WORKER_MODEL
    WORKER_MODEL = load_ppo()


def trace_paths(seed):
    folder = RUN / "traces"
    return (folder / f"seed_{seed}.jsonl.gz", folder / f"seed_{seed}.checkpoints.pkl.gz",
            folder / f"seed_{seed}.json")


def capture(seed, model=None):
    trace_path, state_path, meta_path = trace_paths(seed)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("seed") != seed or meta.get("version") != VERSION or
            meta.get("ppo_sha256") != HASHES["ppo"] or meta.get("cap") != CAP or
            digest(trace_path) != meta.get("trace_sha256") or
            digest(state_path) != meta.get("checkpoints_sha256") or
            meta.get("game_over") == meta.get("capped")):
            raise RuntimeError(f"Invalid completed death-analysis episode: {seed}")
        return meta
    model = model or load_ppo()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    checkpoints = {}
    four_history, hold_history = [], []
    four_prefix, hold_prefix = [0], [0]
    started = time.monotonic()
    trace_tmp = None
    try:
        obs, _ = env.reset(seed=seed)
        with tempfile.NamedTemporaryFile(mode="wb", dir=trace_path.parent,
                                         prefix=trace_path.name+".", suffix=".tmp", delete=False) as file:
            trace_tmp = file.name
            with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=3) as zipped:
                while True:
                    step = env.pieces
                    public = env.core.get_public_observation()
                    mask = env.action_masks()
                    by_id = {p["actionId"]: p for p in env._placement_cache}
                    probabilities = policy_probs(model, obs, mask)
                    ranked = sorted(np.flatnonzero(mask).tolist(), key=lambda a: (-float(probabilities[a]), a))
                    action = ranked[0]
                    if action not in by_id:
                        raise RuntimeError("PPO selected action outside official BFS")
                    if step % CHECKPOINT_EVERY == 0:
                        checkpoints[step] = {"state": env.get_state(), "public_hash": public_hash(public),
                                             "rng_state": env.core.rng.state}
                        predicted = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
                        if predicted != action:
                            raise RuntimeError(f"PPO deterministic action mismatch at {seed}:{step}")
                    structural = features(public["board"])
                    holes, total_h, max_h, _, _ = board_features(public["board"])
                    heights = [20-next((y for y in range(20) if public["board"][y][x] is not None), 20)
                               for x in range(10)]
                    well_depths = [max(0, min(20 if x == 0 else heights[x-1],
                                               20 if x == 9 else heights[x+1])-heights[x])
                                   for x in range(10)]
                    birthplace = sum(public["board"][y][x] is not None for y in range(4) for x in range(3, 7))
                    four_windows = {str(window): four_prefix[-1]-four_prefix[max(0, step-window)]
                                    for window in (100, 500, 1000)}
                    hold_windows = {str(window): hold_prefix[-1]-hold_prefix[max(0, step-window)]
                                    for window in (100, 500, 1000)}
                    row = {"seed": seed, "step": step, "public_hash": public_hash(public),
                           "rng_state": env.core.rng.state, "score": public["score"], "lines": public["lines"],
                           "max_height": max_h, "total_height": total_h, "holes": holes,
                           "covered_holes": structural["covered_holes"],
                           "row_transitions": structural["row_transitions"],
                           "column_transitions": structural["column_transitions"],
                           "bumpiness": structural["bumpiness"],
                           "wells": structural["wells"], "well_depths": well_depths,
                           "deepest_well_column": int(np.argmax(well_depths)),
                           "deepest_well_depth": max(well_depths), "heights": heights,
                           "birth_zone": birthplace,
                           "four_rate_100": four_windows["100"]/max(1, min(step, 100)),
                           "hold_rate_100": hold_windows["100"]/max(1, min(step, 100)),
                           "four_counts": four_windows, "hold_counts": hold_windows,
                           "current": public["currentPiece"]["type"], "hold": public["hold"],
                           "next": public["next"], "action": action,
                           "action_probability": float(probabilities[action]),
                           "top5": [{"action": a, "probability": float(probabilities[a])}
                                    for a in ranked[:5]], "used_hold": bool(by_id[action]["hold"])}
                    obs, _, over, capped, info = env.step(action)
                    row.update({"score_after": info["score"], "lines_after": info["lines"],
                                "cleared": info["cleared_lines"], "game_over_after": bool(over),
                                "capped_after": bool(capped)})
                    zipped.write((json.dumps(row, separators=(",", ":"))+"\n").encode())
                    four_history.append(info["cleared_lines"] == 4)
                    hold_history.append(bool(by_id[action]["hold"]))
                    four_prefix.append(four_prefix[-1]+int(four_history[-1]))
                    hold_prefix.append(hold_prefix[-1]+int(hold_history[-1]))
                    if over or capped:
                        break
            file.flush()
            os.fsync(file.fileno())
        _atomic_gzip(state_path, pickle.dumps(checkpoints, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(trace_tmp, trace_path)
        trace_tmp = None
        meta = {"seed": seed, "version": VERSION, "ppo_sha256": HASHES["ppo"],
                "cap": CAP, "pieces": info["pieces"], "game_over": bool(over),
                "capped": bool(capped), "score": info["score"], "lines": info["lines"],
                "four_clears": sum(four_history), "hold_uses": sum(hold_history),
                "checkpoint_count": len(checkpoints), "trace_sha256": digest(trace_path),
                "checkpoints_sha256": digest(state_path), "wall_seconds": time.monotonic()-started}
        atomic_json(meta_path, meta)
        return meta
    finally:
        if trace_tmp is not None and os.path.exists(trace_tmp):
            os.unlink(trace_tmp)
        env.close()


def worker(seed):
    return capture(seed, WORKER_MODEL)


def phase(seeds):
    rows = {}
    # Each worker loads one immutable PPO copy and writes only its own seed files.
    with ProcessPoolExecutor(max_workers=min(4, len(seeds)), mp_context=get_context("spawn"),
                             initializer=worker_init) as pool:
        futures = {pool.submit(worker, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed = futures[future]
            row = future.result()
            rows[seed] = row
            atomic_json(RUN / "status.json", {"stage": "capturing", "completed_this_phase": len(rows),
                        "phase_total": len(seeds), "last_seed": seed, "pid": os.getpid(),
                        "updated_at": now()})
            print(f"seed {seed}: {row['pieces']} pieces, death={row['game_over']}", flush=True)
    return rows


def run():
    try:
        protocol()
        initial = phase(SEEDS)
        deaths = sum(r["game_over"] for r in initial.values())
        added = []
        if deaths < 12:
            # Pre-registered extension, one batch at a time; never exceed 64 games.
            for start in range(0, len(EXTRA), 4):
                batch = EXTRA[start:start+4]
                rows = phase(batch)
                added.extend(batch)
                deaths += sum(r["game_over"] for r in rows.values())
                if deaths >= 12:
                    break
        atomic_json(RUN / "capture_completed.json", {"initial_seeds": SEEDS,
                    "extra_seeds": added, "true_deaths": deaths,
                    "right_censored": len(SEEDS)+len(added)-deaths, "updated_at": now()})
        atomic_json(RUN / "status.json", {"stage": "capture_completed", "true_deaths": deaths,
                    "games": len(SEEDS)+len(added), "updated_at": now()})
    except BaseException as exc:
        atomic_json(RUN / "status.json", {"stage": "capture_failed", "error": repr(exc),
                    "traceback": traceback.format_exc(), "updated_at": now()})
        raise


if __name__ == "__main__":
    run()
