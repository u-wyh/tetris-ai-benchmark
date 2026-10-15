"""Bounded, resumable same-Core/RNG first-action branches with frozen PPO continuation."""
import gzip
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from training.candidate_features import board_features
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, digest, load_models
from training.evaluation.ppo_death_capture import CAP, RUN, public_hash
from training.evaluation.ppo_death_replay import load_episode
from training.evaluation.ppo_lookahead_v3_audit import policy_probs
from training.train_ppo import atomic_bytes, atomic_json
from training.traditional.v2 import simulate_public

DISTANCES = (30, 80, 200, 400, 700, 900, 2000, 4000)
MAX_STATES = 120
MAX_CANDIDATES = 6
WORKER_MODELS = None


def tier(distance):
    if distance <= 100: return "1-100"
    if distance <= 500: return "101-500"
    if distance <= 1000: return "501-1000"
    return "1001-5000"


def selection():
    path = RUN / "counterfactual_selection.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["ppo_sha256"] != HASHES["ppo"] or len(record["states"]) > MAX_STATES:
            raise RuntimeError("Frozen counterfactual selection mismatch")
        return record
    analysis = json.loads((RUN / "death_analysis.json").read_text())
    pool = []
    for episode in analysis["death_episodes"]:
        seed, pieces = episode["seed"], episode["pieces"]
        for distance in DISTANCES:
            step = pieces-distance
            if step >= 0 and all(abs(step-x["step"]) >= 50 for x in pool if x["seed"] == seed):
                pool.append({"seed": seed, "step": step, "death_distance": distance,
                             "tier": tier(distance), "source_trace_sha256": episode["trace_sha256"]})
    # Reserve equal slots for distance layers, then round-robin independent seeds.
    chosen, remaining = [], []
    for label in ("1-100", "101-500", "501-1000", "1001-5000"):
        by_seed = {seed: sorted([x for x in pool if x["seed"] == seed and x["tier"] == label],
                               key=lambda x: DISTANCES.index(x["death_distance"]))
                   for seed in sorted({x["seed"] for x in pool})}
        layer = []
        while any(by_seed.values()):
            for seed in by_seed:
                if by_seed[seed]:
                    layer.append(by_seed[seed].pop(0))
        chosen.extend(layer[:MAX_STATES//4])
        remaining.extend(layer[MAX_STATES//4:])
    chosen.extend(remaining[:MAX_STATES-len(chosen)])
    record = {"ppo_sha256": HASHES["ppo"], "states": chosen,
              "selection": "fixed_death_distances_round_robin_before_branch_outcomes",
              "max_states": MAX_STATES, "max_candidates": MAX_CANDIDATES,
              "initial_horizon": 1000, "conditional_horizon": 5000}
    atomic_json(path, record)
    return record


def _snapshot_path(seed, step):
    return RUN / "counterfactual_states" / f"seed_{seed}_step_{step}.pkl.gz"


def materialize_state(item, rows, checkpoints):
    seed, target = item["seed"], item["step"]
    path = _snapshot_path(seed, target)
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["source_trace_sha256"] != item["source_trace_sha256"] or digest(path) != meta["sha256"]:
            raise RuntimeError(f"Counterfactual snapshot changed: {path}")
        return meta
    previous = max(step for step in checkpoints if step <= target)
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=seed)
        env.set_state(checkpoints[previous]["state"])
        for step in range(previous, target):
            row = rows[step]
            if public_hash(env.core.get_public_observation()) != row["public_hash"]:
                raise RuntimeError(f"Trace replay diverged at {seed}:{step}")
            action = row["action"]
            if not env.action_masks()[action]:
                raise RuntimeError("Archived action is no longer legal")
            env.step(action)
        public = env.core.get_public_observation()
        if public_hash(public) != rows[target]["public_hash"]:
            raise RuntimeError(f"Target snapshot hash mismatch: {seed}:{target}")
        state = env.get_state()
        path.parent.mkdir(parents=True, exist_ok=True)
        import io
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=3) as file:
            file.write(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        atomic_bytes(path, buffer.getvalue())
        meta = {"seed": seed, "step": target, "source_trace_sha256": item["source_trace_sha256"],
                "public_hash": rows[target]["public_hash"], "rng_state": env.core.rng.state,
                "sha256": digest(path)}
        atomic_json(meta_path, meta)
        return meta
    finally:
        env.close()


def candidates(item, rows, models):
    seed, step = item["seed"], item["step"]
    path = RUN / "counterfactual_candidates" / f"seed_{seed}_step_{step}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["source_trace_sha256"] != item["source_trace_sha256"]:
            raise RuntimeError("Frozen candidate source changed")
        return result
    with gzip.open(_snapshot_path(seed, step), "rb") as file:
        state = pickle.load(file)  # Trusted local experiment snapshot.
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=seed)
        env.set_state(state)
        public, obs, mask = env.core.get_public_observation(), env.get_observation(), env.action_masks()
        by_id = {p["actionId"]: p for p in env._placement_cache}
        top5 = rows[step]["top5"]
        if top5[0]["action"] != rows[step]["action"]:
            raise RuntimeError("Archived PPO Top-1 differs from action")
        safety = {}
        for action, placement in by_id.items():
            child, _ = simulate_public(public, placement)
            holes, _, height, _, _ = board_features(child["board"])
            safety[action] = (child["phase"] == "over", height, holes, action)
        top45 = min(top5[3:5], key=lambda x: safety[x["action"]])
        bc = int(models["bc"].predict(obs, deterministic=True, action_masks=mask)[0])
        safe = min(by_id, key=lambda action: safety[action])
        ordered = [(x["action"], f"ppo_top_{i+1}") for i,x in enumerate(top5[:3])]
        ordered += [(top45["action"], "ppo_top_4_or_5_safer"), (bc, "bc"), (safe, "public_safety")]
        unique = {}
        for action, source in ordered:
            if not mask[action] or action not in by_id:
                raise RuntimeError("Counterfactual candidate is illegal")
            unique.setdefault(action, []).append(source)
        probabilities = policy_probs(models["ppo"], obs, mask)
        ranked_ids = sorted(by_id, key=lambda action: (-float(probabilities[action]), action))
        ranked = {action:(i+1,float(probabilities[action])) for i,action in enumerate(ranked_ids)}
        if ranked_ids[:5] != [x["action"] for x in top5]:
            raise RuntimeError("Reconstructed PPO action ranking changed")
        choices = [{"action": action, "sources": sources, "ppo_rank": ranked[action][0] if action in ranked else None,
                    "ppo_probability": ranked[action][1] if action in ranked else None,
                    "changes_hold": bool(by_id[action]["hold"]) != bool(by_id[rows[step]["action"]]["hold"])}
                   for action,sources in unique.items()]
        result = {"seed": seed, "step": step, "death_distance": item["death_distance"],
                  "tier": item["tier"], "source_trace_sha256": item["source_trace_sha256"],
                  "snapshot_sha256": digest(_snapshot_path(seed, step)),
                  "public_hash": rows[step]["public_hash"], "ppo_action": rows[step]["action"],
                  "start_score": rows[step]["score"], "choices": choices[:MAX_CANDIDATES]}
        atomic_json(path, result)
        return result
    finally:
        env.close()


def prepare(models=None):
    models = models or load_models()
    selected = selection()["states"]
    by_seed = {}
    for item in selected:
        by_seed.setdefault(item["seed"], []).append(item)
    total = 0
    for seed, items in by_seed.items():
        meta, rows, checkpoints = load_episode(seed)
        for item in items:
            if meta["trace_sha256"] != item["source_trace_sha256"]:
                raise RuntimeError("Selected trace changed")
            materialize_state(item, rows, checkpoints)
            total += len(candidates(item, rows, models)["choices"])
        del rows, checkpoints
    atomic_json(RUN / "counterfactual_prepared.json", {"states": len(selected), "branches_1000": total})
    return len(selected), total


def init_worker():
    global WORKER_MODELS
    WORKER_MODELS = load_models()


def _branch_path(seed, step, action, horizon):
    return RUN / f"counterfactual_{horizon}" / f"seed_{seed}" / f"step_{step}_action_{action}.json"


def branch(seed, step, choice, horizon, models):
    action = choice["action"]
    path = _branch_path(seed, step, action, horizon)
    seal = path.with_suffix(".sha256.json")
    if seal.exists():
        record = json.loads(seal.read_text())
        if digest(path) != record["sha256"] or record["snapshot_sha256"] != digest(_snapshot_path(seed, step)):
            raise RuntimeError(f"Completed branch changed: {path}")
        return json.loads(path.read_text())
    with gzip.open(_snapshot_path(seed, step), "rb") as file:
        snapshot = pickle.load(file)
    # A counterfactual may pass the observational episode's 100K cap.
    branch_cap = max(CAP, step+horizon+1)
    snapshot["max_pieces"] = branch_cap
    env = FastCandidateTetrisEnv(max_pieces=branch_cap)
    started = time.monotonic()
    try:
        env.reset(seed=seed)
        env.set_state(snapshot)
        obs = env.get_observation()
        candidate_record = json.loads((RUN / "counterfactual_candidates" /
            f"seed_{seed}_step_{step}.json").read_text())
        if (candidate_record["snapshot_sha256"] != digest(_snapshot_path(seed, step)) or
                candidate_record["public_hash"] != public_hash(env.core.get_public_observation())):
            raise RuntimeError("Branch start state changed")
        if not env.action_masks()[action]:
            raise RuntimeError("Branch first action illegal")
        start_public = env.core.get_public_observation()
        initial_score = env.core.score
        scores, fours, risk = [], [], []
        four = 0
        hold_next_after_first = None
        for offset in range(horizon):
            selected = action if offset == 0 else int(models["ppo"].predict(
                obs, deterministic=True, action_masks=env.action_masks())[0])
            if not env.action_masks()[selected]:
                raise RuntimeError("PPO continuation illegal")
            obs, _, over, capped, info = env.step(selected)
            four += info["cleared_lines"] == 4
            scores.append(info["score"])
            fours.append(four)
            if offset == 0:
                post = env.core.get_public_observation()
                hold_next_after_first = {"current": post["currentPiece"]["type"] if post["currentPiece"] else None,
                                         "hold": post["hold"], "next": post["next"]}
            if offset % 100 == 0 or over or capped:
                public = env.core.get_public_observation()
                holes, _, height, _, wells = board_features(public["board"])
                risk.append({"offset": offset+1, "height": height, "holes": holes, "wells": wells,
                             "birth_zone": sum(public["board"][y][x] is not None for y in range(4) for x in range(3,7))})
            if over or capped:
                break
        result = {"seed": seed, "step": step, "first_action": action, "horizon": horizon,
                  "snapshot_sha256": digest(_snapshot_path(seed, step)),
                  "initial_public_hash": public_hash(start_public), "initial_score": initial_score,
                  "choice": choice, "pieces": len(scores), "game_over": bool(over),
                  "censored": not over, "score_gain": scores[-1]-initial_score,
                  "scores": scores, "four_counts": fours, "risk_checkpoints": risk,
                  "hold_next_after_first": hold_next_after_first,
                  "wall_seconds": time.monotonic()-started}
        atomic_json(path, result)
        atomic_json(seal, {"sha256": digest(path), "snapshot_sha256": result["snapshot_sha256"]})
        return result
    finally:
        env.close()


def worker_task(args):
    return branch(*args, WORKER_MODELS)


def _tasks(horizon):
    tasks = []
    for item in selection()["states"]:
        seed, step = item["seed"], item["step"]
        record = json.loads((RUN / "counterfactual_candidates" / f"seed_{seed}_step_{step}.json").read_text())
        if horizon == 1000:
            choices = record["choices"]
        else:
            baseline = next(c for c in record["choices"] if c["action"] == record["ppo_action"])
            results = {c["action"]:json.loads(_branch_path(seed,step,c["action"],1000).read_text())
                       for c in record["choices"]}
            base = results[baseline["action"]]
            eligible = []
            for choice in record["choices"]:
                if choice["action"] == baseline["action"]:
                    continue
                alt = results[choice["action"]]
                common = min(base["pieces"], alt["pieces"])
                ratio = ((alt["scores"][common-1]-record["start_score"])/
                         max(1,base["scores"][common-1]-record["start_score"]))
                four_ok = base["four_counts"][common-1] < 3 or alt["four_counts"][common-1] >= .9*base["four_counts"][common-1]
                if alt["pieces"] == 1000 and ratio >= .95 and four_ok and (base["game_over"] or item["death_distance"] > 1000):
                    eligible.append((ratio,choice))
            eligible.sort(key=lambda x:(-x[0],x[1]["action"]))
            choices = [baseline]+[choice for _,choice in eligible[:2]] if eligible else []
        tasks += [(seed,step,choice,horizon) for choice in choices]
    return tasks


def evaluate(horizon):
    tasks = _tasks(horizon)
    if len(tasks) > MAX_STATES*(MAX_CANDIDATES if horizon == 1000 else 3):
        raise RuntimeError("Counterfactual branch budget exceeded")
    completed = 0
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=init_worker) as pool:
        futures = {pool.submit(worker_task,args): args for args in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            atomic_json(RUN / "status.json", {"stage": f"counterfactual_{horizon}",
                        "completed": completed, "total": len(tasks),
                        "last_seed": result["seed"], "last_step": result["step"]})
    atomic_json(RUN / f"counterfactual_{horizon}_completed.json", {"branches": completed,
                "states": len({(seed,step) for seed,step,_,_ in tasks}), "horizon": horizon})
    return completed
