"""Independent, resumable V3 development and conditional confirmation queue."""
import gzip
import hashlib
import json
import os
import statistics
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, verify_models
from training.evaluation.ppo_lookahead_queue import CONFIGS, load_models
from training.hybrid_policy import HybridConfig, HybridController
from training.ppo_lookahead_policy import LookaheadController
from training.ppo_lookahead_v3 import V3Config, V3Controller
from training.evaluation.ppo_lookahead_v3_audit import RUN, policy_probs, terms
from training.train_ppo import ROOT, atomic_json, now
from training.traditional.v2 import simulate_public

VERSION = "ppo_lookahead_v3_public_1"
SMOKE = [991000, 991001]
DEV = list(range(991100, 991116))
CONFIRM = list(range(991200, 991232))
V3_CONFIGS = {"V3A": V3Config("V3A"),
              "V3B": V3Config("V3B", hold_extra_margin=0.45)}
WORKER_MODELS = None


def write_gzip_atomic(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name+".",
                                     suffix=".tmp", delete=False) as file:
        temporary = file.name
        with gzip.GzipFile(fileobj=file, mode="wb") as zipped:
            zipped.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_gzip(path):
    with gzip.open(path, "rt") as file:
        return json.load(file)


def protocol():
    verify_models()
    seeds = SMOKE+DEV+CONFIRM
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("V3 seed overlap")
    existing = [str(path) for base in (ROOT / "runs", ROOT / "reports")
                for path in base.rglob("seed_*") if path.stem.split(".")[0] in {f"seed_{s}" for s in seeds}
                and not path.is_relative_to(RUN)]
    if existing:
        raise RuntimeError(f"V3 seed collision: {existing[:5]}")
    spec = {"version": VERSION, "model_sha256": HASHES,
            "seeds": {"smoke": SMOKE, "development": DEV, "confirmation": CONFIRM},
            "variants": {key: vars(value) for key, value in V3_CONFIGS.items()},
            "s1": vars(CONFIGS["S1"]), "caps": {"smoke": 1000, "development": 20000,
                                                       "extension": 50000, "confirmation": 100000},
            "workers": 4, "stop": {"four_drop_pp": 3, "common_score_ratio": 0.95,
                                     "p10_ratio": 0.9, "p95_ms": 150}}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Frozen V3 protocol differs")
    if not path.exists():
        atomic_json(path, spec)
    return spec


def init_worker():
    global WORKER_MODELS
    WORKER_MODELS = load_models()


def event(seed, step, public, by_id, decision, selected, model, observation, mask, variant):
    original = decision["ppo_action"]
    probs = policy_probs(model, observation, mask)
    ranked = sorted(np.flatnonzero(mask).tolist(), key=lambda a: (-float(probs[a]), a))
    alternatives = {}
    for label, action in (("ppo", original), ("search", selected)):
        state, cleared = simulate_public(public, by_id[action])
        alternatives[label] = {"action": action, "rank": ranked.index(action)+1,
                               "probability": float(probs[action]), "hold": bool(by_id[action]["hold"]),
                               "cleared": cleared, "score_gain": state["score"]-public["score"],
                               "phase": state["phase"], "terms": terms(state, public, cleared)}
    digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    heights = [20-next((y for y in range(20) if public["board"][y][x] is not None), 20) for x in range(10)]
    return {"seed": seed, "step": step, "public_hash": digest,
            "public_summary": {"current": public["currentPiece"]["type"], "hold": public["hold"],
                               "next": public["next"], "heights": heights,
                               "score": public["score"], "lines": public["lines"]},
            "ppo": alternatives["ppo"], "search": alternatives["search"],
            "hold_changed": alternatives["ppo"]["hold"] != alternatives["search"]["hold"],
            "mode": variant, "trigger": decision.get("reason", "S1_search_margin"),
            "margin": decision.get("margin", CONFIGS["S1"].switch_margin),
            "score_gap": decision.get("score_gap"),
            "hold_transition": decision.get("hold_transition"),
            "rejected_hold_candidates": decision.get("rejected_hold_candidates", [])}


def run_game(models, variant, seed, cap):
    env = FastCandidateTetrisEnv(max_pieces=cap)
    controller = (V3Controller(models["ppo"], V3_CONFIGS[variant]) if variant in V3_CONFIGS else
                  LookaheadController(models["ppo"], CONFIGS["S1"]) if variant == "S1" else None)
    hybrid = HybridController(models["ppo"], models["bc"], HybridConfig("B", 12, 8)) if variant == "B12" else None
    started = time.monotonic()
    timings, events, milestones, clears = [], [], [], [0]*5
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            public = env.core.get_public_observation()
            began = time.monotonic()
            if controller:
                action, decision = controller.decide(observation, mask, public, env._placement_cache)
                if not decision["top1_retained"]:
                    by_id = {p["actionId"]: p for p in env._placement_cache}
                    events.append(event(seed, env.pieces, public, by_id, decision, action,
                                        models["ppo"], observation, mask, variant))
            elif hybrid:
                action, _ = hybrid.decide(observation, mask, public, env._placement_cache)
            else:
                action = int(models["ppo"].predict(observation, deterministic=True, action_masks=mask)[0])
            timings.append((time.monotonic()-began)*1000)
            if not mask[action]:
                raise RuntimeError("V3 queue selected illegal action")
            observation, _, over, capped, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            if info["pieces"] % 1000 == 0:
                milestones.append({"pieces": info["pieces"], "score": info["score"]})
            if over or capped:
                break
        return {"seed": seed, "variant": variant, "cap": cap, "version": VERSION,
                "model_sha256": HASHES, "pieces": info["pieces"], "score": info["score"],
                "game_over": bool(over), "capped": bool(capped), "clears": clears,
                "milestones": milestones, "replacements": events,
                "mean_ms": float(np.mean(timings)), "p95_ms": float(np.percentile(timings, 95)),
                "wall_seconds": time.monotonic()-started}
    finally:
        env.close()


def worker_game(variant, seed, cap):
    return run_game(WORKER_MODELS, variant, seed, cap)


def summarize(rows):
    pieces = [r["pieces"] for r in rows]
    return {"games": len(rows), "mean_pieces": statistics.mean(pieces),
            "p10_pieces": float(np.percentile(pieces, 10)),
            "capped": sum(r["capped"] for r in rows), "game_over": sum(r["game_over"] for r in rows),
            "mean_score": statistics.mean(r["score"] for r in rows),
            "four_fraction": sum(r["clears"][4] for r in rows)/max(1, sum(sum(r["clears"][1:]) for r in rows)),
            "top1_retention": 1-sum(len(r["replacements"]) for r in rows)/sum(pieces),
            "hold_changes": sum(sum(e["hold_changed"] for e in r["replacements"]) for r in rows),
            "replacement_fours": sum(sum(e["search"]["cleared"] == 4 for e in r["replacements"]) for r in rows),
            "p95_game_ms": float(np.percentile([r["p95_ms"] for r in rows], 95))}


def phase(name, variants, seeds, cap):
    results = {}
    for variant in variants:
        folder = RUN / name / variant
        folder.mkdir(parents=True, exist_ok=True)
        rows = {}
        pending = []
        for seed in seeds:
            path = folder / f"seed_{seed}.json.gz"
            if not path.exists():
                pending.append(seed)
                continue
            row = read_gzip(path)
            if (row.get("seed"), row.get("variant"), row.get("cap"), row.get("version"),
                row.get("model_sha256")) != (seed, variant, cap, VERSION, HASHES):
                raise RuntimeError(f"Invalid completed V3 game: {path}")
            if row["pieces"] > cap or row["game_over"] == row["capped"]:
                raise RuntimeError(f"Invalid outcome in {path}")
            rows[seed] = row
        if pending:
            with ProcessPoolExecutor(max_workers=min(4, len(pending)), mp_context=get_context("spawn"),
                                     initializer=init_worker) as pool:
                futures = {pool.submit(worker_game, variant, seed, cap): seed for seed in pending}
                for future in as_completed(futures):
                    seed = futures[future]
                    row = future.result()
                    write_gzip_atomic(folder / f"seed_{seed}.json.gz", row)
                    rows[seed] = row
                    atomic_json(RUN / "status.json", {"stage": name, "variant": variant,
                        "completed": len(rows), "total": len(seeds), "last_seed": seed,
                        "pid": os.getpid(), "updated_at": now()})
                    print(f"{name} {variant} {seed}: {row['pieces']} {row['score']}", flush=True)
        results[variant] = summarize([rows[s] for s in seeds])
        atomic_json(folder / "summary.json", results[variant])
    return results


def common_ratio(name, variant, seeds):
    ratios = []
    for seed in seeds:
        a = read_gzip(RUN / name / variant / f"seed_{seed}.json.gz")
        b = read_gzip(RUN / name / "ppo" / f"seed_{seed}.json.gz")
        index = min(a["pieces"], b["pieces"])//1000-1
        if index >= 0:
            ratios.append(a["milestones"][index]["score"]/max(1,b["milestones"][index]["score"]))
    return statistics.mean(ratios) if ratios else 0


def quality(name, results, seeds):
    base = results["ppo"]
    outcome = {}
    for variant in V3_CONFIGS:
        if variant not in results:
            continue  # An earlier safety gate may already have eliminated this variant.
        row = results[variant]
        shared = common_ratio(name, variant, seeds)
        severe = []
        for seed in seeds:
            a = read_gzip(RUN / name / variant / f"seed_{seed}.json.gz")
            b = read_gzip(RUN / name / "ppo" / f"seed_{seed}.json.gz")
            if a["game_over"] and a["pieces"] < min(5000, b["pieces"]*.5):
                severe.append(seed)
        outcome[variant] = {"common_score_ratio": shared, "severe_paired_early_death": severe,
            "passes": not severe and row["four_fraction"] >= base["four_fraction"]-.03 and
                       shared >= .95 and row["p10_pieces"] >= .9*base["p10_pieces"] and
                       row["p95_game_ms"] <= 150}
    return outcome


def run():
    try:
        protocol()
        smoke = phase("smoke_1k", ["ppo", "S1", "V3A", "V3B"], SMOKE, 1000)
        if any(smoke[v]["p95_game_ms"] > 150 for v in V3_CONFIGS):
            atomic_json(RUN / "status.json", {"stage": "completed_no_go_performance", "updated_at": now()})
            return
        dev = phase("development_20k", ["ppo", "B12", "S1", "V3A", "V3B"], DEV, 20000)
        gate = quality("development_20k", dev, DEV)
        atomic_json(RUN / "development_gate.json", gate)
        winners = [v for v in V3_CONFIGS if gate[v]["passes"] and
                   dev[v]["mean_pieces"] > dev["ppo"]["mean_pieces"]]
        if not winners:
            atomic_json(RUN / "status.json", {"stage": "completed_no_go_development", "updated_at": now()})
            return
        if dev["ppo"]["capped"] >= 12:
            long = phase("development_50k", ["ppo", "B12", *winners], DEV, 50000)
            gate = quality("development_50k", long, DEV)
            atomic_json(RUN / "extension_gate.json", gate)
            winners = [v for v in winners if gate[v]["passes"] and
                       long[v]["mean_pieces"] > long["ppo"]["mean_pieces"]]
        if not winners:
            atomic_json(RUN / "status.json", {"stage": "completed_no_go_extension", "updated_at": now()})
            return
        chosen = max(winners, key=lambda v: long[v]["mean_pieces"] if dev["ppo"]["capped"] >= 12 else dev[v]["mean_pieces"])
        atomic_json(RUN / "frozen_confirmation.json", {"chosen": chosen, "updated_at": now()})
        phase("confirmation_100k", ["ppo", "B12", chosen], CONFIRM, 100000)
        atomic_json(RUN / "status.json", {"stage": "completed", "chosen": chosen, "updated_at": now()})
    except BaseException as exc:
        atomic_json(RUN / "status.json", {"stage": "failed", "error": repr(exc),
                    "traceback": traceback.format_exc(), "updated_at": now()})
        raise


if __name__ == "__main__":
    run()
