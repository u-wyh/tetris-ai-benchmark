"""Frozen, resumable lookahead smoke/development/conditional confirmation queue."""
import json
import os
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import BC, HASHES, PPO, digest, verify_models
from training.hybrid_policy import HybridConfig, HybridController
from training.ppo_lookahead_policy import LookaheadConfig, LookaheadController
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/ppo30m_lookahead"
SMOKE = [990000, 990001]
DEV = list(range(990100, 990116))
CONFIRM = list(range(990200, 990232))
CONFIGS = {"S1": LookaheadConfig("S1"), "S2": LookaheadConfig("S2")}
VERSION = "ppo_top5_public_lookahead_v1"
WORKER_MODELS = None


def protocol():
    verify_models()
    all_seeds = SMOKE + DEV + CONFIRM
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Seed groups overlap")
    existing = [str(p) for base in (ROOT / "runs", ROOT / "reports")
                for p in base.rglob("seed_*.json") if p.stem in {f"seed_{s}" for s in all_seeds}
                and not p.is_relative_to(RUN)]
    if existing:
        raise RuntimeError(f"Seed collision: {existing[:5]}")
    spec = {"version": VERSION, "model_sha256": HASHES,
            "seeds": {"smoke": SMOKE, "development": DEV, "confirmation": CONFIRM},
            "variants": {v: vars(c) for v, c in CONFIGS.items()},
            "smoke_cap": 1000, "development_cap": 20000, "confirmation_cap": 100000,
            "smoke_p95_limit_ms": 150.0,
            "go": {"four_clear_drop_pp": 3, "minimum_common_score_ratio": 0.95,
                   "minimum_mean_survival_gain": 0.03, "minimum_p10_ratio": 0.9}}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Frozen lookahead protocol changed")
    if not path.exists(): atomic_json(path, spec)
    return spec


def load_models():
    import torch
    torch.set_num_threads(1)
    return {"ppo": MaskablePPO.load(str(PPO), device="cpu"),
            "bc": MaskablePPO.load(str(BC), device="cpu")}


def worker_init():
    global WORKER_MODELS
    WORKER_MODELS = load_models()


def worker_game(variant, seed, cap):
    return run_game(WORKER_MODELS, variant, seed, cap)


def run_game(models, variant, seed, cap):
    env = FastCandidateTetrisEnv(max_pieces=cap)
    lookahead = LookaheadController(models["ppo"], CONFIGS[variant]) if variant in CONFIGS else None
    hybrid = HybridController(models["ppo"], models["bc"], HybridConfig("B", 12, 8)) if variant == "B12" else None
    started = time.monotonic()
    timings, nodes, changes, searched, clear = [], [], 0, 0, [0]*5
    milestones, examples = [], []
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            public = env.core.get_public_observation()
            began = time.monotonic()
            if lookahead:
                action, decision = lookahead.decide(observation, mask, public, env._placement_cache)
                changes += not decision["top1_retained"]
                searched += decision["searched"]
                nodes.append(decision["nodes"])
                if not decision["top1_retained"] and len(examples) < 100:
                    examples.append({"step": env.pieces, **decision})
            elif hybrid:
                action, decision = hybrid.decide(observation, mask, public, env._placement_cache)
            else:
                model = models["bc"] if variant == "bc" else models["ppo"]
                action = int(model.predict(observation, deterministic=True, action_masks=mask)[0])
            timings.append((time.monotonic()-began)*1000)
            if not mask[action]: raise RuntimeError("Selected illegal action")
            observation, _, over, capped, info = env.step(action)
            clear[info["cleared_lines"]] += 1
            if info["pieces"] % 1000 == 0:
                milestones.append({"pieces": info["pieces"], "score": info["score"]})
            if over or capped: break
        return {"seed": seed, "variant": variant, "cap": cap, "version": VERSION,
                "model_sha256": HASHES, "pieces": info["pieces"], "score": info["score"],
                "game_over": bool(over), "capped": bool(capped), "clears": clear,
                "milestones": milestones, "changes": changes, "searched": searched,
                "mean_nodes": float(np.mean(nodes)) if nodes else 0,
                "mean_ms": float(np.mean(timings)), "p95_ms": float(np.percentile(timings, 95)),
                "p99_ms": float(np.percentile(timings, 99)), "max_ms": max(timings),
                "inference_seconds": sum(timings)/1000, "wall_seconds": time.monotonic()-started,
                "examples": examples}
    finally:
        env.close()


def summary(rows):
    pieces = sorted(r["pieces"] for r in rows)
    return {"games": len(rows), "mean_pieces": statistics.mean(pieces),
            "median_pieces": statistics.median(pieces), "p10_pieces": float(np.percentile(pieces, 10)),
            "minimum_pieces": min(pieces), "capped": sum(r["capped"] for r in rows),
            "game_over": sum(r["game_over"] for r in rows),
            "mean_score": statistics.mean(r["score"] for r in rows),
            "score_per_1000": 1000*sum(r["score"] for r in rows)/sum(pieces),
            "four_fraction": sum(r["clears"][4] for r in rows)/max(1, sum(sum(r["clears"][1:]) for r in rows)),
            "top1_retention": 1-sum(r["changes"] for r in rows)/sum(pieces),
            "search_trigger": sum(r["searched"] for r in rows)/sum(pieces),
            "mean_ms": sum(r["mean_ms"]*r["pieces"] for r in rows)/sum(pieces),
            "p95_game_ms": float(np.percentile([r["p95_ms"] for r in rows], 95)),
            "mean_nodes": sum(r["mean_nodes"]*r["pieces"] for r in rows)/sum(pieces)}


def phase(name, variants, seeds, cap, models):
    result = {}
    for variant in variants:
        folder = RUN / name / variant
        folder.mkdir(parents=True, exist_ok=True)
        rows = {}
        pending = []
        for seed in seeds:
            path = folder / f"seed_{seed}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if any((row.get("seed") != seed, row.get("variant") != variant,
                        row.get("cap") != cap, row.get("model_sha256") != HASHES,
                        row.get("version") != VERSION, row["pieces"] > cap,
                        row["game_over"] == row["capped"])):
                    raise RuntimeError(f"Completed game invalid: {path}")
            else:
                pending.append(seed)
                continue
            rows[seed] = row
        if pending and (name.startswith("development") or name.startswith("confirmation")):
            with ProcessPoolExecutor(max_workers=min(4, len(pending)),
                                     mp_context=get_context("spawn"), initializer=worker_init) as pool:
                futures = {pool.submit(worker_game, variant, seed, cap): seed for seed in pending}
                for future in as_completed(futures):
                    seed = futures[future]
                    row = future.result()
                    atomic_json(folder / f"seed_{seed}.json", row)
                    rows[seed] = row
                    atomic_json(RUN / "status.json", {"stage": name, "variant": variant,
                                "seed": seed, "completed": len(rows), "total": len(seeds),
                                "updated_at": now(), "pid": os.getpid()})
                    print(f"{name} {variant} {seed}: {row['pieces']} pieces {row['score']} score", flush=True)
        else:
            for seed in pending:
                row = run_game(models, variant, seed, cap)
                atomic_json(folder / f"seed_{seed}.json", row)
                rows[seed] = row
                atomic_json(RUN / "status.json", {"stage": name, "variant": variant,
                            "seed": seed, "completed": len(rows), "total": len(seeds),
                            "updated_at": now(), "pid": os.getpid()})
                print(f"{name} {variant} {seed}: {row['pieces']} pieces {row['score']} score", flush=True)
        result[variant] = summary([rows[s] for s in seeds])
        atomic_json(folder / "summary.json", result[variant])
    return result


def common_score_ratio(phase_name, variant):
    ratios = []
    for seed in DEV:
        left = json.loads((RUN / phase_name / variant / f"seed_{seed}.json").read_text())
        right = json.loads((RUN / phase_name / "ppo" / f"seed_{seed}.json").read_text())
        point = min(left["pieces"], right["pieces"]) // 1000 - 1
        if point >= 0:
            ratios.append(left["milestones"][point]["score"] /
                          max(1, right["milestones"][point]["score"]))
    return statistics.mean(ratios) if ratios else 0.0


def decision(results, phase_name):
    base = results["ppo"]
    winners = []
    for variant in ("S1", "S2"):
        row = results[variant]
        if (row["four_fraction"] >= base["four_fraction"]-0.03 and
            row["p10_pieces"] >= 0.9*base["p10_pieces"] and
            row["mean_pieces"] >= 1.03*base["mean_pieces"] and
            row["mean_score"] >= base["mean_score"] and
            common_score_ratio(phase_name, variant) >= 0.95 and
            row["top1_retention"] >= 0.8):
            winners.append(variant)
    return winners


def run():
    try:
        spec = protocol()
        models = load_models()
        smoke = phase("smoke_1k", ["ppo", "S1", "S2"], SMOKE, 1000, models)
        if smoke["S2"]["p95_game_ms"] > spec["smoke_p95_limit_ms"]:
            atomic_json(RUN / "status.json", {"stage": "completed_no_go_performance", "smoke": smoke, "updated_at": now()})
            return
        development = phase("development_20k", ["ppo", "B12", "S1", "S2"], DEV, 20000, models)
        judged_phase = "development_20k"
        if development["ppo"]["capped"] >= 12:
            development = phase("development_50k", ["ppo", "B12", "S1", "S2"], DEV, 50000, models)
            judged_phase = "development_50k"
        winners = decision(development, judged_phase)
        atomic_json(RUN / "decision.json", {"winners": winners, "development": development,
                  "judged_phase": judged_phase, "updated_at": now()})
        if not winners:
            atomic_json(RUN / "status.json", {"stage": "completed_no_go", "updated_at": now()})
            return
        chosen = max(winners, key=lambda v: development[v]["mean_score"])
        phase("confirmation_100k", ["ppo", "bc", "B12", chosen], CONFIRM, 100000, models)
        atomic_json(RUN / "status.json", {"stage": "completed", "chosen": chosen, "updated_at": now()})
    except BaseException as exc:
        atomic_json(RUN / "status.json", {"stage": "failed", "error": repr(exc),
                    "traceback": traceback.format_exc(), "updated_at": now()})
        raise


if __name__ == "__main__":
    run()
