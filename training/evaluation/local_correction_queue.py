"""Resumable background experiment; final seeds are evaluated only after freezing."""
import json
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch
from sb3_contrib import MaskablePPO

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, HybridConfig, HybridController, load_models, run_game
from training.evaluation.hybrid_queue import digest as file_digest
from training.local_correction_data import RUN, TRAIN, VALIDATION, DEV, CONFIRM, audit, capture, filter_seed
from training.train_local_correction import dataset, train
from training.train_ppo import ROOT, atomic_json, now

MODELS = None


def _init_models():
    global MODELS
    MODELS = load_models()


def _collect(seed):
    return capture(seed, MODELS)


def _filter(seed):
    return filter_seed(seed, MODELS)


def stage(name, jobs, task, workers=8):
    status = RUN / "status.json"
    atomic_json(status, {"state": "running", "stage": name, "completed": 0,
                         "total": len(jobs), "updated_at": now()})
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                             mp_context=get_context("spawn"), initializer=_init_models) as pool:
        futures = [pool.submit(task, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            atomic_json(status, {"state": "running", "stage": name, "completed": index,
                                 "total": len(jobs), "last_seed": result["seed"],
                                 "updated_at": now()})
            print(name, result["seed"], index, "/", len(jobs), flush=True)


def _eval_init(candidate):
    global MODELS
    MODELS = load_models()
    if candidate:
        MODELS["local"] = MaskablePPO.load(str(candidate), device="cpu")


def _play_model(model, seed, cap, identity):
    env = FastCandidateTetrisEnv(max_pieces=cap)
    started = time.monotonic()
    clears = [0] * 5
    milestones = []
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            action = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
            if not mask[action]:
                raise RuntimeError("Local policy selected illegal Action ID")
            obs, _, ended, capped, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            if info["pieces"] % 1000 == 0:
                milestones.append({"pieces": info["pieces"], "score": info["score"]})
            if ended or capped:
                break
        return {"seed": seed, "variant": identity, "max_pieces": cap,
                "pieces_survived": info["pieces"], "score": info["score"],
                "game_over": bool(ended), "survived_cap": bool(capped),
                "clears_0_to_4": clears, "score_milestones": milestones,
                "wall_seconds": time.monotonic() - started}
    finally:
        env.close()


def _eval_worker(variant, seed, cap, checkpoint, output):
    if variant in ("ppo", "bc", "B12"):
        row = run_game(MODELS, variant, seed, cap)
    else:
        row = _play_model(MODELS["local"], seed, cap, variant)
    atomic_json(output, row)
    return {"seed": seed, "pieces": row["pieces_survived"]}


def evaluate(phase, variant, seeds, cap, checkpoint=None):
    folder = RUN / "evaluations" / phase / variant
    folder.mkdir(parents=True, exist_ok=True)
    identity = file_digest(checkpoint) if checkpoint else HASHES
    config = {"phase": phase, "variant": variant, "seeds": seeds, "cap": cap,
              "checkpoint_sha256": identity, "deterministic": True,
              "official_mask": True}
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Evaluation identity changed on resume")
    if not config_path.exists():
        atomic_json(config_path, config)
    pending = []
    for seed in seeds:
        output = folder / f"seed_{seed}.json"
        if output.exists():
            row = json.loads(output.read_text())
            if (row["seed"] != seed or row["max_pieces"] != cap or
                    row["pieces_survived"] > cap or
                    row["game_over"] == row["survived_cap"]):
                raise RuntimeError(f"Invalid completed game {output}")
        else:
            pending.append((variant, seed, cap, checkpoint, output))
    if pending:
        with ProcessPoolExecutor(max_workers=min(8, len(pending)),
                                 mp_context=get_context("spawn"),
                                 initializer=_eval_init, initargs=(checkpoint,)) as pool:
            futures = [pool.submit(_eval_worker, *job) for job in pending]
            for future in as_completed(futures):
                result = future.result()
                atomic_json(RUN / "status.json", {"state": "running", "stage": phase,
                            "variant": variant, "last_seed": result["seed"],
                            "completed": len(list(folder.glob("seed_*.json"))),
                            "total": len(seeds), "updated_at": now()})
                print(phase, variant, result, flush=True)
    rows = [json.loads((folder / f"seed_{seed}.json").read_text()) for seed in seeds]
    pieces = [r["pieces_survived"] for r in rows]
    scores = [r["score"] for r in rows]
    fours = sum(r["clears_0_to_4"][4] for r in rows)
    events = sum(sum(r["clears_0_to_4"][1:]) for r in rows)
    ordered = sorted(pieces)
    rank = .1 * (len(rows) - 1)
    lo = int(rank)
    summary = {"games": len(rows), "mean_pieces": statistics.mean(pieces),
               "p10_pieces": ordered[lo] + (ordered[min(lo + 1, len(rows) - 1)] - ordered[lo]) * (rank - lo),
               "median_pieces": statistics.median(pieces), "min_pieces": min(pieces),
               "capped": sum(r["survived_cap"] for r in rows),
               "mean_score": statistics.mean(scores),
               "score_per_1000": sum(scores) / sum(pieces) * 1000,
               "clear_counts_1_to_4": [sum(r["clears_0_to_4"][i] for r in rows)
                                       for i in range(1, 5)],
               "four_fraction": fours / events if events else 0.0,
               "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in rows)}
    atomic_json(folder / "summary.json", summary)
    return summary


def _healthy(candidate, ppo):
    return (candidate["mean_pieces"] >= .9 * ppo["mean_pieces"] and
            candidate["p10_pieces"] >= .8 * ppo["p10_pieces"] and
            candidate["mean_score"] >= .9 * ppo["mean_score"] and
            candidate["four_fraction"] >= ppo["four_fraction"] - .05)


def _promising(candidate, ppo):
    return (candidate["mean_pieces"] > ppo["mean_pieces"] and
            candidate["p10_pieces"] > ppo["p10_pieces"] and
            candidate["mean_score"] >= .95 * ppo["mean_score"] and
            candidate["four_fraction"] >= ppo["four_fraction"] - .03)


def _report(state, choice=None):
    from training.evaluation.local_correction_report import generate
    generate(state, choice)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        atomic_json(RUN / "status.json", {"state": "running", "stage": "freeze_and_audit",
                                          "updated_at": now()})
        audit()
        stage("collect_ppo_trajectories", TRAIN + VALIDATION, _collect)
        atomic_json(RUN / "status.json", {"state": "running", "stage": "risk_state_analysis",
                                          "updated_at": now()})
        stage("expert_counterfactual_filtering", TRAIN + VALIDATION, _filter, workers=4)
        atomic_json(RUN / "status.json", {"state": "running", "stage": "dataset_validation",
                                          "updated_at": now()})
        danger, normal = dataset(TRAIN)
        val_danger, val_normal = dataset(VALIDATION)
        atomic_json(RUN / "dataset_summary.json", {"train_A": len(danger),
                    "train_C": len(normal), "validation_A": len(val_danger),
                    "validation_C": len(val_normal), "updated_at": now()})
        if len(danger) < 8 or len(val_danger) < 2:
            state = "insufficient_verified_corrections"
            _report(state)
            atomic_json(RUN / "status.json", {"state": state, "updated_at": now()})
            return
        atomic_json(RUN / "status.json", {"state": "running", "stage": "small_scale_training",
                                          "updated_at": now()})
        train("V1")
        ppo20 = evaluate("development_20k", "ppo", DEV, 20000)
        evaluate("development_20k", "bc", DEV, 20000)
        evaluate("development_20k", "B12", DEV, 20000)
        v1_options = []
        for epoch in (1, 2, 3):
            ckpt = RUN / "training/V1" / f"epoch_{epoch:02d}.zip"
            variant = f"V1_e{epoch}"
            result = evaluate("development_20k", variant, DEV, 20000, ckpt)
            if _healthy(result, ppo20):
                v1_options.append((variant, ckpt, result))
        if not v1_options:
            state = "v1_failed_short_validation"
            _report(state)
            atomic_json(RUN / "status.json", {"state": state, "updated_at": now()})
            return
        best_v1 = max(v1_options, key=lambda x: (x[2]["mean_pieces"], x[2]["mean_score"]))
        ppo100 = evaluate("development_100k", "ppo", DEV, 100000)
        evaluate("development_100k", "bc", DEV, 100000)
        evaluate("development_100k", "B12", DEV, 100000)
        v1_long = evaluate("development_100k", best_v1[0], DEV, 100000, best_v1[1])
        if not _healthy(v1_long, ppo100):
            state = "v1_failed_long_validation"
            _report(state)
            atomic_json(RUN / "status.json", {"state": state, "updated_at": now()})
            return
        # Only investigate risk weighting if the conservative version passed both gates.
        options = [(best_v1[0], best_v1[1], v1_long)]
        if _healthy(v1_long, ppo100):
            train("V2")
            for epoch in (1, 2, 3):
                ckpt = RUN / "training/V2" / f"epoch_{epoch:02d}.zip"
                variant = f"V2_e{epoch}"
                short = evaluate("development_20k", variant, DEV, 20000, ckpt)
                if _healthy(short, ppo20):
                    long = evaluate("development_100k", variant, DEV, 100000, ckpt)
                    if _healthy(long, ppo100):
                        options.append((variant, ckpt, long))
        best = max(options, key=lambda x: (x[2]["p10_pieces"], x[2]["mean_score"]))
        if not _promising(best[2], ppo100):
            state = "no_promising_candidate"
            _report(state)
            atomic_json(RUN / "status.json", {"state": state, "updated_at": now()})
            return
        selection = {"variant": best[0], "checkpoint": str(best[1]),
                     "checkpoint_sha256": file_digest(best[1]),
                     "development_100k": best[2], "frozen_at": now(),
                     "confirmation_seeds": CONFIRM}
        atomic_json(RUN / "status.json", {"state": "running", "stage": "freeze_best_candidate",
                                          "updated_at": now()})
        frozen = RUN / "selection.json"
        if frozen.exists() and json.loads(frozen.read_text()) != selection:
            # Resume uses the previously frozen identity and never reselects using confirmation.
            old = json.loads(frozen.read_text())
            if (old["checkpoint_sha256"] != selection["checkpoint_sha256"] or
                    old["variant"] != selection["variant"]):
                raise RuntimeError("Frozen model selection changed")
        else:
            atomic_json(frozen, selection)
        evaluate("confirmation_100k", "ppo", CONFIRM, 100000)
        evaluate("confirmation_100k", "bc", CONFIRM, 100000)
        evaluate("confirmation_100k", "B12", CONFIRM, 100000)
        evaluate("confirmation_100k", best[0], CONFIRM, 100000, best[1])
        atomic_json(RUN / "status.json", {"state": "running", "stage": "final_report",
                                          "updated_at": now()})
        _report("completed", selection)
        atomic_json(RUN / "status.json", {"state": "completed", "updated_at": now()})
    except Exception:
        atomic_json(RUN / "status.json", {"state": "failed", "error": traceback.format_exc(),
                                          "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
