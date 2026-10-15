"""Atomic paired long-game evaluation for the PPO30M overnight experiment."""
import gzip
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from sb3_contrib import MaskablePPO

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import BC, HASHES, PPO
from training.hybrid_policy import HybridConfig, HybridController
from training.imitation_dataset import digest
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now

RUN = ROOT / "runs/ppo30m_overnight_finetune"
VERSION = "overnight_paired_50k_v1"
DEV = list(range(880100, 880116))
CONFIRM = list(range(880200, 880232))
MODEL_CACHE = {}


def _model(path):
    import torch
    torch.set_num_threads(1)
    key = str(path)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = MaskablePPO.load(key, device="cpu")
    return MODEL_CACHE[key]


def _game(variant, seed, cap, checkpoint, expected_sha, folder):
    path = folder / variant / f"seed_{seed}.json.gz"
    identity = {"version": VERSION, "variant": variant, "seed": seed, "cap": cap,
                "model_sha256": expected_sha, "checkpoint": str(checkpoint)}
    if path.exists():
        row = json.loads(gzip.decompress(path.read_bytes()))
        if any(row.get(k) != v for k,v in identity.items()):
            raise RuntimeError(f"Evaluation resume identity mismatch: {path}")
        return {k:v for k,v in row.items() if k not in ("scores", "fours")}
    env = FastCandidateTetrisEnv(max_pieces=cap)
    began = time.monotonic()
    model = _model(checkpoint)
    controller = (HybridController(model, _model(BC), HybridConfig("B",12,8))
                  if variant == "hybrid_b12" else None)
    try:
        obs, _ = env.reset(seed=seed)
        scores, fours, clears = [], [], [0]*5
        height_risk_steps = 0
        done = capped = False
        while not (done or capped):
            mask = env.action_masks()
            if controller:
                action, _ = controller.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            else:
                action = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
            if not mask[action]: raise RuntimeError("Policy chose an illegal action")
            obs, _, done, capped, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            scores.append(info["score"])
            fours.append(clears[4])
            height_risk_steps += info["height_after"] >= 16
        row = {**identity, "pieces": info["pieces"], "game_over": bool(done),
               "right_censored": bool(capped), "score": info["score"],
               "score_per_1000": 1000*info["score"]/info["pieces"],
               "clears": clears, "four_event_share": clears[4]/max(1,sum(clears[1:])),
               "height_risk_fraction": height_risk_steps/info["pieces"],
               "scores": scores, "fours": fours,
               "elapsed_seconds": time.monotonic()-began, "completed_at": now()}
        atomic_bytes(path, gzip.compress(json.dumps(row,separators=(",", ":")).encode(), compresslevel=3))
        return {k:v for k,v in row.items() if k not in ("scores", "fours")}
    finally:
        env.close()


def evaluate(variant_paths, seeds=DEV, cap=50000, label="development", workers=4):
    folder = RUN / "evaluations" / label
    jobs = []
    for variant, checkpoint in variant_paths.items():
        model_hash = digest(checkpoint)
        for seed in seeds:
            jobs.append((variant,seed,cap,checkpoint,model_hash,folder))
    if workers == 1:
        results = [_game(*job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(_game,*job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda x:(x["variant"],x["seed"]))
    if len(results) != len(jobs): raise RuntimeError("Incomplete paired evaluation")
    summary = {"version": VERSION, "label": label, "cap": cap,
               "seeds": list(seeds), "variants": {}}
    for variant in variant_paths:
        rows = [r for r in results if r["variant"] == variant]
        pieces = sorted(r["pieces"] for r in rows)
        summary["variants"][variant] = {"mean_pieces": statistics.mean(pieces),
            "p10_pieces": pieces[max(0, int(.1*(len(pieces)-1)))],
            "median_pieces": statistics.median(pieces),
            "cap_rate": sum(r["right_censored"] for r in rows)/len(rows),
            "game_overs": sum(r["game_over"] for r in rows),
            "mean_score": statistics.mean(r["score"] for r in rows),
            "mean_score_per_1000": statistics.mean(r["score_per_1000"] for r in rows),
            "four_event_share": sum(r["clears"][4] for r in rows)/max(1,sum(sum(r["clears"][1:]) for r in rows)),
            "four_clears": sum(r["clears"][4] for r in rows),
            "mean_height_risk_fraction": statistics.mean(r["height_risk_fraction"] for r in rows),
            "model_sha256": rows[0]["model_sha256"]}
    for variant in variant_paths:
        if variant == "ppo": continue
        pairs = []
        for seed in seeds:
            left = json.loads(gzip.decompress((folder/"ppo"/f"seed_{seed}.json.gz").read_bytes()))
            right = json.loads(gzip.decompress((folder/variant/f"seed_{seed}.json.gz").read_bytes()))
            common = min(left["pieces"], right["pieces"])
            pairs.append({"seed": seed, "common_pieces": common,
                          "score_ratio": right["scores"][common-1]/max(1,left["scores"][common-1]),
                          "four_ratio": right["fours"][common-1]/max(1,left["fours"][common-1]),
                          "ppo_fours": left["fours"][common-1],
                          "candidate_fours": right["fours"][common-1],
                          "survival_delta": right["pieces"]-left["pieces"]})
        summary["variants"][variant]["paired"] = pairs
        summary["variants"][variant]["mean_common_score_ratio"] = statistics.mean(p["score_ratio"] for p in pairs)
    atomic_json(folder/"summary.json",summary)
    return summary
