"""Bounded, resumable paired development evaluation of one frozen reward candidate."""

import hashlib
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from training.evaluation.failure_curriculum_eval import _common_score, _summary, init_worker, task
from training.evaluation.hybrid_queue import PPO
from training.train_ppo import ROOT, atomic_json, now
from training.train_vector_transaction import verify_task

RUN = ROOT / "runs/ppo30m_survival_balanced"
TRAIN = RUN / "training"
DEV = tuple(range(1521000, 1521016))
MODEL = TRAIN / "committed/task_000016/model.zip"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **details):
    atomic_json(RUN / "evaluation_status.json", {"stage": stage, "at": now(),
                                                 "pid": os.getpid(), **details})


def evaluate(cap):
    label = f"dev_{cap}"
    folder = RUN / "evaluation" / label
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"ppo": PPO, "candidate": MODEL}
    hashes = {name: digest(path) for name, path in paths.items()}
    protocol = {"seeds": list(DEV), "cap": cap, "model_sha256": hashes,
                "fast": True, "fast_legal": True, "workers": 4}
    protocol_path = folder / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("Development evaluation protocol changed")
    else:
        atomic_json(protocol_path, protocol)
    rows = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"),
                             initializer=init_worker, initargs=(paths,)) as pool:
        futures = {}
        for variant in paths:
            (folder / variant).mkdir(exist_ok=True)
            for seed in DEV:
                path = folder / variant / f"seed_{seed}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    if (row["seed"] != seed or row["cap"] != cap or row["variant"] != variant
                            or row["model_sha256"] != hashes):
                        raise RuntimeError(f"Saved game identity changed: {path}")
                    rows[variant, seed] = row
                else:
                    futures[pool.submit(task, (variant, seed, cap, hashes, True, True, None))] = (
                        variant, seed, path)
        for future in as_completed(futures):
            variant, seed, path = futures[future]
            row = future.result()
            atomic_json(path, row)
            rows[variant, seed] = row
            status("evaluating", cap=cap, complete=len(rows), total=2 * len(DEV))
    summary = {name: _summary([rows[name, seed] for seed in DEV]) for name in paths}
    score_pairs = [_common_score(rows["ppo", seed], rows["candidate", seed]) for seed in DEV]
    score_pairs = [pair for pair in score_pairs if pair and pair[0] > 0]
    summary["candidate"]["common_score_ratio_to_ppo"] = (
        sum(pair[1] for pair in score_pairs) / sum(pair[0] for pair in score_pairs)
        if score_pairs else None)
    atomic_json(folder / "summary.json", summary)
    return summary


def short_gate(summary):
    base, candidate = summary["ppo"], summary["candidate"]
    return (candidate["mean_pieces"] >= .95 * base["mean_pieces"]
            and candidate["p10_pieces"] >= .90 * base["p10_pieces"]
            and candidate["early_under_10000"] <= base["early_under_10000"]
            and candidate["mean_score"] >= .80 * base["mean_score"])


def long_gate(summary):
    base, candidate = summary["ppo"], summary["candidate"]
    return (candidate["mean_pieces"] > base["mean_pieces"]
            and candidate["p10_pieces"] > base["p10_pieces"]
            and candidate["early_under_10000"] <= base["early_under_10000"]
            and candidate["mean_score"] >= .80 * base["mean_score"]
            and (candidate["common_score_ratio_to_ppo"] or 0) >= .80
            and candidate["four_fraction"] <= base["four_fraction"] - .05)


def main():
    config = json.loads((TRAIN / "config.json").read_text())
    state = json.loads((TRAIN / "transaction_state.json").read_text())
    if state["committed_task"] != 16 or state["committed_steps"] != 65536:
        raise RuntimeError("Training budget did not finish")
    verify_task(MODEL.parent, 16, config)
    if digest(PPO) != "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34":
        raise RuntimeError("Frozen PPO identity changed")
    status("evaluating_20k")
    short = evaluate(20000)
    if not short_gate(short):
        status("stopped_20k_gate", model_sha256=digest(MODEL))
        return
    status("evaluating_100k")
    long = evaluate(100000)
    status("development_complete", passed=long_gate(long), model_sha256=digest(MODEL))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json(RUN / "evaluation_failure.json", {"at": now(), "error": str(error),
                                                      "traceback": traceback.format_exc()})
        status("failed", error=str(error))
        raise
