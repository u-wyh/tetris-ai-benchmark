"""Paired Candidate training-environment benchmark on identical legal actions."""

import json
import resource
import statistics
import time
from pathlib import Path

from training.tetris_core import TetrisCore
from training.train_ppo import ROOT, atomic_json
from training.train_vector_transaction import TransactionCandidateEnv, TransactionFastCandidateEnv
from training.traditional.v2 import V2, V2Config

SEEDS = (100016, 100017, 100018)
STEPS = 300
REPEATS = 3
OUT = ROOT / "runs/autonomous_rl_research/performance.json"


def actions(seed):
    core = TetrisCore(seed)
    policy = V2(V2Config(mode="hold"))
    result = []
    for _ in range(STEPS):
        action = policy.choose(core.get_public_observation(), core.get_legal_placements())
        result.append(action)
        core.step(action)
    return result


def sample(env_class, seed, trace):
    env = env_class(max_pieces=STEPS + 1)
    try:
        env.reset(seed=seed)
        started = time.perf_counter_ns()
        for action in trace:
            if not env.action_masks()[action]:
                raise RuntimeError("Benchmark action became illegal")
            env.step(action)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        return {"seconds": elapsed, "score": env.core.score, "lines": env.core.lines,
                "board": env.core.board}
    finally:
        env.close()


def main():
    traces = {seed: actions(seed) for seed in SEEDS}
    data = {"protocol": {"seeds": SEEDS, "steps_per_seed": STEPS, "repeats": REPEATS,
                         "same_actions": True, "includes": "action_masks and env.step",
                         "excludes": "action generation and training update"}, "samples": []}
    for rep in range(REPEATS):
        for seed in SEEDS:
            order = (TransactionCandidateEnv, TransactionFastCandidateEnv) if rep % 2 == 0 else (
                TransactionFastCandidateEnv, TransactionCandidateEnv)
            pair = {}
            for cls in order:
                key = "cached" if cls is TransactionFastCandidateEnv else "ordinary"
                pair[key] = sample(cls, seed, traces[seed])
            if any(pair["ordinary"][k] != pair["cached"][k] for k in ("score", "lines", "board")):
                raise RuntimeError("Candidate environment benchmark changed formal game result")
            data["samples"].append({"repeat": rep, "seed": seed,
                                    "ordinary_seconds": pair["ordinary"]["seconds"],
                                    "cached_seconds": pair["cached"]["seconds"]})
    before = [r["ordinary_seconds"] for r in data["samples"]]
    after = [r["cached_seconds"] for r in data["samples"]]
    data["summary"] = {"ordinary_mean_steps_per_second": STEPS / statistics.mean(before),
                       "cached_mean_steps_per_second": STEPS / statistics.mean(after),
                       "speedup": statistics.mean(before) / statistics.mean(after),
                       "paired_speedups": [a / b for a, b in zip(before, after)],
                       "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT, data)
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
