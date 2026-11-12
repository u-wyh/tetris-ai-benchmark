"""Compare frozen policies on the same public observations and root BFS results."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter_ns

from training.evaluation.traditional import ROOT, atomic, now
from training.tetris_core import TetrisCore
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v1 import V1Adapter
from training.traditional.v2 import V2, V2Config

SAMPLE_STEPS = (0, 5, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000, 3000, 4000, 6000, 8000)


def public_states(reference, steps=SAMPLE_STEPS):
    row = json.loads(Path(reference).read_text())
    core = TetrisCore(row["seed"])
    selected = []
    wanted = set(steps)
    for index, action in enumerate(row["actions"]):
        if index > max(wanted):
            break
        if index in wanted:
            selected.append((index, core.get_public_observation()))
        core.step(action)
    return row["seed"], selected


def profile(reference, repeats=3, steps=SAMPLE_STEPS):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    seed, states = public_states(reference, steps)
    samples = []
    for index, observation in states:
        for repetition in range(repeats):
            began = perf_counter_ns()
            root = get_legal_placements(observation)
            root_ms = (perf_counter_ns() - began) / 1e6
            policies = {"v1": V1Adapter(seed), "hold": V2(V2Config(mode="hold")),
                        "beam": V2(V2Config(mode="beam"))}
            elapsed = {}
            actions = {}
            breakdown = {}
            for mode in ("v1", "hold", "beam"):
                began = perf_counter_ns()
                actions[mode] = policies[mode].choose(observation, root)
                elapsed[mode] = (perf_counter_ns() - began) / 1e6
                if mode != "v1":
                    breakdown[mode] = policies[mode].last_timing_ms
            samples.append(dict(step=index, repetition=repetition,
                root_legal_ms=root_ms, policy_ms=elapsed,
                total_ms={mode: root_ms + elapsed[mode] for mode in elapsed},
                action_ids=actions, v2_breakdown=breakdown))
    summary = dict(states=len(states), repetitions=repeats, samples=len(samples))
    for mode in ("v1", "hold", "beam"):
        summary[f"{mode}_mean_total_ms"] = statistics.mean(s["total_ms"][mode] for s in samples)
        summary[f"{mode}_median_total_ms"] = statistics.median(s["total_ms"][mode] for s in samples)
        summary[f"{mode}_mean_policy_ms"] = statistics.mean(s["policy_ms"][mode] for s in samples)
    summary["mean_common_root_legal_ms"] = statistics.mean(s["root_legal_ms"] for s in samples)
    return dict(created_at=now(), reference=str(reference), seed=seed,
                sample_steps=[index for index, _ in states], repeats=repeats,
                python=platform.python_version(), platform=platform.platform(),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                timing="same public states, one root BFS per sample, no Action Mask, Core.step or I/O",
                summary=summary, samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = profile(args.reference, args.repeats)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic(args.output / "profile.json", result)
    lines = ["# Matched public-state decision profile", "",
             f"{result['summary']['states']} observations × {args.repeats} repetitions. One root BFS shared by all policies; no Action Mask. The policies see only public observations.",
             "Timing excludes Core.step and record I/O. Batch-run timing includes V1's Action Mask generation, so use this profile for policy-cost comparison under matched conditions.", "",
             "| Mode | Mean total ms | Median total ms | Mean policy ms |", "|---|---:|---:|---:|"]
    for mode in ("v1", "hold", "beam"):
        s = result["summary"]
        lines.append(f"| {mode} | {s[f'{mode}_mean_total_ms']:.3f} | {s[f'{mode}_median_total_ms']:.3f} | {s[f'{mode}_mean_policy_ms']:.3f} |")
    lines.append(f"\nCommon root BFS mean: {result['summary']['mean_common_root_legal_ms']:.3f} ms.\n")
    (args.output / "report.md").write_text("\n".join(lines))
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
