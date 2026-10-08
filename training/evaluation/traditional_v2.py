"""Resumable independent V2 CPU benchmark over the unchanged TetrisCore."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback
import fcntl

from training.evaluation.traditional import ROOT, aggregate, atomic, now
from training.tetris_core import TetrisCore
from training.traditional.v2 import V2, V2Config, VERSION


def code_digest():
    paths = sorted((ROOT / "training/tetris_core").glob("*.py")) + [
        ROOT / "training/traditional/v2.py", ROOT / "training/evaluation/traditional.py",
        Path(__file__).resolve()]
    sha = hashlib.sha256()
    for path in paths:
        sha.update(str(path.relative_to(ROOT)).encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def play(seed, max_pieces, policy_config, progress=None):
    core, policy = TetrisCore(seed), V2(policy_config)
    actions, times = [], []
    timing_totals = dict(root_legal=0., future_legal=0., simulation_features=0., search_overhead=0.)
    started = time.perf_counter()
    while core.phase != "over" and len(actions) < max_pieces:
        observation = core.get_public_observation()
        beginning = time.perf_counter_ns()
        placements = core.get_legal_placements()
        root_ms = (time.perf_counter_ns()-beginning)/1e6
        action = policy.choose(observation, placements)
        if action not in {item["actionId"] for item in placements}:
            raise AssertionError("V2 selected an illegal placement")
        for name in timing_totals:
            timing_totals[name] += root_ms if name == "root_legal" else policy.last_timing_ms[name]
        times.append((time.perf_counter_ns()-beginning)/1e6)
        core.step(action)
        actions.append(action)
        if progress and len(actions) % 100 == 0:
            progress(len(actions))
    return dict(seed=seed, pieces_survived=len(actions), lines=core.lines, score=core.score,
                game_over=core.phase == "over", truncated=core.phase != "over" and len(actions) == max_pieces,
                actions=actions, decision_ms=times, timing_totals_ms=timing_totals,
                hold_actions=sum(action >= 920 for action in actions),
                wall_seconds=time.perf_counter()-started, completed_at=now())


def report(config, rows, metrics):
    params = config["policy"]
    lines = ["# Traditional V2 independent benchmark", "",
             f"Version: {VERSION}; mode: {params['mode']}; beam width: {params['beam_width']}; lookahead: {params['lookahead']}.",
             f"Seeds: {len(rows)}; max pieces: {config['max_pieces']}; Git base: `{config['git_commit']}`; code SHA-256: `{config['code_sha256']}`.",
             "Weights: `" + json.dumps(params["weights"], sort_keys=True) + "`.",
             "Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.",
             "One action means one Core placement; reaching the piece cap is truncation, not death.", "",
             "| Metric | Value |", "|---|---:|"]
    lines += [f"| {key} | {value:.6f} |" if isinstance(value,float) else f"| {key} | {value} |" for key,value in metrics.items()]
    lines += ["", "| Seed | Pieces | Lines | Score | End | Hold actions |", "|---:|---:|---:|---:|---|---:|"]
    lines += [f"| {row['seed']} | {row['pieces_survived']} | {row['lines']} | {row['score']} | {'Game Over' if row['game_over'] else 'cap'} | {row['hold_actions']} |" for row in rows]
    return "\n".join(lines)+"\n"


def run(output, seeds, max_pieces, policy_config, resume=False):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    if max_pieces < 1:
        raise ValueError("max_pieces must be positive")
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output/"config.json").exists():
        raise ValueError("Resume requires config.json")
    with (output/".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = dict(seeds=list(seeds), max_pieces=max_pieces, strategy=VERSION,
                        policy=asdict(policy_config), code_sha256=code_digest())
        if resume:
            config = json.loads((output/"config.json").read_text())
            if any(config.get(key) != value for key,value in expected.items()):
                raise ValueError("Resume configuration/code mismatch")
        else:
            config = dict(expected, created_at=now(), python=platform.python_version(),
                          platform=platform.platform(), git_commit=subprocess.check_output(
                              ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                          git_status=subprocess.check_output(
                              ["git","status","--short"],cwd=ROOT,text=True),
                          timing="root legal BFS + policy simulation/features/future BFS/search; excludes Core.step and record I/O")
            atomic(output/"config.json",config)
        rows = []
        try:
            for seed in seeds:
                destination = output/f"seed_{seed}.json"
                if destination.exists():
                    row = json.loads(destination.read_text())
                    if (row["seed"] != seed or len(row["actions"]) != row["pieces_survived"]
                            or len(row["decision_ms"]) != row["pieces_survived"]
                            or row["game_over"] == row["truncated"]
                            or not 0 < row["pieces_survived"] <= max_pieces):
                        raise ValueError("Invalid saved seed record")
                else:
                    def progress(pieces):
                        atomic(output/"status.json",dict(state="running",seed=seed,pieces=pieces,
                               completed_seeds=len(rows),total_seeds=len(seeds),updated_at=now()))
                    progress(0)
                    row = play(seed,max_pieces,policy_config,progress)
                    atomic(destination,row)
                rows.append(row)
            metrics = aggregate(rows)
            decisions = sum(row["pieces_survived"] for row in rows)
            for key in ("root_legal","future_legal","simulation_features","search_overhead"):
                metrics[f"mean_{key}_ms"] = sum(row["timing_totals_ms"][key] for row in rows)/decisions
            metrics["hold_action_rate"] = sum(row["hold_actions"] for row in rows)/decisions
            atomic(output/"aggregate.json",metrics)
            if not (output/"report.md").exists():
                (output/"report.md").write_text(report(config,rows,metrics))
            atomic(output/"status.json",dict(state="complete",completed_seeds=len(rows),
                   total_seeds=len(seeds),updated_at=now()))
            return metrics
        except BaseException as error:
            atomic(output/f"error_{time.time_ns()}.json",dict(type=type(error).__name__,
                   message=str(error),traceback=traceback.format_exc(),timestamp=now()))
            atomic(output/"status.json",dict(state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                   completed_seeds=len(rows),total_seeds=len(seeds),updated_at=now()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",nargs="+",type=int)
    parser.add_argument("--max-pieces",type=int,default=5000)
    parser.add_argument("--mode",choices=("heuristic","hold","beam"),default="beam")
    parser.add_argument("--beam-width",type=int,default=8)
    parser.add_argument("--lookahead",type=float,default=.58)
    parser.add_argument("--weights-file",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--status",action="store_true")
    args = parser.parse_args()
    if args.status:
        print((args.output/"status.json").read_text())
        return
    seeds = args.seeds or json.loads((ROOT/"training/evaluation/seeds.json").read_text())["validation"][:16]
    weights = json.loads(args.weights_file.read_text()) if args.weights_file else None
    config = V2Config(mode=args.mode,beam_width=args.beam_width,lookahead=args.lookahead,
                      **({"weights":weights} if weights is not None else {}))
    print(json.dumps(run(args.output,seeds,args.max_pieces,config,args.resume),indent=2))


if __name__ == "__main__":
    main()
