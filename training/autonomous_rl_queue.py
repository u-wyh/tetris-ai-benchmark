"""Bounded, resumable CPU pilot and conditional long-horizon PPO research."""

import hashlib
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from training.evaluation.failure_curriculum_eval import _common_score, _summary, init_worker, task
from training.evaluation.hybrid_queue import BC, PPO, verify_models
from training.train_ppo import ROOT, atomic_json, now
from training.train_transaction import read_state
from training.train_vector_transaction import run, verify_task

RUN = ROOT / "runs/autonomous_rl_research"
TRAIN = RUN / "training"
PRESET = RUN / "preset.json"
DEV = tuple(range(1491000, 1491016))
CONFIRM = tuple(range(1492000, 1492032))
MAX_TASKS = 1220
MODEL_NAMES = ("ppo", "B12", "bc", "candidate")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **extra):
    atomic_json(RUN / "status.json", dict(stage=stage, at=now(), pid=os.getpid(), **extra))


def check_budget():
    import shutil
    if shutil.disk_usage(RUN).free < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB disk free")
    size = sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())
    if size > 6 * 1024**3:
        raise RuntimeError("Autonomous run exceeded 6 GiB")


def frozen_preset():
    return {
        "target_total_steps": MAX_TASKS * 4096, "task_steps": 4096,
        "device": "cpu", "n_envs": 4, "n_steps": 1024,
        "base_seed": 1490000, "candidate_policy": True, "max_pieces": 100000,
        "overnight_monitor": True, "step0_baseline": False,
        "transaction_retention": 3, "milestone_interval": 100000000,
        "validation_interval": 100000000,
        "final_test_seeds": 100, "final_test_max_pieces": 50000,
        "ppo_overrides": {"gamma": 0.999, "gae_lambda": 0.98},
        "initial_model": {"path": str(PPO), "sha256": digest(PPO)},
        "kl_reference": {"path": str(PPO), "sha256": digest(PPO), "beta": 0.03},
    }


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_models()
    for seed in (*range(1490000, 1490004), *DEV, *CONFIRM):
        for root in (ROOT / "runs", ROOT / "reports"):
            for p in root.rglob(f"seed_{seed}.*"):
                if not p.is_relative_to(RUN):
                    raise RuntimeError(f"Research seed already used: {p}")
    preset = frozen_preset()
    protocol = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                       text=True).strip(),
                "training_seeds": list(range(1490000, 1490004)), "development_seeds": list(DEV),
                "confirmation_seeds": list(CONFIRM), "preset": preset,
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                  (ROOT / "training/train_vector_transaction.py",
                                   ROOT / "training/autonomous_rl_queue.py")},
                "bc_sha256": digest(BC), "max_training_steps": MAX_TASKS * 4096}
    if (RUN / "protocol.json").exists():
        protocol["git_commit"] = json.loads((RUN / "protocol.json").read_text())["git_commit"]
    for path, value in ((PRESET, preset), (RUN / "protocol.json", protocol)):
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise RuntimeError(f"Frozen protocol changed: {path}")
        else:
            atomic_json(path, value)


def train_to(target):
    if target > MAX_TASKS:
        raise RuntimeError("Training budget exceeded")
    check_budget()
    current = read_state(TRAIN)["committed_task"] if (TRAIN / "transaction_state.json").exists() else 0
    if current < target:
        status("training", committed_tasks=current, target_tasks=target)
        start = time.monotonic()
        run(TRAIN, resume=current > 0, max_tasks=target - current, config_file=str(PRESET))
        elapsed = time.monotonic() - start
        if current == 0:
            atomic_json(RUN / "pilot_timing.json", {"tasks": target, "elapsed_seconds": elapsed,
                                                    "seconds_per_task": elapsed / target})
    state = read_state(TRAIN)
    if state["committed_task"] != target:
        raise RuntimeError("Training stopped outside intended transaction boundary")
    config = json.loads((TRAIN / "config.json").read_text())
    model = TRAIN / f"committed/task_{target:06d}/model.zip"
    verify_task(model.parent, target, config)
    return model


def evaluate(model, seeds, cap, label):
    folder = RUN / "evaluation" / label
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"ppo": PPO, "bc": BC, "candidate": model}
    hashes = {name: digest(path) for name, path in paths.items()}
    rows = {}
    for name in MODEL_NAMES:
        variant = "B12" if name == "B12" else name
        existing = []
        for seed in seeds:
            path = folder / name / f"seed_{seed}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if (row["seed"] != seed or row["cap"] != cap or
                        row["model_sha256"] != hashes or row["variant"] != variant):
                    raise RuntimeError(f"Saved evaluation protocol mismatch: {path}")
                existing.append(row)
        pending = [s for s in seeds if not (folder / name / f"seed_{s}.json").exists()]
        if pending:
            status("evaluation", label=label, model=name, completed=len(existing), total=len(seeds))
            (folder / name).mkdir(parents=True, exist_ok=True)
            with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"),
                                     initializer=init_worker, initargs=(paths,)) as pool:
                futures = {pool.submit(task, (variant, seed, cap, hashes, True, True,
                           folder / name / f"seed_{seed}.checkpoint.pkl")): seed for seed in pending}
                for future in as_completed(futures):
                    seed = futures[future]
                    atomic_json(folder / name / f"seed_{seed}.json", future.result())
                    check_budget()
                    status("evaluation", label=label, model=name,
                           completed=len(existing) + sum(f.done() and not f.exception() for f in futures),
                           total=len(seeds))
        rows[name] = [json.loads((folder / name / f"seed_{s}.json").read_text()) for s in seeds]
    summary = {name: _summary(group) for name, group in rows.items()}
    pair = []
    for a, b in zip(rows["ppo"], rows["candidate"]):
        common = _common_score(a, b)
        if common and common[0] > 0:
            pair.append((common[0], common[1]))
    summary["candidate"]["common_score_ratio_to_ppo"] = (
        sum(b for _, b in pair) / sum(a for a, _ in pair) if pair else None)
    atomic_json(folder / "summary.json", summary)
    return summary


def gate(summary, long=False):
    p, c = summary["ppo"], summary["candidate"]
    return (c["mean_pieces"] >= p["mean_pieces"] * (1.0 if long else .95)
            and c["p10_pieces"] >= p["p10_pieces"] * (1.0 if long else .90)
            and c["early_under_10000"] <= p["early_under_10000"]
            and c["mean_score"] >= p["mean_score"] * .95
            and c["four_fraction"] >= p["four_fraction"] - .03
            and (not long or (c["common_score_ratio_to_ppo"] or 0) >= .95))


def report(reason):
    details = ["# 自主强化学习阶段结果", "", f"状态：{reason}。", "",
               "CPU 训练使用加长折扣和 GAE、冻结 PPO KL 锚定；触顶右截断。"]
    for label in ("smoke", "dev_20k", "dev_100k", "confirm_100k"):
        path = RUN / "evaluation" / label / "summary.json"
        if path.exists():
            details += ["", f"## {label}", "", "```json", path.read_text(), "```"]
    result = ROOT / "reports/experiments/autonomous_rl_results.md"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("\n".join(details) + "\n")
    final = ROOT / "reports/experiments/autonomous_rl_final_summary.md"
    final.write_text("# 自主强化学习阶段总结\n\n" + f"当前停止原因：{reason}。\n\n"
                     "完整协议见 [设计](autonomous_rl_design.md)，逐 seed 和事务数据见 "
                     "`runs/autonomous_rl_research/`。冻结模型与网页保持不变。\n")


def main():
    try:
        freeze()
        if not (RUN / "evaluation/smoke/summary.json").exists():
            model = train_to(1)
            smoke = evaluate(model, DEV[:2], 2000, "smoke")
        else:
            smoke = json.loads((RUN / "evaluation/smoke/summary.json").read_text())
        timing = json.loads((RUN / "pilot_timing.json").read_text())
        if timing["seconds_per_task"] > 180:
            status("stopped_cpu_too_slow", seconds_per_task=timing["seconds_per_task"])
            report("CPU 每事务超过预设 180 秒；转入性能研究")
            return
        if smoke["candidate"]["four_fraction"] < smoke["ppo"]["four_fraction"] - .10:
            status("stopped_smoke_four_loss")
            report("烟测四消退化超过 10 个百分点")
            return
        if not (RUN / "evaluation/dev_20k/summary.json").exists():
            model = train_to(123)
            summary = evaluate(model, DEV, 20000, "dev_20k")
        else:
            summary = json.loads((RUN / "evaluation/dev_20k/summary.json").read_text())
        if not gate(summary):
            status("stopped_dev_20k")
            report("503808 步开发短局未通过预设门槛")
            return
        if not (RUN / "evaluation/dev_100k/summary.json").exists():
            model = train_to(489)
            summary = evaluate(model, DEV, 100000, "dev_100k")
        else:
            summary = json.loads((RUN / "evaluation/dev_100k/summary.json").read_text())
        if not gate(summary, long=True):
            status("stopped_dev_100k")
            report("2002944 步开发长局未通过预设门槛")
            return
        model = train_to(MAX_TASKS)
        summary = (json.loads((RUN / "evaluation/confirm_100k/summary.json").read_text())
                   if (RUN / "evaluation/confirm_100k/summary.json").exists()
                   else evaluate(model, CONFIRM, 100000, "confirm_100k"))
        passed = gate(summary, long=True) and summary["candidate"]["mean_score"] >= summary["ppo"]["mean_score"]
        status("completed", confirmed=passed, steps=MAX_TASKS * 4096, model_sha256=digest(model))
        report("独立确认通过" if passed else "独立确认未通过")
    except Exception as error:
        RUN.mkdir(parents=True, exist_ok=True)
        atomic_json(RUN / "failure.json", {"time": now(), "error": str(error),
                                          "traceback": traceback.format_exc()})
        status("failed", error=str(error))
        report("队列异常停止，参见 runs/autonomous_rl_research/failure.json")
        raise


if __name__ == "__main__":
    main()
