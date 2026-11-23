"""Train checkpointed PPO in slices and let local Codex pick the next experiment."""

import argparse
import csv
from datetime import datetime
import fcntl
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

from run_lookahead_long_campaign import ROOT, digest, git, now, operation, publish, read_json, write_json


CODEX = Path.home() / ".local/bin/codex"
CONFIG = ROOT / "configs/overnight_student_ppo_20261124.json"
REPORT = Path("reports/experiments/overnight_student_ppo_20261124.json")
DEVELOPMENT_SEEDS = list(range(3266112500, 3266112504))
MINIMUM_TRAINING = 5 * 3600
TARGET_TRAINING = 6 * 3600
SLICE = 45 * 60
VARIANTS = {
    "base": {"seed_offset": 0, "height": 0.0, "beta": 0.3, "learning_rate": 3e-5},
    "try_height": {"seed_offset": 100, "height": 0.004, "beta": 0.3, "learning_rate": 3e-5},
    "try_anchor": {"seed_offset": 200, "height": 0.0, "beta": 0.8, "learning_rate": 1.5e-5},
}
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["continue", "try_height", "try_anchor"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}


def training_totals(output):
    totals = {}
    for name in VARIANTS:
        directory = output / "training" / name
        checkpoint = directory / "transaction_state.json"
        if not checkpoint.is_file():
            continue
        accepted = read_json(checkpoint)
        last_task = accepted["committed_task"]
        metrics = directory / "training_metrics.csv"
        rows = []
        if metrics.is_file():
            rows = [r for r in csv.DictReader(io.StringIO(metrics.read_text()))
                    if int(r["task_id"]) <= last_task]
        next_task = len(rows) + 1
        for task in range(next_task, last_task + 1):
            row = read_json(directory / "metric_history" / f"task_{task:06d}.json")
            rows.append(row)
        seconds = sum(float(row["wall_seconds"]) for row in rows)
        if [int(r["task_id"]) for r in rows] != list(range(1, last_task + 1)) or not math.isfinite(seconds):
            raise RuntimeError(f"Inconsistent committed metrics for {name}")
        recent = {}
        for key in ("reference_kl", "gradient_norm", "mean_risk_increase", "mean_height_penalty"):
            values = [float(row[key]) for row in rows[-8:] if row.get(key) not in (None, "")]
            if values and all(math.isfinite(value) for value in values):
                recent[key] = sum(values) / len(values)
        totals[name] = {
            "steps": accepted["committed_steps"], "task": last_task,
            "training_seconds": seconds, "recent_training_metrics": recent,
        }
    return totals


def completed_seconds(totals):
    return sum(row["training_seconds"] for row in totals.values())


def write_preset(output, baseline, variant):
    change = VARIANTS[variant]
    config = {
        **baseline, "base_seed": baseline["base_seed"] + change["seed_offset"],
        "height_penalty_coef": change["height"],
        "ppo_overrides": {**baseline["ppo_overrides"], "learning_rate": change["learning_rate"]},
        "kl_reference": {**baseline["kl_reference"], "beta": change["beta"]},
    }
    preset = output / f"{variant}.json"
    if preset.is_file():
        if read_json(preset) != config:
            raise RuntimeError(f"Declared training settings changed for {variant}")
    else:
        write_json(preset, config)
    return preset


def run_guarded(output, directory, budget, command, state):
    state["guard"] = str(directory)
    write_json(output / "status.json", state)
    supervisor = [
        sys.executable, str(ROOT / "scripts/resource_guard.py"),
        "--output", str(directory), "--seconds", str(budget), "--", sys.executable, *command,
    ]
    process = subprocess.run(supervisor, cwd=ROOT, check=False)
    result = directory / "result.json"
    return read_json(result) if result.is_file() else {
        "outcome": "guard_result_missing", "exit_code": process.returncode,
    }


def preserve_checkpoint(output, stage, variant, total):
    source = output / "training" / variant / "committed" / f'task_{total["task"]:06d}' / "model.zip"
    target = stage / "candidate.zip"
    temporary = stage / "candidate.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    if digest(source) != digest(target):
        raise RuntimeError("Checkpoint copy differs")
    return target


def decide(stage, evidence, available):
    """Ask authenticated local Codex for a structured experiment choice."""
    decision_path, schema_path = stage / "decision.json", stage / "schema.json"
    write_json(stage / "decision_input.json", evidence)
    write_json(schema_path, SCHEMA)
    prompt = (
        "Supervise a bounded experiment for a local Tetris game benchmark. "
        "Use only the supplied JSON evidence. Choose exactly one allowed action. Do not call tools. "
        "Compare common-length score, actual early endings, high stacks, and divergence from the frozen student. "
        "Consider consistency across stages and seeds. Scores near parity on a small development set are uncertain. "
        "Continue when the evidence is healthy or inconclusive. Choose try_height when stack risk repeatedly rises "
        "while score still holds; choose try_anchor when the trained policy drifts and repeatedly loses score. "
        "A switch starts a fresh controlled comparison from the same frozen student. "
        "Leave a concise reason in Chinese and never recommend deployment from this development set.\n"
        + json.dumps(evidence, ensure_ascii=False, allow_nan=False)
    )
    try:
        with TemporaryDirectory(prefix="tetris-codex-judge-") as scratch, (stage / "codex.log").open("w") as log:
            command = [
                str(CODEX), "exec", "--ephemeral", "--ignore-user-config", "--sandbox", "read-only",
                "--skip-git-repo-check", "--cd", scratch, "--output-schema", str(schema_path),
                "--output-last-message", str(decision_path), "-",
            ]
            subprocess.run(command, input=prompt, text=True, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=140)
        answer = read_json(decision_path)
        if (set(answer) != {"action", "reason"} or answer["action"] not in available
                or not isinstance(answer["reason"], str) or not answer["reason"].strip()):
            raise ValueError("Codex returned a decision outside the supplied options")
        result = {"source": "local_codex", "action": answer["action"], "reason": answer["reason"][:1000]}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result = {"source": "fallback", "action": "continue", "reason": f"Codex decision unavailable: {type(exc).__name__}"}
    write_json(stage / "applied_decision.json", result)
    return result


def seconds_to(deadline):
    return int(deadline.timestamp() - time.time())


def train_and_assess(output, state, baseline, deadline):
    errors = 0
    while seconds_to(deadline) >= 240:
        totals = training_totals(output)
        remaining = TARGET_TRAINING - completed_seconds(totals)
        if remaining <= 0:
            break
        variant = state["variant"]
        stage_number = state["next_stage"]
        state["next_stage"] += 1
        stage = output / "stages" / f"{stage_number:02d}"
        stage.mkdir(parents=True)
        directory = output / "training" / variant
        option = (["--resume"] if (directory / "transaction_state.json").is_file()
                  else ["--config-file", str(write_preset(output, baseline, variant))])
        before = totals.get(variant, {}).get("steps", 0)
        train_budget = min(SLICE, max(300, int(remaining) + 90), seconds_to(deadline) - 210)
        state.update(phase="training", stage=stage_number, training_directory=str(directory),
                     training_seconds=completed_seconds(totals))
        outcome = run_guarded(output, stage / "training_guard", train_budget, [
            "-m", "training.train_vector_transaction", "--run-dir", str(directory), *option,
        ], state)

        totals = training_totals(output)
        current = totals.get(variant, {})
        progress = current.get("steps", 0) > before
        errors = 0 if progress else errors + 1
        row = {"stage": stage_number, "variant": variant, "training_guard": outcome,
               "committed": current, "training_seconds": completed_seconds(totals)}
        state["stages"].append(row)
        write_json(output / "status.json", state)
        if errors >= 3 or outcome.get("outcome") == "interrupted":
            break
        if outcome.get("outcome", "").startswith("stop:"):
            operation(output, f"训练段触发资源保护 {outcome['outcome']}，暂停后重新检查")
            time.sleep(max(0, min(150, seconds_to(deadline) - 60)))
            continue
        if not progress:
            time.sleep(max(0, min(60, seconds_to(deadline) - 60)))
            continue
        candidate = preserve_checkpoint(output, stage, variant, current)
        row["checkpoint"] = {"path": str(candidate), "sha256": digest(candidate)}
        if seconds_to(deadline) < 320:
            write_json(output / "status.json", state)
            break
        state["phase"] = "evaluation"
        evaluation = stage / "evaluation.json"
        row["evaluation_attempts"] = []
        evaluation_guard = {"outcome": "evaluation_not_started"}
        for attempt in (1, 2):
            if seconds_to(deadline) < 320:
                break
            evaluation_guard = run_guarded(
                output, stage / f"evaluation_guard_{attempt}", min(220, seconds_to(deadline) - 160), [
                    "-m", "scripts.evaluate_overnight_student_ppo", "--reference", baseline["initial_model"]["path"],
                    "--candidate", str(candidate), "--output", str(evaluation), "--cap", "4096",
                    "--seeds", *map(str, DEVELOPMENT_SEEDS),
                ], state,
            )
            row["evaluation_attempts"].append(evaluation_guard)
            if evaluation_guard["outcome"] == "completed" and evaluation.is_file():
                break
            if evaluation_guard["outcome"].startswith("stop:"):
                time.sleep(max(0, min(150, seconds_to(deadline) - 60)))
        else:
            evaluation_guard = {"outcome": "evaluation_failed_twice"}
        if not evaluation.is_file() or evaluation_guard["outcome"] != "completed":
            operation(output, f"第 {stage_number} 段回放未完成，保留检查点并停止新增训练")
            write_json(output / "status.json", state)
            break
        report = read_json(evaluation)
        row["evaluation"] = report["summary"]
        state["phase"] = "codex_decision"
        write_json(output / "status.json", state)
        available = ["continue", *(name for name in VARIANTS if name != "base" and name not in state["used"])]
        evidence = {
            "current_variant": variant, "settings": VARIANTS[variant], "allowed_actions": available,
            "development_pairs": report["pairs"], "development_summary": report["summary"],
            "total_training_seconds": row["training_seconds"],
            "recent_training_metrics": current["recent_training_metrics"],
            "previous": [
                {"variant": prior["variant"], "metrics": prior.get("evaluation"),
                 "decision": prior.get("decision", {}).get("action")}
                for prior in state["stages"][:-1]
            ],
        }
        row["judge_attempts"] = []
        for attempt in (1, 2):
            if seconds_to(deadline) < 160:
                break
            decision_stage = stage / f"judge_{attempt}"
            decision_stage.mkdir()
            result = decide(decision_stage, evidence, available)
            row["judge_attempts"].append(result)
            if result["source"] == "local_codex":
                row["decision"] = result
                break
            if attempt == 1 and seconds_to(deadline) >= 200:
                operation(output, f"第 {stage_number} 段 Codex 调用未返回可用决策，准备重试")
                time.sleep(20)
        decision = row.get("decision")
        if decision is None:
            operation(output, f"第 {stage_number} 段缺少可用 Codex 决策，保留检查点并停止新增训练")
            write_json(output / "status.json", state)
            break
        if decision["action"] != "continue":
            state["variant"] = decision["action"]
            state["used"].append(state["variant"])
        operation(output, f"第 {stage_number} 段评测后，{decision['source']} 选择 {decision['action']}")
        write_json(output / "status.json", state)
    return training_totals(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    if any(date.tzinfo is None for date in (args.start, args.end)) or args.end <= args.start:
        parser.error("start and end must be ordered timestamps with explicit timezones")
    if time.time() < args.start.timestamp() or seconds_to(args.end) < 240:
        parser.error("outside the planned overnight training interval")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "overnight.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = output / "status.json"
        branch, base_commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
        if not branch:
            raise RuntimeError("A named branch is needed for archived results")
        baseline = read_json(CONFIG)
        if digest(ROOT / baseline["initial_model"]["path"]) != baseline["initial_model"]["sha256"]:
            raise RuntimeError("Frozen starting model differs")
        specification = {
            "start": args.start.isoformat(), "end": args.end.isoformat(),
            "minimum_training_seconds": MINIMUM_TRAINING, "target_training_seconds": TARGET_TRAINING,
            "training_config_sha256": digest(CONFIG), "variants": VARIANTS, "development_seeds": DEVELOPMENT_SEEDS,
            "source_sha256": baseline["initial_model"]["sha256"],
        }
        plan = output / "protocol.json"
        if plan.is_file() and read_json(plan) != specification:
            raise RuntimeError("Existing overnight protocol differs")
        if not plan.is_file():
            write_json(plan, specification)
        previous = read_json(status) if status.is_file() else {}
        if previous.get("phase") == "finished":
            print("overnight training already archived", previous.get("pushed_commit"), flush=True)
            return
        state = {
            "phase": "starting", "started": previous.get("started", now()),
            "variant": previous.get("variant", "base"), "used": previous.get("used", ["base"]),
            "stages": previous.get("stages", []), "next_stage": previous.get("next_stage", 1),
        }
        write_json(status, state)
        operation(output, "夜间监督任务启动，各段用真实游戏结果决定下一方向")
        try:
            totals = train_and_assess(output, state, baseline, args.end)
            seconds = completed_seconds(totals)
            calls = sum(row.get("decision", {}).get("source") == "local_codex" for row in state["stages"])
            state.update(phase="archiving", ended=now(), training_seconds=seconds,
                         training_minimum_met=seconds >= MINIMUM_TRAINING, verified_codex_decisions=calls)
            write_json(status, state)
            archive = {
                "protocol": specification, "training": totals, "stages": state["stages"],
                "training_seconds": seconds, "training_minimum_met": seconds >= MINIMUM_TRAINING,
                "verified_codex_decisions": calls,
            }
            write_json(output / "archive.json", archive)
            if (git("branch", "--show-current") != branch or git("rev-parse", "HEAD") != base_commit
                    or git("status", "--porcelain", "--", str(REPORT))):
                raise RuntimeError("Checkout or report changed; archive.json is available locally")
            write_json(ROOT / REPORT, archive)
            operation(output, f"夜间训练收尾，已记录训练 {seconds:.0f} 秒、Codex 已应用决策 {calls} 次")
            state["pushed_commit"] = publish(output, branch, base_commit, REPORT, "Archive the overnight Codex-supervised PPO run")
            state["phase"] = "finished"
            write_json(status, state)
        except Exception as exc:
            state.update(phase="failed", ended=now(), error=repr(exc))
            write_json(status, state)
            operation(output, f"夜间监督任务中断，保留已提交训练，原因 {exc!r}")
            raise


if __name__ == "__main__":
    main()
