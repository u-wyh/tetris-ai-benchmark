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
LIVE_REVIEW_INTERVAL = 15 * 60
VARIANTS = {
    "base": {"seed_offset": 0, "height": 0.0, "beta": 0.3, "learning_rate": 3e-5},
    "try_height": {"seed_offset": 100, "height": 0.004, "beta": 0.3, "learning_rate": 3e-5},
    "try_anchor": {"seed_offset": 200, "height": 0.0, "beta": 0.8, "learning_rate": 1.5e-5},
}
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
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


def compatible_variants(baseline):
    if baseline.get("reward_profile", "default") == "default":
        return VARIANTS
    return {name: settings for name, settings in VARIANTS.items() if settings["height"] == 0}


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


def require_runtime():
    """Check the timed service environment before the trainer writes a config."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is missing from the training service PATH")
    subprocess.run([node, "--version"], stdout=subprocess.DEVNULL, check=True, timeout=8)
    if not CODEX.is_file():
        raise RuntimeError("The local Codex executable is missing")


def training_options(output, stage, directory, baseline, variant):
    state_file = directory / "transaction_state.json"
    if state_file.is_file():
        return ["--resume"]
    if (directory / "config.json").is_file():
        # Initialization may have written a config before publishing its first
        # transaction pointer. Keep the failed attempt before trying afresh.
        with (directory / ".train.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if state_file.is_file():
                return ["--resume"]
            committed = directory / "committed"
            if committed.is_dir() and any(committed.iterdir()):
                raise RuntimeError("Training checkpoints exist without a transaction pointer")
            archived = stage / "failed_initialization"
            directory.rename(archived)
        operation(output, f"保留未完成的训练初始化到 {archived}")
    return ["--config-file", str(write_preset(output, baseline, variant))]


def run_guarded(output, directory, budget, command, state, review=None):
    state["guard"] = str(directory)
    write_json(output / "status.json", state)
    supervisor = [
        sys.executable, str(ROOT / "scripts/resource_guard.py"),
        "--output", str(directory), "--seconds", str(budget), "--", sys.executable, *command,
    ]
    requested_review = None
    if review is None:
        code = subprocess.run(supervisor, cwd=ROOT, check=False).returncode
    else:
        with subprocess.Popen(supervisor, cwd=ROOT) as process:
            next_review = time.monotonic() + LIVE_REVIEW_INTERVAL
            while True:
                try:
                    code = process.wait(timeout=max(0, next_review - time.monotonic()))
                    break
                except subprocess.TimeoutExpired:
                    if process.poll() is not None:
                        code = process.returncode
                        break
                    decision = review(directory)
                    enacted = decision["action"] == "evaluate_now" and process.poll() is None
                    decision["applied"] = False
                    write_json(output / "status.json", state)
                    if enacted:
                        requested_review = decision
                        process.terminate()
                        code = process.wait()
                        break
                    next_review += LIVE_REVIEW_INTERVAL
    result = directory / "result.json"
    terminal = read_json(result) if result.is_file() else {"outcome": "guard_result_missing", "exit_code": code}
    if requested_review is not None and terminal["outcome"] == "interrupted":
        requested_review["applied"] = True
        write_json(output / "status.json", state)
        terminal = {**terminal, "guard_outcome": "interrupted", "outcome": "review_requested"}
    return terminal


def preserve_checkpoint(output, stage, variant, total):
    source = output / "training" / variant / "committed" / f'task_{total["task"]:06d}' / "model.zip"
    target = stage / "candidate.zip"
    temporary = stage / "candidate.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    if digest(source) != digest(target):
        raise RuntimeError("Checkpoint copy differs")
    return target


def ask_codex(stage, evidence, available, instructions):
    decision_path, schema_path = stage / "decision.json", stage / "schema.json"
    write_json(stage / "decision_input.json", evidence)
    schema = {**SCHEMA, "properties": {
        **SCHEMA["properties"], "action": {"type": "string", "enum": available},
    }}
    write_json(schema_path, schema)
    prompt = instructions + "\n" + json.dumps(evidence, ensure_ascii=False, allow_nan=False)
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


def decide(stage, evidence, available):
    """Ask authenticated local Codex for a structured experiment choice."""
    return ask_codex(stage, evidence, available, (
        "Supervise a bounded experiment for a local Tetris game benchmark. "
        "Use only the supplied JSON evidence. Choose exactly one allowed action. Do not call tools. "
        "Compare common-length score, actual early endings, high stacks, and divergence from the frozen student. "
        "Consider consistency across stages and seeds. Scores near parity on a small development set are uncertain. "
        "Continue when the evidence is healthy or inconclusive. If available, choose try_height when stack risk repeatedly rises "
        "while score still holds; choose try_anchor when the trained policy drifts and repeatedly loses score. "
        "If base is available, choose it to compare against training without the current variant's intervention. "
        "A switch starts a fresh controlled comparison from the same frozen student. "
        "Leave a concise reason in Chinese and never recommend deployment from this development set."
    ))


def latest_resource_sample(directory):
    sample_file = directory / "samples.jsonl"
    if not sample_file.is_file():
        return None
    with sample_file.open("rb") as stream:
        stream.seek(max(0, sample_file.stat().st_size - 8192))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            sample = json.loads(line)
        except ValueError:
            continue
        return {key: sample.get(key) for key in ("at", "cpu_c", "gpu_c", "memory_mib", "decision")}
    return None


def review_training(output, stage, variant, before, state, reviews, directory):
    check = stage / f"live_{len(reviews) + 1:02d}"
    check.mkdir()
    totals = training_totals(output)
    committed = totals.get(variant, {})
    available = ["continue"]
    if committed.get("steps", 0) > before:
        available.append("evaluate_now")
    evidence = {
        "variant": variant, "allowed_actions": available, "segment_start_steps": before,
        "committed": committed, "resource_sample": latest_resource_sample(directory),
        "last_evaluation": next((r["evaluation"] for r in reversed(state["stages"])
                                 if "evaluation" in r), None),
        "previous_checks": [{"steps": r["steps"], "recent_training_metrics": r["recent_training_metrics"],
                             "action": r["action"]} for r in reviews],
    }
    result = ask_codex(check, evidence, available, (
        "You supervise a running local Tetris PPO training segment. Use only the supplied data and no tools. "
        "Continue if committed checkpoints progress and training data look stable or inconclusive. "
        "Choose evaluate_now only if the observed trend gives a concrete reason to pause at the latest "
        "committed checkpoint and run the paired game evaluation before the regular end of this segment. "
        "Resource limits are independently enforced; do not waive them. "
        "Training loss and KL alone do not establish gameplay quality. Give a concise reason in Chinese."
    ))
    row = {
        **result, "at": now(), "steps": committed.get("steps", 0),
        "recent_training_metrics": committed.get("recent_training_metrics", {}),
    }
    reviews.append(row)
    operation(output, f"第 {state['stage']} 段运行中审查，{row['source']} 建议 {row['action']}")
    return row


def seconds_to(deadline):
    return int(deadline.timestamp() - time.time())


def train_and_assess(output, state, baseline, deadline):
    errors = 0
    variants = compatible_variants(baseline)
    while seconds_to(deadline) >= 240:
        totals = training_totals(output)
        remaining = TARGET_TRAINING - completed_seconds(totals)
        variant = state["variant"]
        if remaining <= 0 or totals.get(variant, {}).get("steps", 0) >= baseline["target_total_steps"]:
            break
        stage_number = state["next_stage"]
        state["next_stage"] += 1
        stage = output / "stages" / f"{stage_number:02d}"
        stage.mkdir(parents=True)
        directory = output / "training" / variant
        option = training_options(output, stage, directory, baseline, variant)
        before = totals.get(variant, {}).get("steps", 0)
        train_budget = min(SLICE, max(300, int(remaining) + 90), seconds_to(deadline) - 210)
        state.update(phase="training", stage=stage_number, training_directory=str(directory),
                     training_seconds=completed_seconds(totals))
        reviews = []
        state["live_reviews"] = reviews
        outcome = run_guarded(output, stage / "training_guard", train_budget, [
            "-m", "training.train_vector_transaction", "--run-dir", str(directory), *option,
        ], state, review=lambda guard: review_training(output, stage, variant, before, state, reviews, guard))

        totals = training_totals(output)
        current = totals.get(variant, {})
        progress = current.get("steps", 0) > before
        errors = 0 if progress else errors + 1
        row = {"stage": stage_number, "variant": variant, "training_guard": outcome,
               "committed": current, "training_seconds": completed_seconds(totals),
               "live_reviews": reviews}
        state["stages"].append(row)
        state["training_seconds"] = row["training_seconds"]
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
        available = ["continue", *(name for name in variants if name not in state["used"])]
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
            if available == ["continue"]:
                row["decision"] = {
                    "source": "fixed_protocol", "action": "continue",
                    "reason": "Codex unavailable; the protocol only permits the current variant.",
                }
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


def load_continuation(output, source, specification):
    if output == source or source in output.parents:
        raise RuntimeError("Continuation requires a separate output directory")
    original = read_json(source / "status.json")
    archive_path = source / "archive.json"
    archive = read_json(archive_path)
    if original.get("phase") != "finished" or archive.get("outcome") != "incomplete" or not original.get("pushed_commit"):
        raise RuntimeError("Continuation requires an archived incomplete run")
    if any(archive["protocol"].get(key) != value for key, value in specification.items() if key not in {"start", "end"}):
        raise RuntimeError("Continuation training protocol differs")
    if archive["stages"] != original["stages"] or training_totals(source) != archive["training"]:
        raise RuntimeError("Source checkpoints or stage records differ from the archive")
    specification["continuation"] = {
        "source": str(source), "source_archive_sha256": digest(archive_path),
        "source_commit": original["pushed_commit"],
        "previous_training_seconds": archive["training_seconds"],
    }
    return original


def copy_continuation_checkpoints(output, source, original):
    expected = training_totals(source)
    training = output / "training"
    if not training.exists():
        with TemporaryDirectory(dir=output) as temporary:
            snapshot = Path(temporary) / "training"
            shutil.copytree(source / "training", snapshot)
            snapshot.rename(training)
    if training_totals(output) != expected:
        raise RuntimeError("Continuation checkpoints differ from the incomplete run")
    return {key: original[key] for key in ("variant", "used", "stages", "next_stage")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=datetime.fromisoformat, required=True)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--initial-variant", choices=VARIANTS, default="base")
    parser.add_argument("--reward-profile", choices=("default", "survival_balanced_v1"), default="default")
    parser.add_argument("--resume-from", type=Path, help="Archived incomplete run whose checkpoints and stages to continue")
    args = parser.parse_args()
    report = (ROOT / args.report).resolve()
    if report.parent != ROOT / "reports/experiments" or report.suffix != ".json":
        parser.error("report must be a JSON file in reports/experiments")
    report_path = report.relative_to(ROOT)
    if any(date.tzinfo is None for date in (args.start, args.end)) or args.end <= args.start:
        parser.error("start and end must be ordered timestamps with explicit timezones")
    if time.time() < args.start.timestamp() or seconds_to(args.end) < 240:
        parser.error("outside the planned training interval")
    require_runtime()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "overnight.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = output / "status.json"
        branch, base_commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
        if not branch:
            raise RuntimeError("A named branch is needed for archived results")
        baseline = read_json(CONFIG)
        if args.reward_profile != "default":
            baseline["reward_profile"] = args.reward_profile
        variants = compatible_variants(baseline)
        if args.initial_variant not in variants:
            parser.error("the initial variant is incompatible with the selected reward profile")
        if digest(ROOT / baseline["initial_model"]["path"]) != baseline["initial_model"]["sha256"]:
            raise RuntimeError("Frozen starting model differs")
        specification = {
            "start": args.start.isoformat(), "end": args.end.isoformat(),
            "minimum_training_seconds": MINIMUM_TRAINING, "target_training_seconds": TARGET_TRAINING,
            "training_config_sha256": digest(CONFIG), "variants": variants, "development_seeds": DEVELOPMENT_SEEDS,
            "live_review_interval_seconds": LIVE_REVIEW_INTERVAL,
            "source_sha256": baseline["initial_model"]["sha256"],
        }
        if args.reward_profile != "default":
            specification["reward_profile"] = args.reward_profile
        if args.initial_variant != "base":
            specification["initial_variant"] = args.initial_variant
        source = args.resume_from.resolve() if args.resume_from else None
        if source is not None:
            original = load_continuation(output, source, specification)
        plan = output / "protocol.json"
        if plan.is_file() and read_json(plan) != specification:
            raise RuntimeError("Existing training protocol differs")
        if not plan.is_file():
            write_json(plan, specification)
        previous = read_json(status) if status.is_file() else {}
        if previous.get("phase") == "finished":
            print("training already archived", previous.get("pushed_commit"), flush=True)
            return 0 if previous.get("outcome") == "completed" else 1
        if report.exists():
            raise RuntimeError("Final report path is already in use")
        if source is not None and not previous:
            previous = copy_continuation_checkpoints(output, source, original)
            operation(output, f"已接续 {source.name} 的检查点，已有训练 {original['training_seconds']:.0f} 秒")
        state = {
            "phase": "starting", "started": previous.get("started", now()),
            "variant": previous.get("variant", args.initial_variant),
            "used": previous.get("used", [args.initial_variant]),
            "stages": previous.get("stages", []), "next_stage": previous.get("next_stage", 1),
        }
        write_json(status, state)
        operation(output, f"Codex 监督训练启动，初始配置 {state['variant']}，各段用真实游戏结果决定下一方向")
        try:
            totals = train_and_assess(output, state, baseline, args.end)
            seconds = completed_seconds(totals)
            calls = sum(row.get("decision", {}).get("source") == "local_codex" for row in state["stages"])
            live_calls = sum(review["source"] == "local_codex" for stage in state["stages"]
                             for review in stage.get("live_reviews", []))
            early_evaluations = sum(review.get("applied", False) for stage in state["stages"]
                                    for review in stage.get("live_reviews", []))
            outcome = "completed" if seconds >= MINIMUM_TRAINING and calls else "incomplete"
            state.update(phase="archiving", outcome=outcome, ended=now(), training_seconds=seconds,
                         training_minimum_met=seconds >= MINIMUM_TRAINING, verified_codex_decisions=calls,
                         verified_codex_live_reviews=live_calls, early_evaluations=early_evaluations)
            write_json(status, state)
            archive = {
                "protocol": specification, "training": totals, "stages": state["stages"], "outcome": outcome,
                "training_seconds": seconds, "training_minimum_met": seconds >= MINIMUM_TRAINING,
                "verified_codex_decisions": calls, "verified_codex_live_reviews": live_calls,
                "early_evaluations": early_evaluations,
            }
            write_json(output / "archive.json", archive)
            if (git("branch", "--show-current") != branch or git("rev-parse", "HEAD") != base_commit
                    or git("status", "--porcelain", "--", str(report_path))):
                raise RuntimeError("Checkout or report changed; archive.json is available locally")
            write_json(report, archive)
            operation(output, f"监督训练收尾，状态 {outcome}，已记录训练 {seconds:.0f} 秒、Codex 已应用决策 {calls} 次")
            state["pushed_commit"] = publish(output, branch, base_commit, report_path, "Archive the Codex-supervised PPO run")
            state["phase"] = "finished"
            write_json(status, state)
            return 0 if outcome == "completed" else 1
        except Exception as exc:
            state.update(phase="failed", ended=now(), error=repr(exc))
            write_json(status, state)
            operation(output, f"监督训练中断，保留已提交训练，原因 {exc!r}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
