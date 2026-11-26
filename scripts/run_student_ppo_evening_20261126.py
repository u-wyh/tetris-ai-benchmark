"""Execute the next controlled survival experiment and archive the evening results."""

from datetime import datetime
import fcntl
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from run_lookahead_long_campaign import ROOT, git, now, operation, publish, read_json, write_json
from run_student_ppo_confirmation import run


OUTPUT = ROOT / "runs/student_ppo_evening_20261126"
RECOVERY = ROOT / "runs/student_ppo_balanced_recovery_20261126/status.json"
BASELINE = Path("reports/experiments/student_ppo_balanced_live_20261125.json")
REPORT = Path("reports/experiments/student_ppo_evening_20261126.json")
TRAINING_REPORT = REPORT.with_stem(f"{REPORT.stem}_stronger_kl")
EVALUATION_REPORT = REPORT.with_stem(f"{REPORT.stem}_evaluation")
SEEDS = list(range(3269114600, 3269114608))
DEADLINE = datetime.fromisoformat("2026-11-26T20:00:00+08:00")
TARGET_SECONDS = 4 * 3600
MINIMUM_SECONDS = 3 * 3600


def validated_fallback(recovery):
    for experiment in recovery.get("experiments", []):
        if experiment["name"] == "holdout" and experiment["decision"]["action"] == "expand_evaluation":
            return experiment["height"]
    return None


def train(output):
    launcher = (
        "from pathlib import Path\n"
        "import sys\n"
        "sys.path.insert(0, str(Path.cwd() / 'scripts'))\n"
        "import supervise_overnight_student_ppo as ppo\n"
        "ppo.VARIANTS = {'base': {**ppo.VARIANTS['base'], 'beta': 0.8}}\n"
        f"ppo.TARGET_TRAINING = {TARGET_SECONDS}\n"
        f"ppo.MINIMUM_TRAINING = {MINIMUM_SECONDS}\n"
        "raise SystemExit(ppo.main())\n"
    )
    command = [
        sys.executable, "-c", launcher, "--output", str(output), "--start", now(),
        "--end", DEADLINE.isoformat(), "--reward-profile", "survival_balanced_v1",
        "--report", str(TRAINING_REPORT),
    ]
    process = subprocess.run(command, cwd=ROOT, check=False)
    status = output / "status.json"
    result = read_json(status) if status.is_file() else {}
    if result.get("phase") != "finished" or not result.get("pushed_commit"):
        raise RuntimeError(f"Training did not archive its result, exit code {process.returncode}")
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status = OUTPUT / "status.json"
    with (OUTPUT / "workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.is_file() and read_json(status).get("phase") == "finished":
            return
        recovery = read_json(RECOVERY)
        if recovery["phase"] not in {"finished", "failed"}:
            raise RuntimeError("The previous campaign still owns the resources")
        state = {"phase": "starting", "started_at": now(), "recovery_status": str(RECOVERY.relative_to(ROOT)),
                 "recovery_commit": recovery.get("pushed_commit"), "recovery_results": recovery.get("experiments", [])}
        try:
            height = validated_fallback(recovery)
            source = BASELINE
            cap = 35_000 if height is not None else 20_000
            if height is None and datetime.now(ZoneInfo("Asia/Shanghai")) < DEADLINE:
                state.update(phase="training", active_experiment="stronger_kl")
                write_json(status, state)
                result = train(OUTPUT / "training")
                state["training"] = {"report": str(TRAINING_REPORT), "commit": result["pushed_commit"],
                                     "outcome": result["outcome"], "seconds": result["training_seconds"],
                                     "codex_decisions": result["verified_codex_decisions"],
                                     "codex_live_reviews": result["verified_codex_live_reviews"]}
                source = TRAINING_REPORT
            elif height is None:
                raise RuntimeError("The reserved training window has closed")
            state.update(phase="evaluating", active_experiment="independent_replay")
            write_json(status, state)
            report = run(OUTPUT / "evaluation", source, EVALUATION_REPORT, SEEDS, cap, height)
            state["evaluation"] = {"report": str(EVALUATION_REPORT), "summary": report["summary"],
                                   "eligible_to_expand": report["eligible_to_expand"],
                                   "codex_decision": report["codex_decision"]}
        except Exception as exc:
            state["error"] = repr(exc)
        state.update(phase="archiving", active_experiment=None, ended_at=now())
        write_json(status, state)
        branch, commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
        if not branch or (ROOT / REPORT).exists():
            raise RuntimeError("A named branch and a new evening report are required")
        write_json(ROOT / REPORT, {key: value for key, value in state.items() if key != "phase"})
        description = "未全部完成，已归档可核实结果" if state.get("error") else "已完成并归档"
        operation(OUTPUT, f"11 月 26 日晚间核验工作流{description}")
        pushed = publish(OUTPUT, branch, commit, REPORT, "Archive November 26 evening experiment results")
        state.update(phase="failed" if state.get("error") else "finished", pushed_commit=pushed)
        write_json(status, state)
        return int(bool(state.get("error")))


if __name__ == "__main__":
    raise SystemExit(main())
