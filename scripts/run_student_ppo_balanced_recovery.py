"""Test earlier high-stack recovery on identical seeds, then validate a passing policy."""

import fcntl
from pathlib import Path

from run_lookahead_long_campaign import ROOT, digest, git, now, operation, publish, read_json, write_json
from run_student_ppo_confirmation import run, specification
from run_student_ppo_followup import next_action


OUTPUT = ROOT / "runs/student_ppo_balanced_recovery_20261126"
SOURCE = Path("reports/experiments/student_ppo_balanced_live_20261125.json")
RAW = Path("reports/experiments/student_ppo_balanced_confirmation_20261125.json")
REPORT = Path("reports/experiments/student_ppo_balanced_recovery_20261126.json")
HOLDOUT_SEEDS = list(range(3269114500, 3269114508))
DEVELOPMENT_CAP = 20_000
HOLDOUT_CAP = 35_000
THRESHOLDS = (8, 6)


def evaluate(state, name, seeds, cap, height):
    report_path = REPORT.with_stem(f"{REPORT.stem}_{name}")
    state.update(phase="running", active_experiment=name, updated_at=now())
    write_json(OUTPUT / "status.json", state)
    result = run(OUTPUT / name, SOURCE, report_path, seeds, cap, height)
    decision = next_action(result)
    state["experiments"].append({"name": name, "height": height, "report": str(report_path),
                                 "summary": result["summary"], "decision": decision})
    write_json(OUTPUT / "status.json", state)
    return decision["action"]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status = OUTPUT / "status.json"
    with (OUTPUT / "workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.is_file() and read_json(status).get("phase") == "finished":
            return
        raw = read_json(ROOT / RAW)
        seeds = raw["protocol"]["seeds"]
        if raw["protocol"] != specification(SOURCE, seeds, DEVELOPMENT_CAP) or set(seeds) & set(HOLDOUT_SEEDS):
            raise RuntimeError("Raw comparison protocol or holdout isolation differs")
        state = {"phase": "starting", "started_at": now(), "active_experiment": None,
                 "raw_report": str(RAW), "raw_sha256": digest(ROOT / RAW), "raw_summary": raw["summary"],
                 "thresholds": THRESHOLDS, "holdout_seeds": HOLDOUT_SEEDS, "experiments": []}
        write_json(status, state)
        try:
            for height in THRESHOLDS:
                action = evaluate(state, f"height_{height}", seeds, DEVELOPMENT_CAP, height)
                if action == "expand_evaluation":
                    action = evaluate(state, "holdout", HOLDOUT_SEEDS, HOLDOUT_CAP, height)
                    break
                if action == "hold_candidate":
                    break
            state.update(phase="archiving", active_experiment=None, ended_at=now(), final_action=action)
            write_json(status, state)
            branch, commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
            report = ROOT / REPORT
            if not branch or report.exists():
                raise RuntimeError("Workflow requires a named branch and a new report")
            write_json(report, {key: value for key, value in state.items() if key != "phase"})
            operation(OUTPUT, f"生存回退对照收尾，共 {len(state['experiments'])} 轮，最后决策 {action}")
            pushed = publish(OUTPUT, branch, commit, REPORT, "Archive balanced PPO survival recovery experiments")
            state.update(phase="finished", pushed_commit=pushed)
            write_json(status, state)
        except Exception as exc:
            state.update(phase="failed", error=repr(exc), ended_at=now())
            write_json(status, state)
            operation(OUTPUT, f"生存回退对照中断，阶段结果保留，原因 {exc!r}")
            raise


if __name__ == "__main__":
    main()
