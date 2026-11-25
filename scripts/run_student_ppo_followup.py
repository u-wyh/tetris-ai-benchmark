"""Execute the next PPO experiments without waiting for a chat turn between runs."""

import fcntl
from pathlib import Path

from run_lookahead_long_campaign import ROOT, git, now, operation, publish, read_json, write_json
from run_student_ppo_confirmation import run


OUTPUT = ROOT / "runs/student_ppo_followup_20261125"
SOURCE = Path("reports/experiments/student_ppo_survival_resume_20261125.json")
REPORT = Path("reports/experiments/student_ppo_followup_20261125.json")
CONFIRMATION_SEEDS = list(range(3269114200, 3269114208))
EXPANSION_SEEDS = list(range(3269114300, 3269114308))
FALLBACK_HEIGHT = 10


def next_action(report):
    if report["codex_decision"] is not None:
        return {"source": "local_codex", "action": report["codex_decision"]["action"]}
    summary = report["summary"]
    if report["eligible_to_expand"]:
        action = "expand_evaluation"
    elif (summary["candidate_deaths"] > summary["reference_deaths"]
          or summary["candidate_pieces"] < summary["reference_pieces"]):
        action = "focus_survival"
    else:
        action = "hold_candidate"
    return {"source": "fixed_protocol", "action": action}


def execute(state, name, seeds, cap, fallback_height=None):
    folder = OUTPUT / name
    report_path = REPORT.with_stem(f"{REPORT.stem}_{name}")
    state.update(phase="running", active_experiment=name, updated_at=now())
    write_json(OUTPUT / "status.json", state)
    report = run(folder, SOURCE, report_path, seeds, cap, fallback_height)
    decision = next_action(report)
    state["experiments"].append({
        "name": name, "report": str(report_path), "summary": report["summary"],
        "fallback_height": fallback_height, "decision": decision,
    })
    write_json(OUTPUT / "status.json", state)
    return decision["action"]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status = OUTPUT / "status.json"
    with (OUTPUT / "workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.is_file() and read_json(status).get("phase") == "finished":
            return
        state = {"phase": "starting", "started_at": now(), "experiments": [], "active_experiment": None}
        write_json(status, state)
        try:
            action = execute(state, "confirmation", CONFIRMATION_SEEDS, 20_000)
            fallback_height = None
            if action == "focus_survival":
                fallback_height = FALLBACK_HEIGHT
                action = execute(state, "fallback", CONFIRMATION_SEEDS, 20_000, fallback_height)
            if action == "expand_evaluation":
                action = execute(state, "expansion", EXPANSION_SEEDS, 35_000, fallback_height)
            state.update(phase="archiving", active_experiment=None, ended_at=now(), final_action=action)
            write_json(status, state)
            branch, commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
            report = ROOT / REPORT
            if not branch or report.exists():
                raise RuntimeError("Workflow requires a named branch and a new summary report")
            write_json(report, {key: value for key, value in state.items() if key != "phase"})
            operation(OUTPUT, f"自动实验流程收尾，完成 {len(state['experiments'])} 轮，最后决策 {action}")
            pushed = publish(OUTPUT, branch, commit, REPORT, "Archive automatically chained PPO experiments")
            state.update(phase="finished", pushed_commit=pushed)
            write_json(status, state)
        except Exception as exc:
            state.update(phase="failed", error=repr(exc), ended_at=now())
            write_json(status, state)
            operation(OUTPUT, f"自动实验流程中断，阶段结果保留，原因 {exc!r}")
            raise


if __name__ == "__main__":
    main()
