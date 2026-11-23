"""Run a detached, resource-guarded high-stack fit and a 100K paired replay."""

import argparse
from datetime import datetime
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")
TRAIN_SEEDS = list(range(3256112460, 3256112484))
VALIDATION_SEEDS = list(range(3256112490, 3256112498))
REPLAY_SEEDS = list(range(3256112510, 3256112518))
REPORT = Path("reports/experiments/lookahead_long_campaign_20261123.json")
FIT_SOURCE = ROOT / "runs/lookahead_gate_20261123/fit_next3"


class CampaignStopped(RuntimeError):
    """A signal requested a controlled shutdown."""


def now():
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def digest(path):
    digestor = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digestor.update(chunk)
    return digestor.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def operation(output, message):
    with (ROOT / "OPERATIONS.md").open("a") as log:
        log.write(f"\n- {now()}：{message}。证据：`{output}/`。\n")


def stop(signum, _frame):
    raise CampaignStopped(signal.Signals(signum).name)


def run_phase(name, seconds, arguments, output, state):
    directory = output / name
    if (directory / "result.json").is_file():
        previous = state["phases"].get(name, {})
        return {**previous, "outcome": "completed", "data": str(directory)}
    attempt_id = datetime.now(SHANGHAI).strftime("%Y%m%d-%H%M%S-%f")
    guard = output / "guards" / f"{name}-{attempt_id}"
    current = {"data": str(directory), "guard": str(guard), "outcome": "running", "started": now()}
    state["phases"][name] = current
    state["phase"] = name
    write_json(output / "status.json", state)
    command = [
        sys.executable, str(ROOT / "scripts/resource_guard.py"), "--output", str(guard), "--seconds", str(seconds),
        "--", sys.executable, *arguments,
    ]
    with subprocess.Popen(command, cwd=ROOT) as worker:
        try:
            exit_code = worker.wait()
        except CampaignStopped as exc:
            worker.terminate()
            try:
                worker.wait(timeout=18)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            return {**current, "outcome": "interrupted", "reason": str(exc), "ended": now()}
    terminal = guard / "result.json"
    result = read_json(terminal) if terminal.exists() else {"outcome": "guard_result_missing"}
    if result["outcome"] == "completed" and not (directory / "result.json").is_file():
        result["outcome"] = "result_missing"
    return {**current, **result, "returncode": exit_code, "ended": now()}


def evidence(output, specification, phases):
    result = {"specification": specification, "phases": {}}
    for name, entry in phases.items():
        component = {**entry}
        directory = output / name
        path = directory / "result.json"
        if path.exists():
            row = read_json(path)
            component["result_sha256"] = digest(path)
            component["result"] = row
            if name == "high_stack" and row["checkpoint_sha256"] is not None:
                if digest(directory / "gate.pt") != row["checkpoint_sha256"]:
                    raise RuntimeError("High-stack checkpoint differs")
            if name == "replay_100k":
                if row["protocol"]["checkpoint_sha256"] != specification["frozen_eval_gate_sha256"]:
                    raise RuntimeError("100K replay used a different frozen gate")
                pairs = row["pairs"]
                totals = {metric: sum(pair[metric] for pair in pairs) for metric in ("student_score", "gate_score")}
                component["score_totals"] = totals
                component["score_ratio"] = totals["gate_score"] / totals["student_score"]
                component["caps_reached"] = {
                    role: sum(pair[role]["pieces"] == row["protocol"]["cap"] and not pair[role]["game_over"]
                              for pair in pairs)
                    for role in ("student", "gate")
                }
        else:
            component["finished_shards"] = sorted(
                p.name for p in (directory / "capture").glob("seed_*.json")
            ) if name == "high_stack" else sorted(p.name for p in directory.glob("seed_*.json"))
        result["phases"][name] = component
    return result


def publish(output, branch, base_commit, report_path=REPORT,
            message="Archive the long Lookahead training and replay"):
    if git("branch", "--show-current") != branch or git("rev-parse", "HEAD") != base_commit:
        raise RuntimeError("Checkout advanced during the campaign; the report is saved locally")
    operations = ROOT / "OPERATIONS.md"
    base_operations = subprocess.check_output(["git", "show", "HEAD:OPERATIONS.md"], cwd=ROOT)
    current = operations.read_bytes()
    paths = [str(report_path)]
    if current.startswith(base_operations):
        appended = [line for line in current[len(base_operations):].decode().splitlines() if line.strip()]
        if appended and all(str(output) in line for line in appended):
            paths.append("OPERATIONS.md")
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "--only", "-m", message, "--", *paths],
                   cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], cwd=ROOT, check=True, timeout=120)
    return git("rev-parse", "HEAD")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Worktree containing the frozen teacher")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, output = args.checkout.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    branch = git("branch", "--show-current")
    if not branch:
        raise RuntimeError("The campaign requires a named branch for archiving")
    base_commit = git("rev-parse", "HEAD")
    with (output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = output / "status.json"
        report = ROOT / REPORT
        if git("status", "--porcelain", "--", str(REPORT)):
            raise RuntimeError("Campaign report already has local changes")
        previous = read_json(status_path) if status_path.is_file() else {}
        if previous.get("status") == "completed" and previous.get("pushed_commit"):
            print("campaign already archived", previous["pushed_commit"], flush=True)
            return 0
        specification = {
            "high_stack_cap": 8192, "train_seeds": TRAIN_SEEDS, "validation_seeds": VALIDATION_SEEDS,
            "replay_cap": 100000, "replay_seeds": REPLAY_SEEDS,
            "frozen_eval_gate_sha256": digest(FIT_SOURCE / "gate.pt"), "branch": branch,
        }
        plan = output / "protocol.json"
        if plan.exists() and read_json(plan) != specification:
            raise RuntimeError("Existing campaign protocol differs")
        if not plan.exists():
            write_json(plan, specification)
        state = {"status": "running", "started": now(), "phase": "starting", "phases": previous.get("phases", {})}
        write_json(status_path, state)
        operation(output, "后台任务启动，先扩充高堆叠训练，再进行独立 100K 配对评测")
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, stop)

        try:
            high = run_phase("high_stack", 3600, [
                "scripts/train_lookahead_high_stack_gate.py", str(checkout), "--output", str(output / "high_stack"),
                "--cap", str(specification["high_stack_cap"]),
                "--train-seeds", *map(str, TRAIN_SEEDS), "--validation-seeds", *map(str, VALIDATION_SEEDS),
            ], output, state)
            state["phases"]["high_stack"] = high
            if high["outcome"] not in ("startup_failed", "interrupted") and not high["outcome"].startswith("stop:"):
                state["phases"]["replay_100k"] = run_phase("replay_100k", 10800, [
                    "scripts/evaluate_lookahead_gate.py", str(checkout), "--fit", str(FIT_SOURCE),
                    "--output", str(output / "replay_100k"), "--cap", str(specification["replay_cap"]),
                    "--seeds", *map(str, REPLAY_SEEDS),
                ], output, state)

            complete = len(state["phases"]) == 2 and all(
                phase["outcome"] == "completed" for phase in state["phases"].values()
            )
            state["status"] = "completed" if complete else "partial"
            state["phase"] = "archiving"
            state["ended"] = now()
            write_json(status_path, state)
            record = evidence(output, specification, state["phases"])
            write_json(output / "archive.json", record)
            if (git("branch", "--show-current") != branch or git("rev-parse", "HEAD") != base_commit
                    or git("status", "--porcelain", "--", str(REPORT))):
                raise RuntimeError("Checkout or report changed; archive.json is saved in the run directory")
            write_json(report, record)
            operation(output, f"后台任务收尾，阶段状态 { {name: p['outcome'] for name, p in state['phases'].items()} }")
            state["pushed_commit"] = publish(output, branch, base_commit)
            state["phase"] = "finished"
            write_json(status_path, state)
            print("campaign", state["status"], state["pushed_commit"], flush=True)
            return 0 if complete else 1
        except Exception as exc:
            state.update(status="failed", ended=now(), error=repr(exc))
            write_json(status_path, state)
            operation(output, f"后台任务中断，进度已保存，原因 {exc!r}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
