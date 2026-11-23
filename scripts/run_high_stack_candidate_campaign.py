"""Run and publish a detached evaluation of the trained high-stack gate."""

import argparse
import fcntl
from pathlib import Path
import signal
from statistics import mean

from run_lookahead_long_campaign import (
    ROOT, digest, git, now, operation, publish, read_json, run_phase, stop, write_json,
)


REPORT = Path("reports/experiments/high_stack_candidate_campaign_20261123.json")
FIT = ROOT / "runs/lookahead_long_campaign_20261123/high_stack"


def summarize(rows, mode):
    pairs = [row for row in rows if mode == "natural" or row["found"]]
    if not pairs:
        return {"seeds": len(rows), "compared_pairs": 0}
    roles = ("student", "gate")
    return {
        "seeds": len(rows), "compared_pairs": len(pairs),
        "deaths": {role: sum(row[role]["game_over"] for row in pairs) for role in roles},
        "mean_pieces": {role: mean(row[role]["pieces"] for row in pairs) for role in roles},
        "common_score_totals": {role: sum(row[role]["common_score"] for row in pairs) for role in roles},
        "gate_only_deaths": sum(row["gate"]["game_over"] and not row["student"]["game_over"] for row in pairs),
        "student_only_deaths": sum(row["student"]["game_over"] and not row["gate"]["game_over"] for row in pairs),
        "wins": sum(row["gate"]["common_score"] > row["student"]["common_score"] for row in pairs),
        "ties": sum(row["gate"]["common_score"] == row["student"]["common_score"] for row in pairs),
    }


def collect_evidence(output, specification, phases):
    document = {"specification": specification, "phases": {}}
    for mode, phase in phases.items():
        directory = output / mode
        component = {**phase}
        result_path = directory / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            protocol = result["protocol"]
            if (protocol["mode"], protocol["checkpoint_sha256"]) != (mode, specification["gate_sha256"]):
                raise RuntimeError("Phase mode or checkpoint differs")
            identity = digest(directory / "protocol.json")
            if protocol["seeds"] != [row["seed"] for row in result["rows"]]:
                raise RuntimeError("Evaluation rows do not match their planned seed set")
            for row in result["rows"]:
                saved = read_json(directory / f'seed_{row["seed"]}.json')
                if saved != row or row["protocol_sha256"] != identity:
                    raise RuntimeError(f'Evaluation shard differs for seed {row["seed"]}')
            component.update(result=result, result_sha256=digest(result_path),
                             summary=summarize(result["rows"], mode))
        else:
            component["finished_shards"] = sorted(p.name for p in directory.glob("seed_*.json"))
        document["phases"][mode] = component
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout, output = args.checkout.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    branch, base_commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if not branch or git("status", "--porcelain", "--", str(REPORT)):
        raise RuntimeError("A named branch and a clean report path are required")

    with (output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = output / "status.json"
        previous = read_json(status) if status.exists() else {}
        if previous.get("status") == "completed" and previous.get("pushed_commit"):
            print("campaign already archived", previous["pushed_commit"], flush=True)
            return 0
        specification = {
            "branch": branch, "gate_sha256": digest(FIT / "gate.pt"),
            "fit_result_sha256": digest(FIT / "result.json"),
            "evaluator_sha256": digest(ROOT / "scripts/evaluate_high_stack_candidate.py"),
        }
        protocol = output / "protocol.json"
        if protocol.exists() and read_json(protocol) != specification:
            raise RuntimeError("Existing campaign protocol differs")
        if not protocol.exists():
            write_json(protocol, specification)
        state = {"status": "running", "started": now(), "phase": "starting", "phases": previous.get("phases", {})}
        write_json(status, state)
        operation(output, "无人值守评测启动，先做高堆叠同局面分叉，再做独立自然 100K 对照")
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, stop)

        try:
            for mode, budget in (("stress", 2100), ("natural", 10800)):
                phase = run_phase(mode, budget, [
                    "scripts/evaluate_high_stack_candidate.py", str(checkout), "--fit", str(FIT),
                    "--output", str(output / mode), "--mode", mode,
                ], output, state)
                state["phases"][mode] = phase
                if phase["outcome"] in ("interrupted", "startup_failed") or phase["outcome"].startswith("stop:"):
                    break
            complete = all(state["phases"].get(mode, {}).get("outcome") == "completed"
                           for mode in ("stress", "natural"))
            state.update(status="completed" if complete else "partial", phase="archiving", ended=now())
            write_json(status, state)
            report = collect_evidence(output, specification, state["phases"])
            write_json(output / "archive.json", report)
            if (git("branch", "--show-current") != branch or git("rev-parse", "HEAD") != base_commit
                    or git("status", "--porcelain", "--", str(REPORT))):
                raise RuntimeError("Checkout or report changed; archive.json is saved in the run directory")
            write_json(ROOT / REPORT, report)
            operation(output, f"无人值守评测收尾，阶段状态 { {mode: phase['outcome'] for mode, phase in state['phases'].items()} }")
            state["pushed_commit"] = publish(output, branch, base_commit, REPORT, "Archive the high-stack gate campaign")
            state["phase"] = "finished"
            write_json(status, state)
            print("campaign", state["status"], state["pushed_commit"], flush=True)
            return 0 if complete else 1
        except Exception as exc:
            state.update(status="failed", ended=now(), error=repr(exc))
            write_json(status, state)
            operation(output, f"无人值守队列中断，已保存阶段记录，原因 {exc!r}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
