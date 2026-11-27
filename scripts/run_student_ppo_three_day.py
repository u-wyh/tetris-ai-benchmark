"""Run and archive matched PPO experiments throughout a bounded unattended window."""

from datetime import datetime, timedelta
import fcntl
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from evaluate_overnight_student_ppo import summarize_pairs
from run_lookahead_long_campaign import ROOT, digest, git, now, operation, publish, read_json, write_json
from run_student_ppo_confirmation import collect_pairs
from supervise_overnight_student_ppo import CONFIG, ask_codex, completed_seconds, training_totals


OUTPUT = ROOT / "runs/student_ppo_three_day_20261127"
REPORT = Path("reports/experiments/student_ppo_three_day_20261127.json")
TZ = ZoneInfo("Asia/Shanghai")
STEPS = 8_388_608
CONTROL = {"beta": 0.3, "reward_profile": "survival_balanced_v1"}
SCREEN = (0.8, 0.8, 0.5, 0.5)
MAX_BLOCKS = 12
PAIR_CAP = 20_000


def clock():
    return datetime.now(TZ)


def report_for(name):
    return REPORT.with_stem(f"{REPORT.stem}_{name}")


def checked_archive(output, report):
    """Recover a report written before a Git or process interruption."""
    path = ROOT / report
    local = output / "archive.json"
    if not path.is_file():
        return None
    if not local.is_file() or read_json(path) != read_json(local):
        raise RuntimeError(f"Existing report does not match its local archive: {report}")
    branch, commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--", str(report)):
        publish(output, branch, commit, report, "Recover interrupted PPO report publication")
    elif read_json(output / "status.json").get("phase") != "finished":
        subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"],
                       cwd=ROOT, check=True, timeout=120)
    return read_json(path)


def train_arm(output, report, train_seed, experiment, deadline):
    output.mkdir(parents=True, exist_ok=True)
    if checked_archive(output, report) is None:
        schedule = output / "schedule.json"
        if schedule.is_file():
            interval = read_json(schedule)
        else:
            started = clock()
            stop = min(started + timedelta(hours=5, minutes=40), deadline)
            interval = {"start": started.isoformat(timespec="seconds"), "end": stop.isoformat(timespec="seconds")}
            write_json(schedule, interval)
        if clock() < datetime.fromisoformat(interval["end"]) - timedelta(minutes=5):
            preset = output / "preset.json"
            config = read_json(CONFIG)
            config.update(base_seed=train_seed, target_total_steps=STEPS)
            if preset.is_file() and read_json(preset) != config:
                raise RuntimeError("Stored matched training preset differs")
            write_json(preset, config)
            launcher = (
                "import sys\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path.cwd() / 'scripts'))\n"
                "import supervise_overnight_student_ppo as ppo\n"
                "ppo.CONFIG = Path(sys.argv.pop(1))\n"
                "ppo.VARIANTS = {'base': {**ppo.VARIANTS['base'], 'beta': float(sys.argv.pop(1))}}\n"
                "ppo.TARGET_TRAINING = 12 * 3600\n"
                "ppo.MINIMUM_TRAINING = 2 * 3600\n"
                "ppo.LIVE_REVIEW_INTERVAL = 30 * 60\n"
                "raise SystemExit(ppo.main())\n"
            )
            result = subprocess.run([
                sys.executable, "-c", launcher, str(preset), str(experiment["beta"]),
                "--output", str(output), "--start", interval["start"], "--end", interval["end"],
                "--reward-profile", experiment["reward_profile"], "--report", str(report),
            ], cwd=ROOT, check=False)
            if not (ROOT / report).is_file() and result.returncode:
                raise RuntimeError(f"Arm exited before archiving: {report} ({result.returncode})")
    archive = checked_archive(output, report)
    totals = training_totals(output)
    committed = totals.get("base", {})
    result = {"experiment": experiment, "seed": train_seed, "steps": committed.get("steps", 0),
              "seconds": completed_seconds(totals), "report": str(report) if archive else None,
              "codex_reviews": archive.get("verified_codex_live_reviews", 0) if archive else 0,
              "codex_decisions": archive.get("verified_codex_decisions", 0) if archive else 0}
    if result["steps"] == STEPS and archive:
        source = output / "training/base/committed" / f"task_{committed['task']:06d}" / "model.zip"
        result["checkpoint"] = {"path": str(source.relative_to(ROOT)), "sha256": digest(source)}
    return result


def replay(output, report, arms, seeds, cap):
    output.mkdir(parents=True, exist_ok=True)
    archived = checked_archive(output, report)
    if archived is not None:
        return archived
    models = {role: arms[role]["checkpoint"] for role in ("reference", "candidate")}
    for model in models.values():
        if digest(ROOT / model["path"]) != model["sha256"]:
            raise RuntimeError("Paired evaluation checkpoint changed")
    protocol = {"models": models, "seeds": seeds, "cap": cap}
    manifest = output / "protocol.json"
    if manifest.is_file() and read_json(manifest) != protocol:
        raise RuntimeError("Stored replay protocol differs")
    write_json(manifest, protocol)
    state = {"phase": "replaying", "started": now(), "completed_pairs": 0, "total_pairs": len(seeds)}
    write_json(output / "status.json", state)
    branch, commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
    operation(output, f"开始等步数模型配对回放，{len(seeds)} 个独立种子，每局最多 {cap:,} 块")
    pairs = collect_pairs(output, state, protocol)
    summary = summarize_pairs(pairs)
    payload = {"protocol": protocol, "arms": arms, "pairs": pairs, "summary": summary, "ended_at": now()}
    write_json(output / "archive.json", payload)
    write_json(ROOT / report, payload)
    operation(output, f"配对回放完成，候选与对照提前结束 {summary['candidate_deaths']}/{summary['reference_deaths']} 局，等长得分比 {summary['common_score_ratio']}")
    pushed = publish(output, branch, commit, report, "Archive a matched PPO survival benchmark")
    state.update(phase="finished", pushed_commit=pushed)
    write_json(output / "status.json", state)
    return payload


def choose_confirmation(blocks, directory):
    path = directory / "choice.json"
    if path.is_file():
        return read_json(path)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for beta in sorted(set(SCREEN)):
        sample = [pair for block in blocks if block["experiment"]["beta"] == beta for pair in block.get("pairs", [])]
        if len(sample) == 16:
            metrics[str(beta)] = summarize_pairs(sample)
    eligible = [beta for beta, result in metrics.items()
                if result["candidate_deaths"] <= result["reference_deaths"]
                and result["candidate_pieces"] >= result["reference_pieces"]
                and (result["common_score_ratio"] or 0) >= 1.0]
    actions = [f"confirm_{beta}" for beta in eligible] or ["reward_control"]
    choice = ask_codex(directory, {"screening": metrics, "eligible": eligible}, actions, (
        "Decide the next matched Tetris PPO experiment using the independent screening results. "
        "Prefer survival reproducibility and equal-length score. reward_control compares standard reward with "
        "the balanced reward at the same KL when neither KL candidate qualifies. Use only the stated evidence. "
        "Give a concise explanation in Chinese. Do not call tools."
    ))
    if choice["source"] != "local_codex":
        action = "reward_control"
        if eligible:
            beta = min(eligible, key=lambda b: (metrics[b]["candidate_deaths"], -metrics[b]["common_score_ratio"], float(b)))
            action = f"confirm_{beta}"
        choice = {"source": "fixed_protocol", "action": action, "reason": choice["reason"]}
    choice["experiment"] = ({"beta": float(choice["action"].split("_")[1]), "reward_profile": CONTROL["reward_profile"]}
                            if choice["action"].startswith("confirm_") else {"beta": 0.3, "reward_profile": "default"})
    choice["screening"] = metrics
    write_json(path, choice)
    return choice


def main():
    status = OUTPUT / "status.json"
    window = read_json(OUTPUT / "window.json")
    deadline = datetime.fromisoformat(window["deadline"])
    with (OUTPUT / "workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.is_file() and read_json(status).get("phase") == "finished":
            return
        archived = checked_archive(OUTPUT, REPORT)
        if archived is not None:
            archived.update(phase="finished", pushed_commit=git("log", "-1", "--format=%H", "--", str(REPORT)))
            write_json(status, archived)
            return
        previous = read_json(status) if status.is_file() else {}
        state = {"phase": "running", "started_at": previous.get("started_at", now()), "deadline": window["deadline"],
                 "target_steps_per_arm": STEPS, "blocks": previous.get("blocks", [])}
        write_json(status, state)
        try:
            for block_number in range(len(state["blocks"]) + 1, MAX_BLOCKS + 1):
                folder = OUTPUT / f"block_{block_number:02d}"
                if not folder.is_dir() and deadline - clock() < timedelta(hours=8, minutes=40):
                    break
                folder.mkdir(exist_ok=True)
                if block_number <= len(SCREEN):
                    experiment = {**CONTROL, "beta": SCREEN[block_number - 1]}
                else:
                    decision = choose_confirmation(state["blocks"][:len(SCREEN)], OUTPUT / "selection")
                    state["confirmation_choice"] = decision
                    experiment = decision["experiment"]
                state.update(active_block=block_number, active_phase="training")
                arms = {}
                order = ("reference", "candidate") if block_number % 2 else ("candidate", "reference")
                for role in order:
                    state.update(active_arm=role, arm_output=str((folder / role).relative_to(ROOT)))
                    write_json(status, state)
                    arms[role] = train_arm(folder / role, report_for(f"b{block_number:02d}_{role}"),
                                           3276110000 + 100 * block_number,
                                           CONTROL if role == "reference" else experiment,
                                           deadline - timedelta(minutes=70))
                row = {"block": block_number, "experiment": experiment, "arms": arms}
                if all("checkpoint" in arm for arm in arms.values()) and deadline - clock() >= timedelta(minutes=70):
                    state.update(active_phase="replay", active_arm=None)
                    write_json(status, state)
                    report = report_for(f"b{block_number:02d}_paired")
                    seeds = list(range(3277110000 + block_number * 100, 3277110008 + block_number * 100))
                    result = replay(folder / "replay", report, arms, seeds, PAIR_CAP)
                    row.update(pairs=result["pairs"], summary=result["summary"], report=str(report))
                state["blocks"].append(row)
                write_json(status, state)
            matched = [row for row in state["blocks"] if "summary" in row]
            confirmation = [pair for row in matched if row["block"] > len(SCREEN) for pair in row["pairs"]]
            if confirmation:
                state["confirmation_summary"] = {"pairs": len(confirmation), **summarize_pairs(confirmation)}
            remaining = deadline - clock()
            tail_plan = OUTPUT / "long_replay_plan.json"
            if not tail_plan.is_file() and matched:
                cap = next((limit for limit, hours in ((100_000, 5.5), (50_000, 3), (35_000, 2))
                            if remaining >= timedelta(hours=hours)), None)
                if cap is not None:
                    write_json(tail_plan, {"block": matched[-1]["block"], "cap": cap})
            if tail_plan.is_file() and remaining >= timedelta(minutes=15):
                plan = read_json(tail_plan)
                source = next(row for row in matched if row["block"] == plan["block"])
                state.update(active_phase="long_replay", active_block=source["block"], active_arm=None)
                write_json(status, state)
                report = report_for("long_replay")
                seeds = list(range(3277113000, 3277113008))
                result = replay(OUTPUT / "long_replay", report, source["arms"], seeds, plan["cap"])
                state["long_replay"] = {**plan, "report": str(report), "summary": result["summary"]}
            state.update(phase="archiving", ended_at=now(), active_block=None, active_arm=None, active_phase=None)
            write_json(status, state)
            report = ROOT / REPORT
            branch, commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
            write_json(OUTPUT / "archive.json", state)
            write_json(report, state)
            operation(OUTPUT, f"三天实验收尾，完成 {len(state['blocks'])} 个计划分组，其中 {sum('summary' in row for row in state['blocks'])} 个有等步数配对结果")
            pushed = publish(OUTPUT, branch, commit, REPORT, "Archive the three-day PPO experiment campaign")
            state.update(phase="finished", pushed_commit=pushed)
            write_json(status, state)
        except Exception as exc:
            state.update(phase="retrying", error=repr(exc), updated_at=now())
            write_json(status, state)
            operation(OUTPUT, f"三天实验遇到失败，保留事务等待服务重试，原因 {exc!r}")
            raise


if __name__ == "__main__":
    main()
