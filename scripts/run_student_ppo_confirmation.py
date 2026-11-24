"""Check the selected PPO checkpoint on eight independent, longer replays."""

import fcntl
from pathlib import Path
import time

from evaluate_overnight_student_ppo import summarize_pairs
from run_lookahead_long_campaign import ROOT, digest, git, now, operation, publish, read_json, write_json
from supervise_overnight_student_ppo import ask_codex, require_runtime, run_guarded


OUTPUT = ROOT / "runs/student_ppo_confirmation_20261124"
SOURCE = Path("reports/experiments/student_ppo_height_live_20261124.json")
REPORT = Path("reports/experiments/student_ppo_confirmation_20261124.json")
SEEDS = list(range(3268114000, 3268114008))
CAP = 20_000
MIN_SCORE_RATIO = 1.15
MIN_WINS = 7


def specification():
    archive = read_json(ROOT / SOURCE)
    selected = min(
        (stage for stage in archive["stages"] if stage["evaluation"]["candidate_deaths"] == 0),
        key=lambda stage: stage["evaluation"]["candidate_high_fraction"],
    )
    candidate = ROOT / Path(selected["checkpoint"]["path"]).relative_to(ROOT)
    reference = ROOT / read_json(ROOT / "configs/overnight_student_ppo_20261124.json")["initial_model"]["path"]
    models = {
        "reference": {"path": str(reference.relative_to(ROOT)), "sha256": archive["protocol"]["source_sha256"]},
        "candidate": {"path": str(candidate.relative_to(ROOT)), "sha256": selected["checkpoint"]["sha256"]},
    }
    for model in models.values():
        if digest(ROOT / model["path"]) != model["sha256"]:
            raise RuntimeError(f"Checkpoint digest differs: {model['path']}")
    if set(SEEDS).intersection(archive["protocol"]["development_seeds"]):
        raise RuntimeError("Confirmation seeds overlap with development seeds")
    return {
        "source_report": str(SOURCE), "source_report_sha256": digest(ROOT / SOURCE),
        "stage": selected["stage"], "candidate_steps": selected["committed"]["steps"],
        "selection": "Lowest development high-stack fraction among stages with no early endings",
        "models": models, "seeds": SEEDS, "cap": CAP,
        "eligibility": {
            "minimum_common_score_ratio": MIN_SCORE_RATIO, "minimum_score_wins": MIN_WINS,
            "maximum_excess_deaths": 0, "minimum_pieces_ratio": 1.0,
        },
    }


def collect_pairs(state, protocol):
    pairs = []
    for seed in SEEDS:
        directory = OUTPUT / f"seed_{seed}"
        directory.mkdir(exist_ok=True)
        result = directory / "pair.json"
        state.update(active_seed=seed, completed_pairs=len(pairs), updated_at=now())
        if not result.is_file():
            for _ in range(2):
                outcome = run_guarded(OUTPUT, directory / f"guard_{time.time_ns()}", 220, [
                    "-m", "scripts.evaluate_overnight_student_ppo",
                    "--reference", protocol["models"]["reference"]["path"],
                    "--candidate", protocol["models"]["candidate"]["path"],
                    "--output", str(result), "--seeds", str(seed), "--cap", str(CAP),
                ], state)
                if outcome["outcome"] == "completed" and result.is_file():
                    break
                if outcome["outcome"].startswith("stop:"):
                    raise RuntimeError(f"Resource guard stopped seed {seed}: {outcome['outcome']}")
                if result.is_file() or outcome["outcome"] == "interrupted":
                    raise RuntimeError(f"Seed {seed} did not finish cleanly: {outcome['outcome']}")
            else:
                raise RuntimeError(f"Both replay attempts failed for seed {seed}")
        replay = read_json(result)
        if replay["cap"] != CAP or [pair["seed"] for pair in replay["pairs"]] != [seed]:
            raise RuntimeError(f"Stored replay does not match seed {seed}")
        pairs.append(replay["pairs"][0])
        state.update(completed_pairs=len(pairs), updated_at=now())
        write_json(OUTPUT / "status.json", state)
    return pairs


def main():
    require_runtime()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status = OUTPUT / "status.json"
    with (OUTPUT / "campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        branch, commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
        if not branch or (ROOT / REPORT).exists():
            raise RuntimeError("A named branch and a new report path are required")
        protocol = specification()
        manifest = OUTPUT / "protocol.json"
        if manifest.is_file() and read_json(manifest) != protocol:
            raise RuntimeError("Stored confirmation protocol differs")
        write_json(manifest, protocol)
        state = {"phase": "replaying", "started_at": now(), "completed_pairs": 0, "total_pairs": len(SEEDS)}
        write_json(status, state)
        operation(OUTPUT, f"独立回放启动，冻结第 {protocol['stage']} 段检查点，与冻结学生比较 8 个新种子，每局最多 20,000 块")
        try:
            pairs = collect_pairs(state, protocol)
            summary = summarize_pairs(pairs)
            wins = sum(pair["candidate"]["common_score"] > pair["reference"]["common_score"] for pair in pairs)
            survival_ok = (
                summary["candidate_deaths"] <= summary["reference_deaths"]
                and summary["candidate_pieces"] >= summary["reference_pieces"]
            )
            eligible = survival_ok and (summary["common_score_ratio"] or 0) >= MIN_SCORE_RATIO and wins >= MIN_WINS
            report = {"protocol": protocol, "summary": summary, "score_wins": wins,
                      "eligible_to_expand": eligible, "pairs": pairs}
            actions = ["hold_candidate"]
            if eligible:
                actions.append("expand_evaluation")
            elif not survival_ok:
                actions.append("focus_survival")
            state.update(phase="codex_decision", active_seed=None, summary=summary)
            write_json(status, state)
            judge = OUTPUT / "judge"
            judge.mkdir(exist_ok=True)
            result = ask_codex(judge, {**report, "allowed_actions": actions}, actions, (
                "Review this independent Tetris replay. Use the supplied paired results and the predeclared "
                "survival and score criteria. Choose an allowed action for the next experiment. "
                "Use focus_survival for repeatable premature losses or hold_candidate if they are inconclusive. "
                "Explain in Chinese how early endings and high stacks qualify any equal-length score gain. "
                "Expansion means another independent evaluation, and deployment is outside the experiment."
            ))
            report["codex_decision"] = result if result["source"] == "local_codex" else None
            if result["source"] != "local_codex":
                report["codex_error"] = result["reason"]
            state.update(phase="archiving", codex_decision=report["codex_decision"])
            write_json(status, state)
            write_json(OUTPUT / "archive.json", report)
            if git("rev-parse", "HEAD") != commit or git("branch", "--show-current") != branch or (ROOT / REPORT).exists():
                raise RuntimeError("Checkout or report changed during evaluation; archive.json is preserved")
            write_json(ROOT / REPORT, report)
            decision = result["action"] if report["codex_decision"] else "未返回可用决策"
            operation(OUTPUT, f"独立回放完成，等长得分比 {summary['common_score_ratio']}，候选提前结束 {summary['candidate_deaths']}/8 局，Codex 后续决策 {decision}")
            pushed = publish(OUTPUT, branch, commit, REPORT, "Archive independent PPO checkpoint confirmation")
            state.update(phase="finished", pushed_commit=pushed, ended_at=now())
            write_json(status, state)
        except Exception as exc:
            state.update(phase="failed", error=repr(exc), ended_at=now())
            write_json(status, state)
            operation(OUTPUT, f"独立回放中断，已完成配对保留，原因 {exc!r}")
            raise


if __name__ == "__main__":
    main()
