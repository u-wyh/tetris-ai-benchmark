"""Frozen, resumable 20M Candidate selection and independent V2 comparison."""

import argparse
import fcntl
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from sb3_contrib import MaskablePPO

from training.evaluation.champion_pipeline import (
    bootstrap, evaluate_seed, paired, summarize, validate_row,
)
from training.evaluation.traditional_v2 import code_digest, run as run_v2
from training.long_run import SEEDS_FILE, validate_evaluation
from training.train_ppo import ROOT, atomic_json, now
from training.train_transaction import fsync_dir
from training.train_vector_transaction import sha256
from training.traditional.v2 import V2Config, VERSION

RUN = ROOT / "runs/ppo_candidate_champion_20m_selection"
OLD = ROOT / "runs/ppo_candidate_champion"
SOURCE = ROOT / "runs/ppo_candidate_raw_20m_seed42"
OLD_TRAD = ROOT / "runs/ppo_candidate_vs_v2_hold_final100/traditional"
REFERENCE = ROOT / "reports/traditional_longlife/validation10_100000_20261008/hold/config.json"
STEPS = (7_000_064, 19_001_344, 20_000_768)
STAGES = ("finish_20m_validation", "validate_7m_19m_20m_32seeds",
          "paired_comparison", "freeze_new_champion", "independent_100seed_test",
          "traditional_v2_paired_test", "generate_reports", "completed")
REPORT_DIR = ROOT / "reports/experiments"


def frozen_protocol():
    original = json.loads(SEEDS_FILE.read_text())
    assert original["validation"] == list(range(100000, 100032))
    candidate = list(range(400000, 400100))
    if set(candidate) & (set(original["validation"]) | set(original["final_test"])
                         | set(range(42, 50))):
        raise RuntimeError("Independent seeds overlap known fixed sets")
    # Scan seed identities in existing artifacts, not arbitrary score/action numbers.
    for base in (ROOT / "runs", ROOT / "reports"):
        if any(not path.is_relative_to(RUN)
               for path in base.rglob("seed_4000[0-9][0-9].json")):
            raise RuntimeError("Independent seed already has a result")
    return {"version": 1, "validation_seeds": original["validation"],
            "validation_max_pieces": 20000, "test_seeds": candidate,
            "test_max_pieces": 50000, "deterministic": True, "action_mask": True,
            "selection_rule": {"primary": "highest 32-seed mean capped survival",
                               "near_threshold": 1000,
                               "near_tiebreak": ["cap_count", "p10", "paired_wins", "mean"],
                               "bootstrap_samples": 10000, "bootstrap_seed": 20261010,
                               "test_excluded": True},
            "test_seed_note": "300000..300099 were used in earlier traditional pilots; 400000..400099 were selected before evaluation",
            "core": "training.tetris_core.TetrisCore", "reward": "placement-reward-v1",
            "evaluation_code_sha256": sha256(Path(__file__)),
            "seed_file_sha256": sha256(SEEDS_FILE)}


def save_once(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise RuntimeError(f"Frozen artifact differs: {path}")
    else:
        atomic_json(path, data)


def status(stage, state="running", **extra):
    atomic_json(RUN / "status.json", {"stage": stage, "state": state,
                                     "updated_at": now(), **extra})


def log(message):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "events.log").open("a") as f:
        f.write(f"{now()} {message}\n")
        f.flush()
        os.fsync(f.fileno())
    print(message, flush=True)


def sources():
    old = json.loads((OLD / "source_checkpoints.json").read_text())[str(STEPS[0])]
    checks = {str(STEPS[0]): old}
    config = json.loads((SOURCE / "config.json").read_text())
    state = json.loads((SOURCE / "transaction_state.json").read_text())
    if (state["committed_steps"] != STEPS[-1] or state["status"] != "completed"
            or config["reward_version"] != "placement-reward-v1"
            or config["policy"] != "CandidateScoringPolicy"):
        raise RuntimeError("20M source training is not fully committed or changed")
    prior = json.loads((ROOT / "runs/ppo_candidate_raw_10m_seed42/config.json").read_text())
    for key in ("reward_version", "observation_version", "candidate_feature_version",
                "action_space_version", "policy", "network", "ppo", "run_seed", "worker_seeds"):
        if config[key] != prior[key]:
            raise RuntimeError(f"Continuation changed {key}")
    for step in STEPS[1:]:
        folder = SOURCE / "milestones" / f"step_{step:09d}"
        meta = json.loads((folder / "metadata.json").read_text())
        digest = json.loads((folder / "checksum.json").read_text())["model_sha256"]
        if (meta["actual_committed_steps"] != step
                or meta["task_id"] * config["task_steps"] != step
                or sha256(folder / "model.zip") != digest):
            raise RuntimeError(f"Invalid milestone {step}")
        checks[str(step)] = {"model": str((folder / "model.zip").resolve()),
                             "sha256": digest, "committed_steps": step,
                             "git_commit": meta["git_commit"],
                             **{k: config[k] for k in ("reward_version", "observation_version",
                                "candidate_feature_version", "action_space_version", "policy")}}
    if sha256(Path(old["model"])) != old["sha256"]:
        raise RuntimeError("Old Champion source changed")
    return checks


def wait_for_20m():
    path = SOURCE / "evaluations/validation/step_020000768_longlife.json"
    while True:
        state = json.loads((SOURCE / "transaction_state.json").read_text())
        pid = state.get("pid")
        active = False
        if pid and Path(f"/proc/{pid}/cmdline").exists():
            active = str(SOURCE).encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
        if path.exists() and not active:
            break
        if not active:
            log("20M post-commit validation stopped; resuming trainer from committed task")
            subprocess.run([sys.executable, "-m", "training.train_vector_transaction",
                            "--run-dir", str(SOURCE), "--resume"], cwd=ROOT, check=True)
        else:
            time.sleep(20)
    result = json.loads(path.read_text())
    validate_evaluation(result, STEPS[-1], "longlife", 16, 20000)
    source = sources()[str(STEPS[-1])]
    if (sha256(Path(result["model_checkpoint"])) != source["sha256"]
            or sha256(Path(source["model"])) != source["sha256"]):
        raise RuntimeError("20M model changed during validation")
    log("20M longlife validation complete")


def persisted_rows(label, source, seeds, cap):
    folder = RUN / "per_seed" / label
    folder.mkdir(parents=True, exist_ok=True)
    model_path = Path(source["model"])
    if sha256(model_path) != source["sha256"]:
        raise RuntimeError("Checkpoint digest changed")
    model = None
    rows = []
    for seed in seeds:
        path = folder / f"seed_{seed}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if any(record.get(k) != v for k, v in {
                "seed": seed, "model_sha256": source["sha256"],
                "committed_steps": source["committed_steps"], "max_pieces": cap,
                "deterministic": True, "action_mask": True}.items()):
                raise RuntimeError(f"Saved seed protocol differs: {path}")
            row = record["result"]
        else:
            if model is None:
                import torch
                torch.set_num_threads(1)
                model = MaskablePPO.load(str(model_path), device="cpu")
                if model.num_timesteps != source["committed_steps"]:
                    raise RuntimeError("Model timesteps disagree with checkpoint")
            row = evaluate_seed(model, seed, cap)
            validate_row(row, seed, cap)
            atomic_json(path, {"seed": seed, "model_sha256": source["sha256"],
                               "committed_steps": source["committed_steps"],
                               "max_pieces": cap, "deterministic": True,
                               "action_mask": True, "result": row})
            log(f"{label} seed={seed} pieces={row['pieces_survived']} seconds={row['wall_seconds']:.1f}")
        validate_row(row, seed, cap)
        rows.append(row)
    return rows


def validation(checks, proto):
    old_record = json.loads((OLD / "validation_7000064.json").read_text())
    if (old_record["seeds"] != proto["validation_seeds"]
            or old_record["max_pieces"] != 20000
            or old_record["checkpoint"]["sha256"] != checks[str(STEPS[0])]["sha256"]
            or len(old_record["per_seed"]) != 32):
        raise RuntimeError("Old 7M validation does not match frozen protocol")
    for step in STEPS:
        source = checks[str(step)]
        old_rows = {r["seed"]: r for r in old_record["per_seed"]} if step == STEPS[0] else {}
        folder = RUN / "per_seed" / f"validation_{step}"
        folder.mkdir(parents=True, exist_ok=True)
        if old_rows:
            for seed in proto["validation_seeds"]:
                row = old_rows[seed]
                validate_row(row, seed, 20000)
                save_once(folder / f"seed_{seed}.json",
                          {"seed": seed, "model_sha256": source["sha256"],
                           "committed_steps": step, "max_pieces": 20000,
                           "deterministic": True, "action_mask": True, "result": row})
        rows = persisted_rows(f"validation_{step}", source, proto["validation_seeds"], 20000)
        # Existing 16-seed longlife files omit diagnostics. Verify every replayed
        # survival/line/score result against that independent historical record.
        if step != STEPS[0]:
            original = json.loads((SOURCE / "evaluations/validation"
                                   / f"step_{step:09d}_longlife.json").read_text())
            validate_evaluation(original, step, "longlife", 16, 20000)
            for a, b in zip(rows[:16], original["per_seed"]):
                if any(a[k] != b[k] for k in ("seed", "pieces_survived", "lines",
                                                 "score", "game_over", "survived_cap")):
                    raise RuntimeError(f"Historical replay mismatch step={step} seed={a['seed']}")
        save_once(RUN / f"validation_{step}.json",
                  {"checkpoint": source, "seeds": proto["validation_seeds"],
                   "max_pieces": 20000, "per_seed": rows,
                   "aggregate": summarize(rows, 20000)})
        log(f"validation step={step} complete 32/32")


def select(checks, proto):
    results = {step: json.loads((RUN / f"validation_{step}.json").read_text()) for step in STEPS}
    best = max(STEPS, key=lambda s: results[s]["aggregate"]["mean_pieces"])
    pairs = {f"{a}_minus_{b}": paired(results[a]["per_seed"], results[b]["per_seed"])
             for a in STEPS for b in STEPS if a > b}
    near = [best]
    for step in STEPS:
        if step != best:
            p = paired(results[best]["per_seed"], results[step]["per_seed"])
            if (results[best]["aggregate"]["mean_pieces"] - results[step]["aggregate"]["mean_pieces"] <= 1000
                    and p["bootstrap_95ci"][0] <= 0 <= p["bootstrap_95ci"][1]):
                near.append(step)
    def key(step):
        agg = results[step]["aggregate"]
        wins = sum(paired(results[step]["per_seed"], results[other]["per_seed"])["wins"]
                   for other in near if other != step)
        return (agg["capped_count"], agg["p10_pieces"], wins, agg["mean_pieces"])
    selected = best if len(near) == 1 else max(near, key=key)
    decision = {"selected_step": selected, "highest_mean_step": best,
                "near_candidates": near, "selection_uncertain": len(near) > 1,
                "paired": pairs, "selection_rule": proto["selection_rule"],
                "selected_before_independent_test": now()}
    decision = json.loads(json.dumps(decision))
    path = RUN / "selection_decision.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous.get(key) != decision[key] for key in decision if key != "selected_before_independent_test"):
            raise RuntimeError("Frozen selection differs")
        return previous
    atomic_json(path, decision)
    return decision


def freeze(checks):
    decision = json.loads((RUN / "selection_decision.json").read_text())
    source = checks[str(decision["selected_step"])]
    target = RUN / "model.zip"
    if not target.exists():
        temporary = RUN / "model.zip.tmp"
        shutil.copy2(source["model"], temporary)
        if sha256(temporary) != source["sha256"]:
            raise RuntimeError("Frozen model copy mismatch")
        os.replace(temporary, target)
        fsync_dir(RUN)
    if sha256(target) != source["sha256"]:
        raise RuntimeError("Frozen Champion changed")
    path = RUN / "champion.json"
    if not path.exists():
        atomic_json(path, {"source": source, "selected_step": decision["selected_step"],
                           "model_sha256": source["sha256"], "frozen_at": now(),
                           "selection_decision": "selection_decision.json"})


def independent_test(proto):
    champion = json.loads((RUN / "champion.json").read_text())
    source = dict(champion["source"], model=str(RUN / "model.zip"))
    rows = persisted_rows("independent_test_v2", source, proto["test_seeds"], 50000)
    save_once(RUN / "independent_test_v2.json",
              {"protocol": "candidate_test_v2", "checkpoint": champion,
               "seeds": proto["test_seeds"], "max_pieces": 50000,
               "per_seed": rows, "aggregate": summarize(rows, 50000)})


def traditional_test(proto):
    historical = json.loads(REFERENCE.read_text())
    if (historical["strategy"] != VERSION or historical["policy"]["mode"] != "hold"
            or historical["code_sha256"] != code_digest()):
        raise RuntimeError("V2 implementation changed from validated reference")
    policy = V2Config(**historical["policy"])
    if asdict(policy) != historical["policy"]:
        raise RuntimeError("V2 policy parameters changed")
    folder = RUN / "traditional_v2"
    run_v2(folder, proto["test_seeds"], 50000, policy,
           resume=(folder / "config.json").exists())


def render_reports(proto):
    decision = json.loads((RUN / "selection_decision.json").read_text())
    vals = {s: json.loads((RUN / f"validation_{s}.json").read_text()) for s in STEPS}
    selected = decision["selected_step"]
    lines = ["# Candidate PPO 20M 阶段 checkpoint 选择", "",
             "选择先于独立测试冻结。三模型统一使用 validation seeds 100000–100031，20,000 块上限，确定性推理和 Action Mask；上限为右截断。", "",
             "| Steps | 平均截断存活 | 中位 | P10 | P90 | 截断 | Game Over | 消行均值 | 分数均值 | 新洞/100块 | 死亡高度 | 死亡洞数 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in STEPS:
        a = vals[s]["aggregate"]
        lines.append(f"| {s:,} | {a['mean_pieces']:.1f} | {a['median_pieces']:.1f} | {a['p10_pieces']} | {a['p90_pieces']} | {a['capped_count']}/32 | {32-a['capped_count']}/32 | {a['mean_lines']:.1f} | {a['mean_score']:.1f} | {a['new_holes_per_100_pieces']:.2f} | {a['mean_death_height']} | {a['mean_death_holes']} |")
    lines += ["", f"按预定规则选中 **{selected:,} steps**；选择不确定性：{decision['selection_uncertain']}。平均值、截断率、低分位并列评估，不能把截断当真实死亡。", "",
              "| 配对差 | 平均块数 | 95% bootstrap 区间 | 胜/平/负 |", "|---|---:|---|---:|"]
    for name, p in decision["paired"].items():
        lines.append(f"| {name} | {p['mean_difference']:.1f} | {p['bootstrap_95ci']} | {p['wins']}/{p['ties']}/{p['losses']} |")
    lines += ["", "逐 seed 截断存活差（前者减后者）：", "",
              "| Seed | 7M | 19M | 20M | 19M−7M | 19M−20M |", "|---:|---:|---:|---:|---:|---:|"]
    for i, seed in enumerate(proto["validation_seeds"]):
        a,b,c = [vals[s]["per_seed"][i]["pieces_survived"] for s in STEPS]
        lines.append(f"| {seed} | {a} | {b} | {c} | {b-a:+} | {b-c:+} |")
    lines += ["", "19M/20M 原有 16-seed 长局结果仅用于一致性重放核验；完整诊断与后 16 seeds 均由逐局原子评测补齐。7M 复用已校验的原 32-seed 结果。", ""]
    (REPORT_DIR / "ppo_candidate_20m_checkpoint_selection.md").write_text("\n".join(lines))

    test = json.loads((RUN / "independent_test_v2.json").read_text())
    test_aggregate = test["aggregate"]
    lines = ["# Candidate Champion v2 独立 100-seed 测试", "",
             f"冻结 checkpoint：{selected:,} steps，SHA256 `{test['checkpoint']['model_sha256']}`。种子 400000–400099 在本轮选模前固定；300000–300099 已用于早期传统开发集，故未使用。确定性推理、Action Mask、每局 50,000 块上限。", "",
             f"平均截断存活 {test_aggregate['mean_pieces']:.1f}（95% bootstrap 区间 {test_aggregate['mean_survival_bootstrap_95ci']}），中位 {test_aggregate['median_pieces']:.1f}，P10/P90 {test_aggregate['p10_pieces']}/{test_aggregate['p90_pieces']}。", "",
             f"平均消行 {test_aggregate['mean_lines']:.1f}，平均正式分数 {test_aggregate['mean_score']:.1f}；截断 {test_aggregate['capped_count']}/100，Game Over {100-test_aggregate['capped_count']}/100。50,000 块截断不是真实死亡寿命。", "",
             "| Seed | 存活 | 消行 | 分数 | 结局 |", "|---:|---:|---:|---:|---|"]
    for r in test["per_seed"]:
        lines.append(f"| {r['seed']} | {r['pieces_survived']} | {r['lines']} | {r['score']} | {'截断' if r['survived_cap'] else 'Game Over'} |")
    (REPORT_DIR / "ppo_candidate_champion_v2_final.md").write_text("\n".join(lines)+"\n")

    trad = [json.loads((RUN / "traditional_v2" / f"seed_{s}.json").read_text()) for s in proto["test_seeds"]]
    candidate = test["per_seed"]
    differences = [a["pieces_survived"]-b["pieces_survived"] for a,b in zip(candidate,trad)]
    total_wall = sum(r["wall_seconds"] for r in trad)
    lines = ["# Candidate Champion v2 与传统 V2 Hold-only 配对测试", "",
             "同一 400000–400099 seeds、同一 Python TetrisCore/7-Bag/Hold/Next/合法落点/正式计分，每局上限 50,000 块。两者均在测试前冻结；截断不是死亡。", "",
             f"Candidate 平均截断存活 {statistics.mean(r['pieces_survived'] for r in candidate):.1f}，V2 {statistics.mean(r['pieces_survived'] for r in trad):.1f}；差值 {statistics.mean(differences):.1f}，配对 bootstrap 95% 区间 {bootstrap(differences)}。", "",
             f"逐 seed Candidate 胜/平/负：{sum(x>0 for x in differences)}/{sum(x==0 for x in differences)}/{sum(x<0 for x in differences)}；到达上限 {sum(r['survived_cap'] for r in candidate)}/100 vs {sum(r['truncated'] for r in trad)}/100。", "",
             f"V2 总运行 {total_wall/3600:.2f} 小时，吞吐 {sum(r['pieces_survived'] for r in trad)/total_wall:.1f} 块/秒。旧 200000–200099 对照只完成 29/100 局，未混入本次结果。", "",
             "| Seed | Candidate 块数 | V2 块数 | Candidate 分数 | V2 分数 | 差值 |", "|---:|---:|---:|---:|---:|---:|"]
    for a,b,d in zip(candidate,trad,differences):
        lines.append(f"| {a['seed']} | {a['pieces_survived']} | {b['pieces_survived']} | {a['score']} | {b['score']} | {d:+} |")
    (REPORT_DIR / "ppo_candidate_vs_traditional_v2.md").write_text("\n".join(lines)+"\n")
    changelog = ROOT / "CHANGELOG.md"
    text = changelog.read_text()
    entry = (f"- Candidate PPO 20M 独立 32-seed 选模选中 {selected:,} steps；"
             f"新 100-seed 测试平均 {test_aggregate['mean_pieces']:.1f} 块、{test_aggregate['capped_count']}/100 截断；"
             "随后完成 V2 Hold-only 同 seed 配对。详见 reports/experiments/ppo_candidate_20m_checkpoint_selection.md、ppo_candidate_champion_v2_final.md 和 ppo_candidate_vs_traditional_v2.md。\n")
    if entry not in text:
        marker = "## 2026-10-11\n\n"
        if marker not in text:
            raise RuntimeError("CHANGELOG date marker missing")
        changelog.write_text(text.replace(marker, marker+entry, 1))


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proto = frozen_protocol()
        save_once(RUN / "protocol.json", proto)
        checks = sources()
        save_once(RUN / "source_checkpoints.json", checks)
        for stage in STAGES[:-1]:
            status(stage)
            log(f"STAGE_STARTED {stage}")
            try:
                if stage == STAGES[0]: wait_for_20m()
                elif stage == STAGES[1]: validation(checks, proto)
                elif stage == STAGES[2]: select(checks, proto)
                elif stage == STAGES[3]: freeze(checks)
                elif stage == STAGES[4]: independent_test(proto)
                elif stage == STAGES[5]: traditional_test(proto)
                elif stage == STAGES[6]: render_reports(proto)
                log(f"STAGE_COMPLETED {stage}")
            except Exception as error:
                status(stage, "failed", error=repr(error), traceback=traceback.format_exc())
                log(f"STAGE_FAILED {stage} {traceback.format_exc()}")
                raise
        status("completed", "completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        state = json.loads((RUN / "status.json").read_text()) if (RUN / "status.json").exists() else {}
        counts = {str(s): len(list((RUN / "per_seed" / f"validation_{s}").glob("seed_*.json"))) for s in STEPS}
        counts["independent_test_v2"] = len(list((RUN / "per_seed/independent_test_v2").glob("seed_*.json")))
        counts["traditional_v2"] = len(list((RUN / "traditional_v2").glob("seed_*.json")))
        print(json.dumps({"status": state, "completed_games": counts}, ensure_ascii=False, indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
