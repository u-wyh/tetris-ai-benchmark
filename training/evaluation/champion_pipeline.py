"""Recoverable Candidate checkpoint selection and one-shot formal evaluation."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from training.env.candidate_env import CandidateTetrisEnv
from training.evaluation.evaluate import aggregate
from training.long_run import SEEDS_FILE
from training.train_ppo import ROOT, atomic_json, now
from training.train_transaction import fsync_dir
from training.train_vector_transaction import sha256

SOURCE = ROOT / "runs/ppo_candidate_raw_10m_seed42"
RUN = ROOT / "runs/ppo_candidate_champion"
RAW = ROOT / "runs/ppo_raw_10m_seed42"
SELECTION_REPORT = ROOT / "reports/experiments/ppo_candidate_champion_selection.md"
FINAL_REPORT = ROOT / "reports/experiments/ppo_candidate_final_100seed.md"
STEPS = (7_000_064, 8_003_584, 9_003_008, 10_002_432)
STAGES = ("validate_7m_8m_9m_10m", "compare_checkpoints", "freeze_champion",
          "champion_final_test_100", "compare_raw_final", "traditional_comparison_plan",
          "generate_report", "completed")
ROW_FIELDS = ("seed", "pieces_survived", "lines", "score", "episode_reward",
              "game_over", "survived_cap", "new_holes_events", "new_holes_total", "holes_at_end",
              "height_at_end", "wall_seconds")


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def bootstrap(values, seed=20261010, samples=10_000):
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(samples, len(data)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def protocol():
    seeds = json.loads(SEEDS_FILE.read_text())
    return {"version": 1, "validation_seeds": seeds["validation"],
            "validation_max_pieces": 20_000, "final_seeds": seeds["final_test"],
            "final_max_pieces": 50_000, "deterministic": True, "action_mask": True,
            "steps": list(STEPS), "selection_rule": {
                "primary": "highest mean capped survival over 32 validation seeds",
                "near_threshold_pieces": 1000,
                "near_definition": "within 1000 mean pieces and paired bootstrap CI includes zero",
                "near_tiebreak": ["higher cap rate", "higher P10", "more paired wins",
                                  "higher mean capped survival"],
                "bootstrap_samples": 10_000, "bootstrap_seed": 20261010,
                "final_test_excluded_from_selection": True},
            "evaluation_code_sha256": sha256(Path(__file__)),
            "seed_file_sha256": sha256(SEEDS_FILE)}


def log(message):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "events.log").open("a", encoding="utf8") as file:
        file.write(f"{now()} {message}\n")
        file.flush()
        os.fsync(file.fileno())
    print(message, flush=True)


def status(stage, state="running", **extra):
    atomic_json(RUN / "status.json", {"stage": stage, "status": state,
                                     "updated_at": now(), **extra})


def source_checkpoints():
    config = json.loads((SOURCE / "config.json").read_text())
    state = json.loads((SOURCE / "transaction_state.json").read_text())
    if state["status"] != "completed" or state["committed_steps"] != STEPS[-1]:
        raise RuntimeError("Candidate source run is not fully committed")
    if (config["policy"] != "CandidateScoringPolicy" or config["device"] != "cuda"
            or config["ppo"]["n_envs"] != 8):
        raise RuntimeError("Candidate source configuration changed")
    checks = {}
    immutable = ("reward_version", "observation_version", "candidate_feature_version",
                 "action_space_version", "policy", "run_seed", "ppo", "network")
    for step in STEPS:
        folder = SOURCE / "milestones" / f"step_{step:09d}"
        meta = json.loads((folder / "metadata.json").read_text())
        checksum = json.loads((folder / "checksum.json").read_text())["model_sha256"]
        if (meta["actual_committed_steps"] != step or meta["task_id"] * config["task_steps"] != step
                or any(meta["training_config"][key] != config[key] for key in immutable)
                or sha256(folder / "model.zip") != checksum):
            raise RuntimeError(f"Milestone integrity or training configuration failed: {step}")
        checks[str(step)] = {"model": str((folder / "model.zip").resolve()),
                             "sha256": checksum, "committed_steps": step,
                             "git_commit": meta["git_commit"],
                             "reward_version": config["reward_version"],
                             "observation_version": config["observation_version"],
                             "candidate_feature_version": config["candidate_feature_version"],
                             "action_space_version": config["action_space_version"],
                             "policy": config["policy"]}
    return checks


def initialize():
    RUN.mkdir(parents=True, exist_ok=True)
    expected = protocol()
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != expected:
        raise RuntimeError("Frozen champion protocol or evaluation code changed")
    if not path.exists():
        atomic_json(path, expected)
    checks = source_checkpoints()
    path = RUN / "source_checkpoints.json"
    if path.exists() and json.loads(path.read_text()) != checks:
        raise RuntimeError("Source checkpoint identities changed")
    if not path.exists():
        atomic_json(path, checks)
    return expected, checks


def evaluate_seed(model, seed, max_pieces):
    """Use the same deterministic MaskablePPO policy and official action masks."""
    env = CandidateTetrisEnv(max_pieces=max_pieces)
    started = time.monotonic()
    try:
        observation, _ = env.reset(seed=seed)
        new_holes = 0
        new_holes_events = 0
        while True:
            mask = env.action_masks()
            action, _ = model.predict(observation, deterministic=True, action_masks=mask)
            observation, _, terminated, truncated, info = env.step(int(action))
            new_holes += info["new_holes"]
            new_holes_events += int(info["new_holes"] > 0)
            if terminated or truncated:
                break
        return {"seed": seed, "pieces_survived": info["pieces"], "lines": info["lines"],
                "score": info["score"], "episode_reward": env.episode_reward,
                "game_over": bool(terminated), "survived_cap": bool(truncated),
                "new_holes_events": new_holes_events,
                "new_holes_total": new_holes, "holes_at_end": info["holes_after"],
                "height_at_end": info["height_after"],
                "wall_seconds": time.monotonic() - started}
    finally:
        env.close()


def validate_row(row, seed, cap):
    if (not all(key in row for key in ROW_FIELDS) or row["seed"] != seed
            or not 1 <= row["pieces_survived"] <= cap
            or row["game_over"] == row["survived_cap"]
            or (row["survived_cap"] and row["pieces_survived"] != cap)
            or row["new_holes_total"] < 0 or row["holes_at_end"] < 0
            or not 0 <= row["height_at_end"] <= 20):
        raise RuntimeError(f"Invalid persisted evaluation row for seed {seed}")


def rows_for(label, model_path, model_sha, steps, seeds, cap):
    import torch
    from sb3_contrib import MaskablePPO

    if sha256(model_path) != model_sha:
        raise RuntimeError("Model changed since protocol freeze")
    directory = RUN / "per_seed" / label
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    model = None
    for seed in seeds:
        destination = directory / f"seed_{seed}.json"
        if destination.exists():
            record = json.loads(destination.read_text())
            if (record.get("model_sha256") != model_sha or record.get("committed_steps") != steps
                    or record.get("max_pieces") != cap or record.get("deterministic") is not True
                    or record.get("action_mask") is not True):
                raise RuntimeError(f"Persisted seed protocol differs: {destination}")
            row = record["result"]
        else:
            if model is None:
                torch.set_num_threads(1)
                model = MaskablePPO.load(str(model_path), device="cpu")
                if (model.num_timesteps != steps
                        or model.policy.__class__.__name__ != "CandidateScoringPolicy"):
                    raise RuntimeError("Loaded policy or timestep disagrees with milestone")
            try:
                row = evaluate_seed(model, seed, cap)
            except Exception:
                log(f"SEED_FAILED label={label} seed={seed} error={traceback.format_exc()!r}")
                raise
            validate_row(row, seed, cap)
            atomic_json(destination, {"model_sha256": model_sha,
                                      "committed_steps": steps, "max_pieces": cap,
                                      "deterministic": True, "action_mask": True,
                                      "result": row})
            log(f"SEED_COMPLETED label={label} seed={seed} pieces={row['pieces_survived']} "
                f"seconds={row['wall_seconds']:.1f}")
        validate_row(row, seed, cap)
        rows.append(row)
    return rows


def summarize(rows, cap):
    result = aggregate(rows, cap)
    values = [row["pieces_survived"] for row in rows]
    deaths = [row for row in rows if row["game_over"]]
    result.update(p10_pieces=percentile(values, 0.1),
                  p90_pieces=percentile(values, 0.9),
                  capped_count=sum(row["survived_cap"] for row in rows),
                  mean_survival_bootstrap_95ci=bootstrap(values),
                  mean_death_height=(statistics.mean(row["height_at_end"] for row in deaths)
                                     if deaths else None),
                  mean_death_holes=(statistics.mean(row["holes_at_end"] for row in deaths)
                                    if deaths else None))
    return result


def validation(checks, proto):
    for step in STEPS:
        key = str(step)
        source = checks[key]
        label = f"validation_{step}"
        rows = rows_for(label, Path(source["model"]), source["sha256"], step,
                        proto["validation_seeds"], proto["validation_max_pieces"])
        result = {"protocol": "champion_validation_v1", "checkpoint": source,
                  "seeds": proto["validation_seeds"], "max_pieces": 20_000,
                  "per_seed": rows, "aggregate": summarize(rows, 20_000)}
        atomic_json(RUN / f"{label}.json", result)
        log(f"VALIDATION_COMPLETED step={step} mean={result['aggregate']['mean_pieces']:.1f}")


def paired(left, right):
    if [r["seed"] for r in left] != [r["seed"] for r in right]:
        raise RuntimeError("Paired seeds differ")
    differences = [a["pieces_survived"] - b["pieces_survived"]
                   for a, b in zip(left, right)]
    return {"mean_difference": statistics.mean(differences),
            "bootstrap_95ci": bootstrap(differences),
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "per_seed": dict(zip((r["seed"] for r in left), differences))}


def comparison():
    results = {step: json.loads((RUN / f"validation_{step}.json").read_text())
               for step in STEPS}
    for step, result in results.items():
        if (result["seeds"] != protocol()["validation_seeds"]
                or result["max_pieces"] != 20_000 or len(result["per_seed"]) != 32):
            raise RuntimeError(f"Incomplete 32-seed validation: {step}")
    pairs = {f"{a}_minus_{b}": paired(results[a]["per_seed"], results[b]["per_seed"])
             for a in STEPS for b in STEPS if a > b}
    best_mean = max(results, key=lambda step: results[step]["aggregate"]["mean_pieces"])
    near = [best_mean]
    for step in STEPS:
        if step == best_mean:
            continue
        pair = paired(results[best_mean]["per_seed"], results[step]["per_seed"])
        if (results[best_mean]["aggregate"]["mean_pieces"]
                - results[step]["aggregate"]["mean_pieces"] <= 1000
                and pair["bootstrap_95ci"][0] <= 0 <= pair["bootstrap_95ci"][1]):
            near.append(step)
    if len(near) == 1:
        selected = best_mean
    else:
        def key(step):
            agg = results[step]["aggregate"]
            wins = sum(paired(results[step]["per_seed"], results[other]["per_seed"])["wins"]
                       for other in near if other != step)
            return (agg["survival_cap_rate"], agg["p10_pieces"], wins,
                    agg["mean_pieces"])
        selected = max(near, key=key)
    decision = {"selected_step": selected, "highest_mean_step": best_mean,
                "near_candidates": near, "selection_uncertain": len(near) > 1,
                "paired": pairs, "aggregate": {str(step): results[step]["aggregate"]
                                             for step in STEPS},
                "selected_before_final_test": now(),
                "rule": json.loads((RUN / "protocol.json").read_text())["selection_rule"]}
    path = RUN / "selection_decision.json"
    if path.exists() and json.loads(path.read_text())["selected_step"] != selected:
        raise RuntimeError("Frozen selection differs; refusing to use final-test data")
    if not path.exists():
        atomic_json(path, decision)
    return json.loads(path.read_text())


def freeze(checks):
    decision = json.loads((RUN / "selection_decision.json").read_text())
    source = checks[str(decision["selected_step"])]
    destination = RUN / "model.zip"
    if destination.exists():
        if sha256(destination) != source["sha256"]:
            raise RuntimeError("Frozen champion model checksum changed")
    else:
        temporary = RUN / "model.zip.tmp"
        shutil.copy2(source["model"], temporary)
        if sha256(temporary) != source["sha256"]:
            raise RuntimeError("Copied champion checksum failed")
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        fsync_dir(RUN)
    record = {"source": source, "model_sha256": source["sha256"],
              "selected_step": decision["selected_step"],
              "validation_report": "validation_" + str(decision["selected_step"]) + ".json",
              "selection_decision": "selection_decision.json", "frozen_at": now()}
    path = RUN / "champion.json"
    if path.exists():
        if json.loads(path.read_text())["model_sha256"] != record["model_sha256"]:
            raise RuntimeError("Champion already frozen to another model")
    else:
        atomic_json(path, record)
    return json.loads(path.read_text())


def final_test(proto):
    champion = json.loads((RUN / "champion.json").read_text())
    model = RUN / "model.zip"
    if sha256(model) != champion["model_sha256"]:
        raise RuntimeError("Frozen champion checksum failed before final test")
    rows = rows_for("final_100", model, champion["model_sha256"],
                    champion["selected_step"], proto["final_seeds"], 50_000)
    result = {"protocol": "champion_final_100_v1", "checkpoint": champion,
              "seeds": proto["final_seeds"], "max_pieces": 50_000,
              "per_seed": rows, "aggregate": summarize(rows, 50_000)}
    atomic_json(RUN / "final_100.json", result)
    log("FINAL_TEST_COMPLETED seeds=100")


def compare_raw(proto):
    candidate = json.loads((RUN / "final_100.json").read_text())
    raw_path = RAW / "evaluations/final/step_009502720.json"
    raw = json.loads(raw_path.read_text())
    raw_model = RAW / "best/model.zip"
    raw_sha = json.loads((RAW / "best/checksum.json").read_text())["model_sha256"]
    if (sha256(raw_model) != raw_sha or raw["protocol"] != "final_test"
            or raw["committed_steps"] != 9_502_720
            or raw["max_pieces"] != 50_000 or raw["seeds"] != proto["final_seeds"]
            or candidate["seeds"] != raw["seeds"] or len(raw["per_seed"]) != 100):
        raise RuntimeError("Existing Raw formal result is not protocol-compatible")
    result = {"raw_record": str(raw_path), "raw_model_sha256": raw_sha,
              "raw_training_steps": raw["committed_steps"],
              "candidate_training_steps": candidate["checkpoint"]["selected_step"],
              "paired": paired(candidate["per_seed"], raw["per_seed"]),
              "raw_aggregate": raw["aggregate"],
              "candidate_aggregate": candidate["aggregate"]}
    atomic_json(RUN / "raw_comparison.json", result)


def traditional_plan():
    cases = {}
    for label, folder in (("Legacy V1", "v1"), ("V2 Hold-only", "hold"),
                          ("V2 Beam-8", "beam")):
        aggregate_path = (ROOT / "reports/traditional_longlife"
                          / "validation10_100000_20261008" / folder / "aggregate.json")
        data = json.loads(aggregate_path.read_text())
        rate = data["end_to_end_pieces_per_second"]
        cases[label] = {"observed_pieces_per_second": rate,
                        "100_seeds_x_50000_cap_hours_upper_bound": 5_000_000 / rate / 3600,
                        "basis": str(aggregate_path.relative_to(ROOT))}
    result = {"predeclared_protocol": {"seeds": list(range(200000, 200100)),
                                       "max_pieces": 50_000,
                                       "same_core_rules": True,
                                       "same_seed_and_piece_sequence": True,
                                       "same_public_hold_next": True,
                                       "same_score": True},
              "sequential_cost_estimates": cases,
              "note": "Legacy may die early; these are cap-based upper bounds from a different hardware load. Do not use final-test outcomes to change the frozen Candidate checkpoint."}
    atomic_json(RUN / "traditional_plan.json", result)


def reports():
    selection = json.loads((RUN / "selection_decision.json").read_text())
    results = {step: json.loads((RUN / f"validation_{step}.json").read_text())
               for step in STEPS}
    rows = ["# Candidate PPO 长寿命 Champion 选择", "",
            "统一协议：validation seeds 100000–100031；每局最多 20,000 块；确定性推理；官方 Action Mask。达到上限按右截断处理。选择在正式测试之前冻结。", "",
            "| Checkpoint | 平均存活 | 中位 | P10 | P90 | 截断 | 平均消行 | 平均分 | 新增洞/100块 | 死亡最高列 | 死亡洞数 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step in STEPS:
        a = results[step]["aggregate"]
        rows.append(f"| {step:,} | {a['mean_pieces']:.1f} | {a['median_pieces']:.1f} | "
                    f"{a['p10_pieces']} | {a['p90_pieces']} | {a['capped_count']}/32 | "
                    f"{a['mean_lines']:.1f} | {a['mean_score']:.1f} | "
                    f"{a['new_holes_per_100_pieces']:.2f} | "
                    f"{a['mean_death_height'] if a['mean_death_height'] is not None else '—'} | "
                    f"{a['mean_death_holes'] if a['mean_death_holes'] is not None else '—'} |")
    rows += ["", f"选中 **{selection['selected_step']:,} steps**；平均存活最高的为 "
             f"{selection['highest_mean_step']:,} steps。接近候选：{selection['near_candidates']}。"
             f"选择不确定性：{selection['selection_uncertain']}。规则优先比较平均截断存活；"
             "差距不超过 1000 块且配对 bootstrap 区间包含零时，再比较截断率、P10 与配对胜局。", "",
             "配对存活差（前者减后者；32 局重采样 10,000 次）：", "",
             "| 配对 | 平均差 | 95% bootstrap 区间 | 胜/平/负 |",
             "|---|---:|---|---:|"]
    for name, pair in selection["paired"].items():
        rows.append(f"| {name} | {pair['mean_difference']:.1f} | "
                    f"[{pair['bootstrap_95ci'][0]:.1f}, {pair['bootstrap_95ci'][1]:.1f}] | "
                    f"{pair['wins']}/{pair['ties']}/{pair['losses']} |")
    rows += ["", "每个 checkpoint 的逐 seed 原始结果、SHA256、配置与选模记录保存在 `runs/ppo_candidate_champion/`。20,000 块截断只表明至少存活到上限，不能解释为真实寿命。", ""]
    SELECTION_REPORT.write_text("\n".join(rows))

    candidate = json.loads((RUN / "final_100.json").read_text())
    compare = json.loads((RUN / "raw_comparison.json").read_text())
    plan = json.loads((RUN / "traditional_plan.json").read_text())
    a, b = candidate["aggregate"], compare["raw_aggregate"]
    bins = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 20000),
            (20000, 50000), (50000, 50001)]
    distribution = [(f"{lo}–{hi-1}", sum(lo <= row["pieces_survived"] < hi
                                           for row in candidate["per_seed"]))
                    for lo, hi in bins]
    lines = ["# Candidate PPO Champion 正式 100-seed 测试", "",
             f"冻结模型：{candidate['checkpoint']['selected_step']:,} steps，SHA256 "
             f"`{candidate['checkpoint']['model_sha256']}`。100 个 seeds 200000–200099；"
             "每局最多 50,000 块；确定性推理和 Action Mask。正式测试在选择后只运行一次。"
             "这些 seeds 此前用于 Raw PPO 正式报告，不能称为全新测试集。", "",
             "| 指标 | Candidate Champion | Raw PPO 已冻结 best |",
             "|---|---:|---:|",
             f"| 平均存活 | {a['mean_pieces']:.1f} | {b['mean_pieces']:.1f} |",
             f"| 中位存活 | {a['median_pieces']:.1f} | {b['median_pieces']:.1f} |",
             f"| P10 / P90 存活 | {a['p10_pieces']} / {a['p90_pieces']} | — |",
             f"| 平均消行 | {a['mean_lines']:.1f} | {b['mean_lines']:.1f} |",
             f"| 平均分 | {a['mean_score']:.1f} | {b['mean_score']:.1f} |",
             f"| 50,000 块截断 | {a['capped_count']}/100 | {int(b['survival_cap_rate']*100)}/100 |",
             f"| Game Over | {100-a['capped_count']}/100 | {int(b['game_over_rate']*100)}/100 |", "",
             f"Candidate 平均截断存活 95% bootstrap 区间：{a['mean_survival_bootstrap_95ci']}；"
             f"配对 Candidate−Raw 平均差 {compare['paired']['mean_difference']:.1f} 块，"
             f"95% 区间 {compare['paired']['bootstrap_95ci']}。"
             "达到 50,000 块属于右截断，不能当成死亡寿命。", "",
             "存活块数分布：", "", "| 区间 | 局数 |", "|---|---:|"]
    lines += [f"| {label} | {count} |" for label, count in distribution]
    lines += ["", "与传统算法公平比较：同一组 100 个 seeds、50,000 块上限、相同 Core 规则与方块序列、公开 Hold/Next、相同计分；Candidate checkpoint 冻结。"
              "已有传统 10 局×100,000 块结果与本轮协议不同，不能直接判胜负。", "",
              "| 策略 | 估计完整 100×50,000 上限耗时 |",
              "|---|---:|"]
    for label, cost in plan["sequential_cost_estimates"].items():
        lines.append(f"| {label} | {cost['100_seeds_x_50000_cap_hours_upper_bound']:.1f} 小时 |")
    lines += ["", "传统 V2 Hold-only 与 Beam-8 在已有 10 局×100,000 块记录中均为 10/10 达到上限；"
              "由于种子与上限不同，本轮还不能量化 Champion 与它们的公平寿命差距。", "",
              "以上按历史吞吐推算，实际死亡较早会缩短时间；传统策略应逐个运行，不并发启动多个高负载实验。", "",
              "逐 seed 正式结果：", "", "| Seed | 存活 | 消行 | 分数 | 结局 |",
              "|---:|---:|---:|---:|---|"]
    for row in candidate["per_seed"]:
        lines.append(f"| {row['seed']} | {row['pieces_survived']} | {row['lines']} | "
                     f"{row['score']} | {'上限截断' if row['survived_cap'] else 'Game Over'} |")
    lines += ["", "下一步建议以已冻结 Champion 做同协议传统策略配对评测；不要根据正式测试调整本次 Champion。"
              "7M–10M 长寿命验证已有回落，暂不建议仅延长训练步数；应先分析失败棋盘和稳定性，再决定算法改进或新一轮训练。", ""]
    FINAL_REPORT.write_text("\n".join(lines))
    changelog = ROOT / "CHANGELOG.md"
    text = changelog.read_text()
    marker = "## 2026-10-10\n"
    entry = (f"\n- Candidate PPO Champion 从 7M/8M/9M/10M 统一 32-seed、20000 块验证中预先选出 "
             f"{selection['selected_step']:,} steps；冻结后完成 100-seed、50000 块正式测试："
             f"平均存活 {a['mean_pieces']:.1f} 块、{a['capped_count']}/100 局达到上限。"
             "详见 `reports/experiments/ppo_candidate_champion_selection.md` 和 "
             "`reports/experiments/ppo_candidate_final_100seed.md`。\n")
    if entry not in text:
        if marker not in text:
            raise RuntimeError("CHANGELOG date header missing")
        changelog.write_text(text.replace(marker, marker + entry, 1))


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / ".pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proto, checks = initialize()
        for stage in STAGES[:-1]:
            status(stage)
            log(f"STAGE_STARTED {stage}")
            try:
                if stage == "validate_7m_8m_9m_10m":
                    validation(checks, proto)
                elif stage == "compare_checkpoints":
                    comparison()
                elif stage == "freeze_champion":
                    freeze(checks)
                elif stage == "champion_final_test_100":
                    final_test(proto)
                elif stage == "compare_raw_final":
                    compare_raw(proto)
                elif stage == "traditional_comparison_plan":
                    traditional_plan()
                elif stage == "generate_report":
                    reports()
                log(f"STAGE_COMPLETED {stage}")
            except Exception as error:
                status(stage, "failed", error=repr(error))
                log(f"STAGE_FAILED {stage} {traceback.format_exc()!r}")
                raise
        status("completed", "completed")
        log("PIPELINE_COMPLETED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        state = json.loads((RUN / "status.json").read_text()) if (RUN / "status.json").exists() else {}
        counts = {label: len(list((RUN / "per_seed" / label).glob("seed_*.json")))
                  for label in [*(f"validation_{step}" for step in STEPS), "final_100"]}
        print(json.dumps({"state": state, "completed_games": counts}, ensure_ascii=False, indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
