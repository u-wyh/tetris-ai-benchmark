"""Paired staged report for the one frozen B12 + V2 Hold rescue rule."""

import json
import statistics

import numpy as np

from training.train_ppo import ROOT, atomic_json

RUN = ROOT / "runs/hybrid_b12_v2_rescue"
REPORT = ROOT / "reports/experiments/hybrid_b12_v2_rescue.md"


def paired_ci(values):
    rng = np.random.default_rng(1441032)
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    return [float(v) for v in np.quantile(values[draws].mean(axis=1), [.025, .975])]


def records(phase, variant, seeds):
    return [json.loads((RUN / phase / variant / f"seed_{seed}.json").read_text()) for seed in seeds]


def write_report(protocol, outcome):
    phases = [("development_20k", protocol["dev"])]
    if (RUN / "development_100k_gate.json").exists():
        phases.append(("development_100k", protocol["dev"]))
    if outcome == "confirmation_completed":
        phases.append(("confirmation_100k", protocol["confirm"]))
    lines = ["# Hybrid B12 + 传统 V2 单步紧急救援", "",
             "冻结原 B12、PPO 30M、BC Champion 和传统 V2 Hold。历史 1421xxx 案例只用于实现核查；新种子为开发组 1440000..1440031、确认组 1441000..1441031。触顶均为右截断。", "",
             f"模型 SHA256：PPO `{protocol['models']['ppo']}`；BC `{protocol['models']['bc']}`。救援规则 SHA256 `{protocol['rescue_rule_sha256']}`；Core SHA256 `{protocol['core_sha256']}`。", "",
             "## 冻结规则", "",
             "每一步先取得原 B12 决策；仅当当前公开棋盘最高列≥17 且凹凸度≥25 时调用传统 V2 Hold。V2 动作必须是正式合法动作、与 PPO 和 BC 都不同、且可见一步模拟不立即 Lock Out，才替换当前一步。下一步重新执行完整 B12 判断。规则不读取隐藏 Bag 或 RNG。阈值是历史案例启发的待验证假设。", "",
             "## 分阶段结果", ""]
    for phase, seeds in phases:
        summaries = {v: json.loads((RUN / phase / v / "summary.json").read_text()) for v in protocol["variants"]}
        lines += [f"### {phase}", "",
                  "| 模型 | 平均存活 | P10 | 真死亡 | 触顶 | <10K 早死 | 平均得分（亿） | 四消占比 | V2 调用/改动作 | 推理毫秒/块 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in protocol["variants"]:
            s = summaries[v]
            lines.append(f"| {v} | {s['mean_pieces']:.0f} | {s['p10_pieces']:.0f} | {s['game_over']} | {s['capped']}/{s['games']} | {s['early_under_10000']} | {s['mean_score']/1e8:.2f} | {s['four_fraction']:.1%} | {s['v2_calls']}/{s['v2_changed']} | {1000*s['mean_inference_seconds']/s['mean_pieces']:.3f} |")
        lines.append("")
        if phase != "confirmation_100k":
            gate = json.loads((RUN / f"{phase}_gate.json").read_text())
            lines += [f"冻结门槛：**{'通过' if gate['pass'] else '未通过'}**，逐项：{gate['checks']}。", ""]
    final_phase, seeds = phases[-1]
    rescue = records(final_phase, "Rescue", seeds)
    b12 = records(final_phase, "B12", seeds)
    calls = sum(r["v2_calls"] for r in rescue)
    changes = sum(r["v2_changed"] for r in rescue)
    reasons = {}
    for row in rescue:
        for e in row["intercepts"]:
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    new_early = [r["seed"] for r, b in zip(rescue, b12) if r["pieces"] < 10000 <= b["pieces"]]
    prevented = [r["seed"] for r, b in zip(rescue, b12) if b["pieces"] < 10000 <= r["pieces"]]
    lines += ["## 救援行为", "",
              f"完成阶段 V2 实际调用 {calls} 次，其中改变原 B12 动作 {changes} 次，占调用的 {changes/max(1,calls):.1%}；原因计数 {reasons}。各次事件保存在对局结果中，包含当步公开高度、凹凸度及 PPO/BC/V2 Action ID。", "",
              f"相对 B12 新增 <10K 早死 seed：{new_early}；避免 <10K 早死 seed：{prevented}。逐 seed 结果见下表；不同轨迹的结局差异不能直接归因于单次触发。", ""]
    if outcome == "confirmation_completed":
        diffs = [r["pieces"] - b["pieces"] for r, b in zip(rescue, b12)]
        score_diffs = [r["score"] - b["score"] for r, b in zip(rescue, b12)]
        ci, score_ci = paired_ci(diffs), paired_ci(score_diffs)
        a = json.loads((RUN / final_phase / "Rescue" / "summary.json").read_text())
        b = json.loads((RUN / final_phase / "B12" / "summary.json").read_text())
        checks = {"p10_higher": a["p10_pieces"] > b["p10_pieces"],
                  "mean_higher": a["mean_pieces"] > b["mean_pieces"],
                  "paired_ci_positive": ci[0] > 0,
                  "score_95pct": a["mean_score"] >= .95 * b["mean_score"],
                  "four_90pct": a["four_fraction"] >= .9 * b["four_fraction"],
                  "no_extra_early_death": a["early_under_10000"] <= b["early_under_10000"]}
        decision = {"checks": checks, "confirmed": all(checks.values()),
                    "survival_mean_difference": statistics.mean(diffs), "survival_ci95": ci,
                    "score_mean_difference": statistics.mean(score_diffs), "score_ci95": score_ci}
        atomic_json(RUN / "confirmation_decision.json", decision)
        lines += ["## 独立确认", "",
                  f"按 seed 配对的平均已观测存活差 {statistics.mean(diffs):+.0f} 块，Bootstrap 95% 区间 [{ci[0]:+.0f}, {ci[1]:+.0f}]；总分差 {statistics.mean(score_diffs)/1e8:+.2f} 亿，区间 [{score_ci[0]/1e8:+.2f}, {score_ci[1]/1e8:+.2f}] 亿。", "",
                  f"预设确认门槛：{checks}；**{'通过' if decision['confirmed'] else '未通过'}**。", ""]
    else:
        decision = {"confirmed": False}
        lines += ["## 独立确认", "", "开发阶段未通过冻结门槛，独立确认种子未运行。", ""]
    lines += ["| Seed | B12 存活/亿分 | Rescue 存活/亿分 | 差值块数 | V2 调用/改动作 |",
              "|---:|---:|---:|---:|---:|"]
    for r, b in zip(rescue, b12):
        lines.append(f"| {r['seed']} | {b['pieces']}{'+' if b['capped'] else ''}/{b['score']/1e8:.2f} | {r['pieces']}{'+' if r['capped'] else ''}/{r['score']/1e8:.2f} | {r['pieces']-b['pieces']:+d} | {r['v2_calls']}/{r['v2_changed']} |")
    lines += ["", "## 结论", "",
              "若未通过全部预设门槛，保留原 B12，不调整本规则，也不复用确认种子。短局触顶不代表长期存活，且总分受寿命和等级影响；本轮未训练模型或修改网页默认 AI。", ""]
    REPORT.write_text("\n".join(lines) + "\n")
    return decision
