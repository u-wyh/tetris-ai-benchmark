"""Paired confirmation report for frozen Hybrid V2 rules."""
import json
import random
import statistics

from training.evaluation.hybrid_v2_queue import RUN, DEV, CONFIRM
from training.train_ppo import ROOT


def ci(values, repeats=4000):
    rng = random.Random(20261014)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return means[int(.025 * repeats)], means[int(.975 * repeats)]


def main():
    protocol = json.loads((RUN / "protocol.json").read_text())
    variants = protocol["variants"]
    data = {phase: {v: {seed: json.loads((RUN / phase / v / f"seed_{seed}.json").read_text())
                        for seed in seeds} for v in variants}
            for phase, seeds in (("development_20k", DEV), ("development_100k", DEV),
                                 ("confirmation_100k", CONFIRM))}
    lines = ["# Hybrid V2 统一对照", "",
        "冻结 PPO 30M、BC Champion 与原 B12 不变。C1/C2 规则在确认组之前写入 `hybrid_v2_design.md` 并以源码 SHA256 冻结。",
        f"模型 SHA256：PPO `{protocol['models']['ppo']}`；BC `{protocol['models']['bc']}`；"
        f"V2 规则 SHA256 `{protocol['v2_rules_sha256']}`。开发种子 710000–710015，独立确认种子 710100–710131；每局确定性、正式 Action Mask、相同 100K 上限。", "",
        "## 汇总", ""]
    for phase in data:
        lines += [f"### {phase}", "",
            "| 模型 | 平均存活 | 中位存活 | P10 存活 | 跑满 | 平均总分 | 每千块得分 | 四消占比 | PPO 动作保留率 | 平均耗时（秒） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in variants:
            s = json.loads((RUN / phase / v / "summary.json").read_text())
            lines.append(f"| {v} | {s['mean_pieces']:.0f} | {s['median_pieces']:.0f} | {s['p10_pieces']:.0f} | "
                f"{s['capped']}/{s['games']} | {s['mean_score']:.0f} | {s['score_per_1000_pieces']:.0f} | "
                f"{s['four_clear_fraction']:.2%} | {s['ppo_action_fraction']:.2%} | {s['mean_wall_seconds']:.1f} |")
        lines.append("")
    lines += ["## 确认组逐 seed 配对", "",
        "| Seed | " + " | ".join(f"{v} 存活/得分" for v in variants) + " |",
        "|---:|" + "---:|" * len(variants)]
    for seed in CONFIRM:
        lines.append("| " + str(seed) + " | " + " | ".join(
            f"{data['confirmation_100k'][v][seed]['pieces_survived']}/{data['confirmation_100k'][v][seed]['score']}"
            for v in variants) + " |")
    lines += ["", "## 配对差异与共同存活区间", "",
        "95% 区间是对固定 32 个 seed 的配对均值做确定性 bootstrap；仅描述这个种子样本。触顶局按右截断，不能据此推断真实死亡时间。",
        "共同存活区间使用双方均达到的最后一个完整千块点累计得分；不足 1000 块的配对不进入该项。", "",
        "| 对比 | 平均存活差（95% CI） | 平均总分差（95% CI） | 共同区间得分差 |",
        "|---|---:|---:|---:|"]
    confirm = data["confirmation_100k"]
    for v in variants:
        if v in ("ppo", "bc"):
            continue
        for base in ("B12", "bc", "ppo"):
            if v == base:
                continue
            survival = [confirm[v][s]["pieces_survived"] - confirm[base][s]["pieces_survived"] for s in CONFIRM]
            scores = [confirm[v][s]["score"] - confirm[base][s]["score"] for s in CONFIRM]
            common = []
            for seed in CONFIRM:
                a, b = confirm[v][seed], confirm[base][seed]
                point = min(a["pieces_survived"], b["pieces_survived"]) // 1000 - 1
                if point >= 0:
                    common.append(a["score_milestones"][point]["score"] -
                                  b["score_milestones"][point]["score"])
            slo, shi = ci(survival)
            qlo, qhi = ci(scores)
            lines.append(f"| {v} − {base} | {statistics.mean(survival):+.0f} ({slo:+.0f}, {shi:+.0f}) | "
                         f"{statistics.mean(scores):+.0f} ({qlo:+.0f}, {qhi:+.0f}) | "
                         f"{statistics.mean(common):+.0f}（{len(common)} 对） |")
    lines += ["", "## BC 拦截与候选风险", "",
        "风险差为相同拦截状态下 PPO 候选减 BC 候选；正数表示 BC 的该指标更低。只统计真正采用 BC 的切换点。", "",
        "| 模型 | BC 动作 | BC 连续区间 | 平均高度差 | 平均新增洞差 | 平均中央顶部占用差 |",
        "|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        if v in ("ppo", "bc"):
            continue
        rows = list(confirm[v].values())
        events = [event for row in rows for event in row["switches"] if event["source"] == "bc"]
        if events:
            differences = {key: statistics.mean(e["ppo_risk"][key] - e["bc_risk"][key]
                                               for e in events)
                           for key in ("height", "new_holes", "central_top4")}
            lines.append(f"| {v} | {sum(r['actions_by_source']['bc'] for r in rows)} | {len(events)} | "
                f"{differences['height']:.2f} | {differences['new_holes']:.2f} | "
                f"{differences['central_top4']:.2f} |")
        else:
            lines.append(f"| {v} | 0 | 0 | — | — | — |")
    summaries = {v: json.loads((RUN / "confirmation_100k" / v / "summary.json").read_text())
                 for v in variants}
    lines += ["", "## 判断", ""]
    promising = []
    for v in variants:
        if v not in ("C1", "C2"):
            continue
        s, b, p = summaries[v], summaries["B12"], summaries["ppo"]
        lines.append(f"- {v} 相对 B12：平均存活 {s['mean_pieces']-b['mean_pieces']:+.0f} 块，"
                     f"P10 {s['p10_pieces']-b['p10_pieces']:+.0f} 块，"
                     f"平均总分 {s['mean_score']-b['mean_score']:+.0f}，"
                     f"四消事件占比为 PPO 的 {s['four_clear_fraction']/max(p['four_clear_fraction'],1e-9):.1%}。")
        if (s["p10_pieces"] > b["p10_pieces"] and
            s["mean_score"] >= .9 * b["mean_score"] and
            s["four_clear_fraction"] >= .8 * p["four_clear_fraction"]):
            promising.append(v)
    lines += ["", "上述为预设指标的描述性比较，应结合配对区间与异常早死局判断，不把单次确认解释为所有种子的保证。", ""]
    if promising:
        lines.append("按 P10 改善、至少保留 B12 九成平均分及 PPO 八成四消占比的预设门槛，"
                     + "/".join(promising) + " 值得进入单模型蒸馏设计；蒸馏仍需独立批准和新实验。")
    else:
        lines.append("没有方案同时达到预设的低分位生存、得分和四消门槛；暂不建议启动蒸馏训练，"
                     "应回到 PPO 30M 危险状态的局部专家示范分析。")
    lines.append("")
    lines += ["", "## 拦截轨迹与解释", "",
        "逐局结果在 `runs/hybrid_v2/`：`source_runs` 可恢复每一步的控制策略，`switches` 记录切换时 PPO/BC 候选风险。"
        "得分、每千块得分和共同存活区间分开报告，避免提前死亡局造成的误读。", "",
        "## 后续训练方向", "",
        "只有在独立确认组同时显示低分位生存和四消/得分的可复现改善时，才考虑蒸馏。候选单模型以 PPO 30M 初始化，"
        "对有明确风险改善的状态学习 BC 局部动作，对安全状态保持 PPO 四消动作，并用生存 P10、四消占比和总分共同选模。"
        "本实验没有启动 PPO、BC、DAgger 或蒸馏训练。", ""]
    report = ROOT / "reports/experiments/hybrid_v2_comparison.md"
    report.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
