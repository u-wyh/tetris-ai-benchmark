"""Render V3 results without treating incomplete groups as representative means."""
import json

from training.evaluation.ppo_lookahead_v3_audit import RUN
from training.evaluation.ppo_lookahead_v3_queue import DEV, SMOKE, CONFIRM, common_ratio
from training.train_ppo import ROOT


def render():
    status_path = RUN / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stage": "not_started"}
    lines = ["# PPO 30M 保守搜索 V3 对照", "", f"后台状态：`{status['stage']}`。",
             "已知 seed 990109 仅用于诊断；新开发 seed 为 991100–991115。达到上限均为右截断。", ""]
    stages = (("smoke_1k", SMOKE), ("development_20k", DEV),
              ("development_50k", DEV), ("confirmation_100k", CONFIRM))
    for stage, seeds in stages:
        folder = RUN / stage
        if not folder.exists():
            continue
        lines += [f"## {stage}", "", "| 策略 | 完成/计划 | 平均存活 | P10 | 触顶 | 平均总分 | 四消占比 | Top-1 保留 | Hold 改写 | 改写当步四消 | 各局 P95 的 P95 ms | 共同存活分数比 PPO |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in ("ppo", "B12", "S1", "V3A", "V3B"):
            path = folder / variant / "summary.json"
            if not (folder / variant).exists():
                continue
            count = len(list((folder / variant).glob("seed_*.json.gz")))
            if not path.exists() or count != len(seeds):
                lines.append(f"| {variant} | {count}/{len(seeds)} | — | — | — | — | — | — | — | — | — | — |")
                continue
            row = json.loads(path.read_text())
            ratio = common_ratio(stage, variant, seeds) if (folder / "ppo" / "summary.json").exists() else None
            retention = "—" if variant == "B12" else f"{row['top1_retention']:.2%}"
            hold_changes = "—" if variant in ("ppo", "B12") else str(row["hold_changes"])
            replacement_fours = "—" if variant in ("ppo", "B12") else str(row["replacement_fours"])
            lines.append(f"| {variant} | {count}/{len(seeds)} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | "
                         f"{row['capped']}/{count} | {row['mean_score']:.0f} | {row['four_fraction']:.1%} | "
                         f"{retention} | {hold_changes} | {replacement_fours} | "
                         f"{row['p95_game_ms']:.2f} | {ratio:.3f} |" if ratio is not None else
                         f"| {variant} | {count}/{len(seeds)} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | "
                         f"{row['capped']}/{count} | {row['mean_score']:.0f} | {row['four_fraction']:.1%} | "
                         f"{retention} | {hold_changes} | {replacement_fours} | "
                         f"{row['p95_game_ms']:.2f} | — |")
        lines.append("")
    gate_path = RUN / "development_gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        lines += ["## 开发门槛", ""]
        for variant, result in gate.items():
            lines.append(f"- {variant}: 共同存活分数比 {result['common_score_ratio']:.3f}；"
                         f"新早死 seed {result['severe_paired_early_death']}；通过预设安全门槛：{result['passes']}。")
        lines.append("")
    extension_path = RUN / "extension_gate.json"
    if extension_path.exists():
        gate = json.loads(extension_path.read_text())
        lines += ["## 50K 扩展门槛", ""]
        for variant, result in gate.items():
            lines.append(f"- {variant}: 共同存活分数比 {result['common_score_ratio']:.3f}；"
                         f"严重新增早死筛选 {result['severe_paired_early_death']}；"
                         f"通过预设安全门槛：{result['passes']}。")
        lines.append("")
    if status["stage"] == "completed_no_go_extension":
        rows = {v: json.loads((RUN / "development_50k" / v / "summary.json").read_text())
                for v in ("ppo", "B12", "V3B")}
        from training.evaluation.ppo_lookahead_v3_queue import read_gzip
        paired = []
        for seed in DEV:
            a = read_gzip(RUN / "development_50k" / "V3B" / f"seed_{seed}.json.gz")
            b = read_gzip(RUN / "development_50k" / "ppo" / f"seed_{seed}.json.gz")
            paired.append((seed, a["pieces"]-b["pieces"]))
        gains = [(seed, diff) for seed, diff in paired if diff > 0]
        losses = [(seed, diff) for seed, diff in paired if diff < 0]
        lines += ["## 最终判断", "",
                  f"V3-B 在 50K 的平均存活比 PPO 低 {rows['ppo']['mean_pieces']-rows['V3B']['mean_pieces']:.0f} 块，"
                  f"P10 低 {rows['ppo']['p10_pieces']-rows['V3B']['p10_pieces']:.0f} 块，"
                  f"触顶 {rows['V3B']['capped']}/16 对 PPO {rows['ppo']['capped']}/16；"
                  "四消占比只低约 0.3 个百分点，但长期生存没有改善。",
                  f"逐 seed 存活改善：{gains}；退步：{losses}。其余相同触顶 seed 的差值按右截断解释，不能当成真实生存相等。",
                  "V3-A 在 20K 已有同快照确认的有害替换；V3-B 虽拦住了 Hold 改写，仍在多个 50K 种子提前死亡。"
                  "V3-B 末次改写常发生于濒死状态，不能仅凭轨迹把先前一次四消替换断言为这些死亡的原因。",
                  "两方案均未通过长期生存目标，按预设 no-go 结束；独立 100K 确认未启动，不进行蒸馏或任何神经网络训练。"
                  "保留 PPO 30M 与 Hybrid B12 为基线。", "",
                  f"决策耗时按各局 P95 的 P95 汇总：PPO {rows['ppo']['p95_game_ms']:.2f} ms，"
                  f"V3-B {rows['V3B']['p95_game_ms']:.2f} ms，"
                  f"B12 {rows['B12']['p95_game_ms']:.2f} ms。V3-B 推理在 CPU 上运行，"
                  "开销可接受；本轮停止原因是效果未达标。", ""]
    lines += ["## 判读", "", "16 个开发 seed 与已知失败 seed 隔离，独立确认 seed 为 991200–991231。"
              "仅在完整配对组上报告均值；进行中的逐局文件只显示完成数量，不能代表模型均值。"
              "每个替换事件存于对应逐局压缩 JSON，可恢复全部事件；不同存活长度的总分和共同存活得分分别看。", "",
              "诊断与候选规则见 [失败诊断](ppo30m_lookahead_v3_diagnosis.md)。", ""]
    output = ROOT / "reports/experiments/ppo30m_lookahead_v3_comparison.md"
    output.write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    print(render())
