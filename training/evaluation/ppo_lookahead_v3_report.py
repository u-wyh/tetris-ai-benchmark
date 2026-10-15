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
            lines.append(f"| {variant} | {count}/{len(seeds)} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | "
                         f"{row['capped']}/{count} | {row['mean_score']:.0f} | {row['four_fraction']:.1%} | "
                         f"{row['top1_retention']:.2%} | {row['hold_changes']} | {row['replacement_fours']} | "
                         f"{row['p95_game_ms']:.2f} | {ratio:.3f} |" if ratio is not None else
                         f"| {variant} | {count}/{len(seeds)} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | "
                         f"{row['capped']}/{count} | {row['mean_score']:.0f} | {row['four_fraction']:.1%} | "
                         f"{row['top1_retention']:.2%} | {row['hold_changes']} | {row['replacement_fours']} | "
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
    lines += ["## 判读", "", "16 个开发 seed 与已知失败 seed 隔离，独立确认 seed 为 991200–991231。"
              "仅在完整配对组上报告均值；进行中的逐局文件只显示完成数量，不能代表模型均值。"
              "每个替换事件存于对应逐局压缩 JSON，可恢复全部事件；不同存活长度的总分和共同存活得分分别看。", "",
              "诊断与候选规则见 [失败诊断](ppo30m_lookahead_v3_diagnosis.md)。", ""]
    output = ROOT / "reports/experiments/ppo30m_lookahead_v3_comparison.md"
    output.write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    print(render())
