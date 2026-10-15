"""Summarize completed, paired lookahead games without treating caps as deaths."""
import json

from training.evaluation.ppo_lookahead_queue import DEV, RUN
from training.train_ppo import ROOT


def _rows(phase, variant):
    folder = RUN / phase / variant
    return {int(path.stem.split("_")[-1]): json.loads(path.read_text())
            for path in folder.glob("seed_*.json")} if folder.exists() else {}


def common_score_ratio(phase, variant, baseline="ppo"):
    left, right = _rows(phase, variant), _rows(phase, baseline)
    ratios = []
    for seed in set(left) & set(right):
        a, b = left[seed], right[seed]
        point = min(a["pieces"], b["pieces"]) // 1000 - 1
        if point >= 0:
            score_a = a["milestones"][point]["score"]
            score_b = b["milestones"][point]["score"]
            ratios.append(score_a / max(1, score_b))
    return sum(ratios) / len(ratios) if ratios else None, len(ratios)


def render():
    status = json.loads((RUN / "status.json").read_text())
    lines = ["# PPO 30M Top-K 有限前瞻对照", "",
             f"当前后台状态：`{status['stage']}`。达到评测上限按右截断，短局触顶不表示可长期存活。", "",
             "PPO30M 冻结 SHA256：`eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34`。", ""]
    for phase in ("smoke_1k", "development_20k", "development_50k", "confirmation_100k"):
        folder = RUN / phase
        if not folder.exists(): continue
        lines += [f"## {phase}", "", "| 策略 | 完成局数 | 平均存活 | P10 | 触顶 | 平均正式得分 | 四消占比 | Top-1 保留 | S2 触发 | 平均节点 | 平均决策 ms | P95 局 ms | 共同存活分数比 PPO |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in ("ppo", "bc", "B12", "S1", "S2"):
            path = folder / variant / "summary.json"
            if not path.exists(): continue
            row = json.loads(path.read_text())
            common, pairs = common_score_ratio(phase, variant)
            ratio = f"{common:.3f} ({pairs})" if common is not None else "—"
            lines.append(f"| {variant} | {row['games']} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | {row['capped']}/{row['games']} | {row['mean_score']:.0f} | {row['four_fraction']:.1%} | {row['top1_retention']:.1%} | {row['search_trigger']:.1%} | {row['mean_nodes']:.1f} | {row['mean_ms']:.2f} | {row['p95_game_ms']:.2f} | {ratio} |")
        lines.append("")
    decision = RUN / "decision.json"
    if decision.exists():
        record = json.loads(decision.read_text())
        lines += ["## 开发门槛", "", f"通过方案：{', '.join(record['winners']) or '无'}。", ""]
    lines += ["## 解释", "",
              "S1 只评价当前 PPO Top-5 落点；S2 仅在公开的高风险触发条件满足时，增加一层官方 BFS 搜索。两者均保留 PPO Top-1 回退。",
              "每局记录首次 100 次替换动作及可见状态边界；完整逐 seed 数据在 `runs/ppo30m_lookahead/`。",
              "若 S2 与 S1 无显著差异，应先检查触发率及一层搜索是否区分根候选，不直接增加深度或蒸馏。", ""]
    output = ROOT / "reports/experiments/ppo30m_lookahead_comparison.md"
    output.write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    print(render())
