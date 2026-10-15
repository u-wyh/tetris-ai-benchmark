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
        lines += [f"## {phase}", "", "| 策略 | 完成局数 | 平均存活 | P10 | 触顶 | 平均正式得分 | 四消占比 | Top-1 保留 | S2 触发 | 平均节点 | 平均决策 ms | 各局 P95 的 P95 ms | 共同存活分数比 PPO |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in ("ppo", "bc", "B12", "S1", "S2"):
            path = folder / variant / "summary.json"
            if not path.exists(): continue
            row = json.loads(path.read_text())
            common, pairs = common_score_ratio(phase, variant)
            ratio = f"{common:.3f} ({pairs})" if common is not None else "—"
            top1 = f"{row['top1_retention']:.1%}" if variant in ("ppo", "S1", "S2") else "—"
            trigger = f"{row['search_trigger']:.1%}" if variant == "S2" else "—"
            nodes = f"{row['mean_nodes']:.1f}" if variant in ("S1", "S2") else "—"
            lines.append(f"| {variant} | {row['games']} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | {row['capped']}/{row['games']} | {row['mean_score']:.0f} | {row['four_fraction']:.1%} | {top1} | {trigger} | {nodes} | {row['mean_ms']:.2f} | {row['p95_game_ms']:.2f} | {ratio} |")
        lines.append("")
    decision = RUN / "decision.json"
    if decision.exists():
        record = json.loads(decision.read_text())
        lines += ["## 开发门槛", "", f"通过方案：{', '.join(record['winners']) or '无'}。", ""]
    if status["stage"] == "completed_no_go_counterfactual":
        partial_s1 = _rows("development_50k", "S1")
        partial_s2 = _rows("development_50k", "S2")
        lines += ["## 停止决定", "",
                  "20K 四方案各 16 局已完成。50K 的 PPO 与 B12 各 16 局完成；S1 仅完成 "
                  f"{len(partial_s1)}/16 局，S2 完成 {len(partial_s2)}/16 局。"
                  "按完成顺序截取的 S1 部分结果有选择偏差，不计算 50K 模型平均值或宣称胜负。",
                  "由于两个搜索版本都出现同一 seed 的新增异常早死，且同快照分支确认一次 S1 替换可直接导致近端死亡，提前停止余下 50K 和 100K 独立确认。", ""]
    lines += ["## 解释", "",
              "S1 只评价当前 PPO Top-5 落点；S2 仅在公开的高风险触发条件满足时，增加一层官方 BFS 搜索。两者均保留 PPO Top-1 回退。",
              "每局记录首次 100 次替换动作及可见状态边界；完整逐 seed 数据在 `runs/ppo30m_lookahead/`。",
              "推理使用 CPU、4 个独立 seed worker；运行中一次进程 RSS 快照约 3.4 GiB（含主进程和子进程，非峰值），模型没有占用 GPU 显存。"
              "表内耗时是模型决策与搜索时间，不包括 Core 落子和特征重建。各局 P95 的 P95 是跨局保守汇总，并非所有单步合并后的精确 P95。",
              "若 S2 与 S1 无显著差异，应先检查触发率及一层搜索是否区分根候选，不直接增加深度或蒸馏。", ""]
    failure = RUN / "diagnostics/S1_seed_990109.json"
    if failure.exists():
        record = json.loads(failure.read_text())
        changes = record["changes"]
        hold_changes = sum((x["alternatives"]["ppo"]["action"] >= 920) !=
                           (x["alternatives"]["search"]["action"] >= 920) for x in changes)
        fours = sum(x["alternatives"]["search"]["cleared"] == 4 for x in changes)
        lines += ["## 新增早死局诊断", "",
                  f"seed 990109 的 S1 在 {record['pieces']} 块结束。搜索共改写 {len(changes)} 次动作，其中 {hold_changes} 次改变 Hold，{fours} 次改写动作当步四消。末次改写在 {changes[-1]['step']} 块。",
                  "这 24 次替换显示一步价值过于奖励立即四消；但仅凭重放不能断定每次替换的因果影响。", ""]
    branch = RUN / "diagnostics/S1_branch_seed_990109_step_2849.json"
    if branch.exists():
        record = json.loads(branch.read_text())
        a, b = record["branches"]["ppo"], record["branches"]["search"]
        lines += ["### 同快照分支", "",
                  f"在第 {record['step']} 块选择前冻结完全相同的 Core 状态与 RNG；只将首动作改为 PPO #{record['first_actions']['ppo']} 或搜索 #{record['first_actions']['search']}，以后均由 S1 控制。"
                  f"搜索动作分支 {b['pieces']} 块后 Game Over；PPO 动作分支到 {a['pieces']} 块观察上限仍存活（右截断）。",
                  "这是该特定状态、首动作和后续 S1 控制条件下的反事实证据，不外推成所有四消动作有害。", ""]
    if status["stage"] == "completed_no_go_counterfactual":
        lines += ["## 后续判断", "",
                  "S2 在 20K 只触发约 0.8%，与 S1 生存完全相同，未修复关键早死；目前无依据继续加深搜索。"
                  "搜索策略也未通过安全门槛，不宜蒸馏到 PPO。下一轮应先改正对立即四消与可见风险的估值，"
                  "用同快照分支验证修正是否消除已知失败，同时保持 PPO Top-1 的长期布局；新规则必须在新开发种子上冻结验证。", "",
                  "本实验没有启动神经网络训练、没有修改 PPO/BC checkpoint、Core、计分或 Action Mask。队列已结束，独立 100K 确认未启动。", ""]
    output = ROOT / "reports/experiments/ppo30m_lookahead_comparison.md"
    output.write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    print(render())
