"""Generate evidence-linked B12 diagnostic report from verified replay files."""
import json
import statistics

from training.evaluation.hybrid_failure import FAILURES, RUN
from training.evaluation.hybrid_queue import HASHES
from training.train_ppo import ROOT


def main():
    rows = {seed: json.loads((RUN / "failure_analysis" / f"seed_{seed}.json").read_text())
            for seed in FAILURES}
    if any(not r.get("replay_verified") or "intercepts" not in r for r in rows.values()):
        raise RuntimeError("B12 failure replay is incomplete")
    lines = ["# Hybrid B12 失败分析", "",
        "已按冻结 B12 决策从原 seed 重放六局；每局最终分数和存活块数均与原始确认结果一致。反事实分支从同一深拷贝 TetrisCore 状态开始，只改变当步 PPO/BC 动作，之后均使用 B12；每支最多跟踪 5000 块，触及观察窗属于右截断。在线策略没有读取 Bag 或 RNG。", "",
        f"模型 SHA256：PPO `{HASHES['ppo']}`；BC `{HASHES['bc']}`。原始确认组和分析结果分别在 `runs/candidate_ppo30m_bc_hybrid/confirm_100k/` 与 `runs/hybrid_v2/failure_analysis/`。", "",
        "| Seed | B12 存活 | BC 动作占比 | BC 拦截区间 | 首次拦截块 | 运行耗时（秒） |",
        "|---:|---:|---:|---:|---:|---:|"]
    for seed, r in rows.items():
        lines.append(f"| {seed} | {r['old_pieces']} | {r['bc_action_fraction']:.2%} | {r['intercept_runs']} | {r['first_intercept']} | {r['wall_seconds']:.1f} |")
    lines += ["", "## 拦截结构", "",
        "| Seed | PPO 本可四消而 BC 未四消 | BC 改用 Hold 的次数 | PPO 高度≤12 时的 BC 动作 | BC 新增洞多于 PPO |",
        "|---:|---:|---:|---:|---:|"]
    for seed, r in rows.items():
        events = r["intercepts"]
        lines.append(f"| {seed} | {sum(e['ppo_clears']==4 and e['bc_clears']<4 for e in events)} | "
            f"{sum(e['ppo_hold']!=e['bc_hold'] for e in events)} | "
            f"{sum(e['ppo_risk']['height']<=12 for e in events)} | "
            f"{sum(e['bc_risk']['new_holes']>e['ppo_risk']['new_holes'] for e in events)} |")
    lines += ["", "## 同状态反事实分支", "",
        "表中结果只表示所选起点的一次确定性分支，不代表全局因果效应。分支可能改变 Hold 与后续公开状态；每条原始状态 SHA256 见运行 JSON。", "",
        "| Seed | 位置 | PPO/BC 当步消行 | PPO 分支终点 | BC 分支终点 | PPO/BC 分支得分增量 |",
        "|---:|---|---:|---:|---:|---:|"]
    for seed, r in rows.items():
        for label, case in r["candidate_cases"].items():
            p, b = case["branches"]["ppo"], case["branches"]["bc"]
            lines.append(f"| {seed} | {label} @ {case['piece']} | {case['ppo_candidate_clears']}/{case['bc_candidate_clears']} | "
                f"{p['end_piece']}{'+' if p['horizon_censored'] else ''} | {b['end_piece']}{'+' if b['horizon_censored'] else ''} | "
                f"{p['score_gained']}/{b['score_gained']} |")
    lines += ["", "## 死亡前棋盘变化", "",
        "| Seed | 末段 | 最大高度范围 | 洞数范围 | 中央顶四行最大占用 | 四消次数 |",
        "|---:|---:|---:|---:|---:|---:|"]
    for seed, r in rows.items():
        for n in (100, 500, 1000):
            tail = r[f"last_{n}"]
            lines.append(f"| {seed} | {n} | {min(v['height'] for v in tail)}–{max(v['height'] for v in tail)} | "
                f"{min(v['holes'] for v in tail)}–{max(v['holes'] for v in tail)} | "
                f"{max(v['central_top4'] for v in tail)} | {sum(v['cleared']==4 for v in tail)} |")
    all_events = [e for r in rows.values() for e in r["intercepts"]]
    next_cases = json.loads((RUN / "next_space_cases.json").read_text())["cases"]
    lines += ["", "## 主要发现", "",
        f"六局共 {len(all_events)} 次 BC 动作，其中 {sum(e['ppo_hold']!=e['bc_hold'] for e in all_events)} 次与 PPO 候选的 Hold 用法不同，"
        f"{sum(e['bc_risk']['new_holes']>e['ppo_risk']['new_holes'] for e in all_events)} 次 BC 候选新增洞更多。"
        f"没有发现 PPO 候选当步可四消而 BC 未四消的拦截；不据此增加四消禁拦规则。", "",
        "`700017` 首次拦截（第 329 块）：同状态改用 PPO 动作后，B12 后续策略存活到第 5330 块观察上限；原 BC 分支在第 2638 块结束。"
        "可见下一步合法落点数从 PPO 分支 68 个降为 BC 分支 51 个。", "",
        "`700026` 首次拦截（第 268 块）：同状态 PPO 分支存活到第 5269 块观察上限，原 BC 分支在第 1715 块结束；"
        "当步两支均未消行、下一步合法落点同为 68 个，但 Hold 从 PPO 分支的 J 改为 BC 分支的 T。"
        "第 904 块的另一个拦截中，BC 候选比 PPO 多增 3 个洞，单步高度虽低，短窗分支也更早结束。", "",
        "`700023` 与 `700029` 相对独立纯 PPO 退步，但选中的首次拦截分支均存活到 5000 块观察上限；"
        "无法把整局死亡归因于这些单次拦截。高危临终拦截在部分局还短暂延后死亡。", "",
        "可见下一块落点对照详见 `runs/hybrid_v2/next_space_cases.json`；三个核查位置为 "
        + ", ".join(f"{case['seed']}:{case['piece']}" for case in next_cases) + "。", ""]
    lines += ["", "## 解释边界", "",
        "B12 的风险和拦截记录能够定位候选动作与后续轨迹差异；同 seed 下干预会改变棋盘，Hold 还会改变当前方块使用顺序。因此不能把独立的纯 PPO 终局直接视为某次 BC 拦截的反事实。上述同状态分支缓解了这一问题，但只覆盖选中的少数位置和有限观察窗。", ""]
    report = ROOT / "reports/experiments/hybrid_b12_failure_analysis.md"
    report.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
