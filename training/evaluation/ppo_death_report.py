"""Write descriptive death and bounded counterfactual reports from sealed artifacts."""
import json
from collections import Counter

from training.evaluation.hybrid_queue import HASHES, digest
from training.evaluation.ppo_death_capture import RUN
from training.evaluation.ppo_death_counterfactual import _branch_path, selection
from training.train_ppo import ROOT, atomic_json, now

REPORTS = ROOT / "reports/experiments"


def _read_branch(seed, step, action, horizon):
    path = _branch_path(seed, step, action, horizon)
    seal = json.loads(path.with_suffix(".sha256.json").read_text())
    if digest(path) != seal["sha256"]:
        raise RuntimeError(f"Counterfactual result changed: {path}")
    return json.loads(path.read_text())


def compare(base, alt):
    """Finite-window evidence; never infer survival beyond a right-censored branch."""
    common = min(base["pieces"], alt["pieces"])
    if common < 1:
        raise ValueError("Empty branch")
    base_score = base["scores"][common-1] - base["initial_score"]
    alt_score = alt["scores"][common-1] - alt["initial_score"]
    base_fours = base["four_counts"][common-1]
    alt_fours = alt["four_counts"][common-1]
    score_ratio = alt_score/max(1, base_score)
    four_ratio = alt_fours/max(1, base_fours)
    strong = (base["game_over"] and alt["pieces"] >= base["pieces"]+250 and
              alt["pieces"] >= 1.5*base["pieces"] and score_ratio >= .95 and
              (base_fours < 3 or four_ratio >= .9))
    return {"common_pieces": common, "base_common_score": base_score,
            "alt_common_score": alt_score, "score_ratio": score_ratio,
            "base_common_fours": base_fours, "alt_common_fours": alt_fours,
            "four_ratio": four_ratio, "survival_gain_observed": alt["pieces"]-base["pieces"],
            "strong_local_correction": strong,
            "indeterminate_beyond_window": bool(base["censored"] and alt["censored"])}


def summarize_counterfactual():
    selected = selection()["states"]
    states = []
    branch_count = Counter()
    worker_seconds = 0.0
    for item in selected:
        seed, step = item["seed"], item["step"]
        candidate = json.loads((RUN / "counterfactual_candidates" /
            f"seed_{seed}_step_{step}.json").read_text())
        base_action = candidate["ppo_action"]
        initial = {c["action"]: _read_branch(seed,step,c["action"],1000)
                   for c in candidate["choices"]}
        base = initial[base_action]
        comparisons = []
        for choice in candidate["choices"]:
            action = choice["action"]
            result = initial[action]
            branch_count[1000] += 1
            worker_seconds += result["wall_seconds"]
            if action != base_action:
                comparisons.append({"choice": choice, "horizon": 1000,
                                    "result": compare(base,result),
                                    "alt_pieces": result["pieces"],
                                    "alt_game_over": result["game_over"]})
        extended = {}
        for choice in candidate["choices"]:
            path = _branch_path(seed,step,choice["action"],5000)
            if path.exists():
                extended[choice["action"]] = _read_branch(seed,step,choice["action"],5000)
                branch_count[5000] += 1
                worker_seconds += extended[choice["action"]]["wall_seconds"]
        if base_action in extended:
            for choice in candidate["choices"]:
                action = choice["action"]
                if action != base_action and action in extended:
                    comparisons.append({"choice": choice, "horizon": 5000,
                                        "result": compare(extended[base_action],extended[action]),
                                        "alt_pieces": extended[action]["pieces"],
                                        "alt_game_over": extended[action]["game_over"]})
        states.append({"seed": seed, "step": step, "death_distance": item["death_distance"],
                       "tier": item["tier"], "candidate_count": len(candidate["choices"]),
                       "baseline_1000_pieces": base["pieces"],
                       "baseline_1000_game_over": base["game_over"],
                       "baseline_5000_pieces": extended.get(base_action,{}).get("pieces"),
                       "baseline_5000_game_over": extended.get(base_action,{}).get("game_over"),
                       "comparisons": comparisons})
    by_tier = {}
    for tier in ("1-100","101-500","501-1000","1001-5000"):
        group = [s for s in states if s["tier"] == tier]
        by_tier[tier] = {"states": len(group),
                         "strong_1000_states": sum(any(c["horizon"]==1000 and c["result"]["strong_local_correction"]
                                                       for c in s["comparisons"]) for s in group),
                         "strong_5000_states": sum(any(c["horizon"]==5000 and c["result"]["strong_local_correction"]
                                                       for c in s["comparisons"]) for s in group)}
    strong = [(s,c) for s in states for c in s["comparisons"] if c["result"]["strong_local_correction"]]
    summary = {"generated_at": now(), "ppo_sha256": HASHES["ppo"], "bc_sha256": HASHES["bc"],
               "states": len(states), "branches_1000": branch_count[1000],
               "branches_5000": branch_count[5000],
               "cpu_worker_hours": worker_seconds/3600,
               "strong_1000_comparisons": sum(c["horizon"]==1000 for _,c in strong),
               "strong_5000_comparisons": sum(c["horizon"]==5000 for _,c in strong),
               "strong_bc_comparisons": sum("bc" in c["choice"]["sources"] for _,c in strong),
               "strong_ppo_top5_comparisons": sum(any(source.startswith("ppo_top") for source in c["choice"]["sources"])
                                                   for _,c in strong),
               "by_tier": by_tier, "state_rows": states}
    atomic_json(RUN / "counterfactual_summary.json", summary)
    return summary


def write_reports():
    analysis = json.loads((RUN / "death_analysis.json").read_text())
    capture = json.loads((RUN / "capture_completed.json").read_text())
    counter = summarize_counterfactual()
    deaths = analysis["death_episodes"]
    counts = Counter(label for row in deaths for label in row["labels"])
    lines = ["# PPO 30M 长期死亡机制：实测结果", "",
             f"生成时间：{now()}。冻结 PPO SHA256：`{HASHES['ppo']}`。",
             f"新诊断种子：{analysis['episode_count']}；真正死亡：{analysis['death_count']}；100K 右截断：{analysis['right_censored_count']}。",
             f"平均已观测存活 {analysis['mean_pieces']:.0f} 块，P10 {analysis['p10_pieces']:.0f} 块。触顶局的真实寿命未知。", "",
             "## 死亡前结构变化", "",
             "以下按 episode 统计；标签可重叠，仅为探索性关联，不是因果证明。", "",
             "| 类型 | 独立死亡 seed 数 |", "|---|---:|"]
    for label,count in sorted(counts.items()):
        lines.append(f"| {label} | {count} |")
    lines += ["", "| Seed | 死亡块数 | 得分 | 最后 5000 块最高列变化 | 洞数变化 | 四消数 | 匹配存活对照 | 标签 |",
              "|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in deaths:
        w = row["windows"]["5000"]
        match = row["match"]
        lines.append(f"| {row['seed']} | {row['pieces']} | {row['score']} | {w['max_height_start']}→{w['max_height_end']} | "
                     f"{w['holes_start']}→{w['holes_end']} | {w['four_count']} | "
                     f"{match['seed'] if match else '无'} | {', '.join(row['labels'])} |")
    lines += ["", "详细窗口（5000、1000、500、100、20）、对照状态与数据哈希："
              f"`runs/ppo30m_death_analysis/death_analysis.json`。图位于 `runs/ppo30m_death_analysis/figures/`。",
              "最高列、洞数或 Hold 变化不能单独证明其导致死亡；对照只匹配观测变量。", ""]
    (REPORTS / "ppo30m_death_mechanism_results.md").write_text("\n".join(lines))
    lines = ["# PPO 30M 单动作反事实研究", "",
             f"生成时间：{now()}。状态 {counter['states']} 个；1000 块分支 {counter['branches_1000']} 个；5000 块分支 {counter['branches_5000']} 个。",
             f"分支计算约 {counter['cpu_worker_hours']:.1f} CPU 小时；只替换首动作，之后固定冻结 PPO 续控。", "",
             "强局部纠错预设标准：原动作分支真实死亡；替代分支至少多活 250 块且达原分支 1.5 倍；共同存活期间得分不少于 95%，若原分支至少 3 次四消则四消不少于 90%。", "",
             "| 死亡距离层 | 状态数 | 1000 块窗口发现强纠错的状态 | 5000 块窗口发现强纠错的状态 |",
             "|---|---:|---:|---:|"]
    for tier,row in counter["by_tier"].items():
        lines.append(f"| {tier} | {row['states']} | {row['strong_1000_states']} | {row['strong_5000_states']} |")
    lines += ["", f"强纠错比较：1000 块 {counter['strong_1000_comparisons']} 个，5000 块 {counter['strong_5000_comparisons']} 个；"
              f"其中 BC 来源 {counter['strong_bc_comparisons']} 个，PPO Top-K 来源 {counter['strong_ppo_top5_comparisons']} 个。"
              "来源可重叠，不应相加。", "",
              "| Seed | 死亡距离 | 首动作 PPO 排名 | 来源 | 窗口 | 原分支块数 | 替代分支块数 | 共同得分比 | 共同四消比 | 证据 |",
              "|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for state in counter["state_rows"]:
        for comp in state["comparisons"]:
            c, result = comp["choice"], comp["result"]
            if result["strong_local_correction"] or comp["horizon"]==5000:
                lines.append(f"| {state['seed']} | {state['death_distance']} | {c['ppo_rank']} | "
                             f"{','.join(c['sources'])} | {comp['horizon']} | "
                             f"{state['baseline_1000_pieces'] if comp['horizon']==1000 else state['baseline_5000_pieces']} | "
                             f"{comp['alt_pieces']} | {result['score_ratio']:.3f} | {result['four_ratio']:.3f} | "
                             f"{'强局部纠错' if result['strong_local_correction'] else '未达标准'} |")
    lines += ["", "完整逐 seed、逐状态和右截断结果：`runs/ppo30m_death_analysis/counterfactual_summary.json`。"
              "两个分支都触顶时，只能认为本窗口内无法区分更长期寿命。Hold 动作会改变后续方块经历，已在分支文件中记录。", "",
              "## 下一步判断", ""]
    independent = len({s["seed"] for s in counter["state_rows"] for c in s["comparisons"]
                       if c["result"]["strong_local_correction"]})
    lines.append(f"满足预设纠错标准的独立 seed 为 {independent}/{analysis['death_count']}；"
                 f"探索性结构标签跨 seed 分布为 {dict(counts)}。单步可纠错比例仅限本候选集合和固定 PPO 续控。")
    if independent >= 3 and independent >= analysis["death_count"]/2:
        lines.append("当前证据更支持部分死亡由关键动作排序失误触发。下一步只建议研究基于反事实排名的定向纠错学习；本轮未训练。")
    elif independent >= 3:
        lines.append("局部失误和长期结构退化可能并存，目前不能断定哪一种占主导。下一步只建议研究关键动作出现前的长期布局退化；本轮未训练。")
    else:
        lines.append("当前证据不足以把早死归因为普遍的局部动作排序错误；下一步只建议研究更早期的布局退化与长期信用分配，本轮未训练。")
    lines.append("候选动作有限、续控固定为 PPO、观察窗口有限；结论不能推广为全动作最优或整体策略提升。")
    (REPORTS / "ppo30m_death_counterfactual_analysis.md").write_text("\n".join(lines)+"\n")
    return counter
