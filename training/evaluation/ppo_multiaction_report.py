"""Summarize the frozen pilot using same-state, same-grader BC comparisons."""
import json
from collections import Counter

from training.evaluation.ppo_multiaction_counterfactual import RUN, VERSION, freeze_selection
from training.evaluation.hybrid_queue import HASHES
from training.evaluation.local_correction_active_v2 import V2, _row_hash
from training.imitation_dataset import digest
from training.train_ppo import ROOT, atomic_json, now

REPORTS = ROOT / "reports/experiments"


def main():
    selection = freeze_selection()["states"]
    states, options, branches, hashes = [], [], [], {}
    selection_sha = digest(RUN/"selection.json")
    for item in selection:
        seed, step = item["seed"], item["step"]
        state_file = RUN / "states" / f"seed_{seed}_step_{step}.json"
        if not state_file.exists():
            continue
        state = json.loads(state_file.read_text())
        candidate = json.loads((RUN / "candidates" / f"seed_{seed}_step_{step}.json").read_text())
        candidate_file = RUN/"candidates"/f"seed_{seed}_step_{step}.json"
        candidate_sha = digest(candidate_file)
        source = json.loads((V2/"branches"/f"seed_{seed}"/f"step_{step}.json").read_text())["row"]
        if (state["version"] != VERSION or candidate["version"] != VERSION or
            state["models"] != HASHES or candidate["models"] != HASHES or
            state["selection_sha256"] != selection_sha or candidate["selection_sha256"] != selection_sha or
            state["candidate_sha256"] != candidate_sha or
            state["state_sha256"] != item["state_sha256"] or candidate["state_sha256"] != item["state_sha256"] or
            _row_hash(source) != item["row_sha256"]):
            raise RuntimeError("Report input version/hash mismatch")
        legal = set(source["legal_ids"])
        if len(candidate["options"]) != len({o["action"] for o in candidate["options"]}) or any(o["action"] not in legal for o in candidate["options"]):
            raise RuntimeError("Candidate uniqueness or legal action mismatch")
        hashes[str(state_file.relative_to(RUN))] = digest(state_file)
        hashes[str(candidate_file.relative_to(RUN))] = candidate_sha
        states.append(state)
        for option in candidate["options"]:
            options.append((state, option))
            path = RUN / "branches" / f"seed_{seed}_step_{step}_action_{option['action']}.json"
            saved = json.loads(path.read_text())
            if (saved["version"] != VERSION or saved["models"] != HASHES or
                saved["seed"] != seed or saved["step"] != step or saved["action"] != option["action"] or
                saved["state_sha256"] != item["state_sha256"] or saved["selection_sha256"] != selection_sha or
                saved["candidate_sha256"] != candidate_sha or
                saved["branch"]["pieces"] != len(saved["branch"]["scores"]) or
                saved["branch"]["pieces"] != len(saved["branch"]["fours"])):
                raise RuntimeError(f"Branch identity/content mismatch: {path}")
            branches.append(saved["branch"])
            hashes[str(path.relative_to(RUN))] = digest(path)
    if len(states) != len(selection):
        raise RuntimeError(f"Incomplete pilot: {len(states)}/{len(selection)}")
    atomic_json(RUN/"dataset_manifest.json", {"version": VERSION, "models": HASHES,
                "selection_sha256": selection_sha, "file_sha256": hashes})
    grades = [(s, o, s["grades"].get(str(o["action"]))) for s,o in options]
    alt = [(s,o,g) for s,o,g in grades if g]
    a = [(s,o,g) for s,o,g in alt if g["category"] == "A"]
    bc = [(s,o,g) for s,o,g in alt if "bc" in o["sources"]]
    bc_a = [x for x in bc if x[2]["category"] == "A"]
    found = {(s["seed"],s["step"]) for s,o,g in a}
    found_bc = {(s["seed"],s["step"]) for s,o,g in bc_a}
    non_bc = [x for x in a if "bc" not in x[1]["sources"]]
    hours = sum(b["elapsed_seconds"] for b in branches)/3600
    bc_hours = sum(b["elapsed_seconds"] for s,o in options if "bc" in o["sources"] and "ppo_top1" not in o["sources"] for b in [json.loads((RUN/"branches"/f"seed_{s['seed']}_step_{s['step']}_action_{o['action']}.json").read_text())["branch"]])/3600
    base_hours = sum(b["elapsed_seconds"] for s,o in options if "ppo_top1" in o["sources"] for b in [json.loads((RUN/"branches"/f"seed_{s['seed']}_step_{s['step']}_action_{o['action']}.json").read_text())["branch"]])/3600
    summary = {"version": VERSION, "completed_at": now(), "states": len(states),
               "tiers": dict(Counter(s["tier"] for s in states)), "seeds": len(set(s["seed"] for s in states)),
               "candidate_actions": len(options), "alternative_actions": len(alt),
               "A_actions": len(a), "A_states": len(found), "BC_A_actions": len(bc_a),
               "BC_A_states": len(found_bc), "non_BC_A_actions": len(non_bc),
               "non_BC_A_states": len({(s["seed"],s["step"]) for s,o,g in non_bc}),
               "A_top5_actions": sum("ppo_top5" in o["sources"] for _,o,_ in a),
               "A_four_preserving": sum(g["four_ok"] and g["well_preserved"] for _,_,g in a),
               "A_with_observed_ppo_fours_at_least_2": sum(g["ppo_fours"] >= 2 for _,_,g in a),
               "A_common_score_ratio_mean": sum(g["score_ratio"] for _,_,g in a)/max(1,len(a)),
               "A_survival_gain_mean": sum(g["survival_delta"] for _,_,g in a)/max(1,len(a)),
               "A_hold_changed": sum(not g["future_same"] for _,_,g in a),
               "B_actions": sum(g["category"] == "B" for _,_,g in alt),
               "B_future_changed": sum(g["category"] == "B" and not g["future_same"] for _,_,g in alt),
               "A_by_tier": dict(Counter(s["tier"] for s,o,g in a)),
               "BC_alternatives": len(bc), "BC_A_per_alternative": len(bc_a)/max(1,len(bc)),
               "multi_A_per_alternative": len(a)/max(1,len(alt)),
               "BC_A_per_state": len(found_bc)/len(states), "multi_A_per_state": len(found)/len(states),
               "branch_cpu_hours": hours, "BC_comparison_cpu_hours": base_hours+bc_hours,
               "multi_A_per_cpu_hour": len(found)/max(hours,1e-9),
               "BC_A_per_cpu_hour": len(found_bc)/max(base_hours+bc_hours,1e-9),
               "A_sources": dict(Counter(source for _,o,_ in a for source in o["sources"])),
               "A_seeds": dict(Counter(s["seed"] for s,o,g in a)),
               "A_rank": dict(Counter(o["ppo_rank"] for _,o,_ in a)),
               "A_mean_ppo_probability": sum(o["ppo_probability"] for _,o,_ in a)/max(1,len(a))}
    summary["A_states_without_BC_A"] = len(found-found_bc)
    summary["A_four_clear_ratio_mean"] = sum(g["candidate_fours"]/max(1,g["ppo_fours"]) for _,_,g in a)/max(1,len(a))
    summary["A_common_candidate_score_per_1000_mean"] = sum(1000*g["candidate_score"]/g["common_pieces"] for _,_,g in a)/max(1,len(a))
    summary["A_common_ppo_score_per_1000_mean"] = sum(1000*g["ppo_score"]/g["common_pieces"] for _,_,g in a)/max(1,len(a))
    summary["expand"] = len(non_bc) >= 3 and len({s["seed"] for s,o,g in non_bc}) >= 2 and summary["multi_A_per_cpu_hour"] >= .8*summary["BC_A_per_cpu_hour"]
    atomic_json(RUN/"summary.json", summary)
    lines = ["# PPO 30M 多候选反事实试点结果", "", f"完成时间：{summary['completed_at']}。协议：`{VERSION}`。", "",
             f"共 {summary['states']} 个训练组状态、{summary['seeds']} 个 seed、{summary['candidate_actions']} 个唯一候选动作（含 PPO 基线）。分层：{summary['tiers']}。", "",
             "| 同起点、同一分级规则 | PPO vs BC | 多候选 |", "|---|---:|---:|",
             f"| A 类状态 | {summary['BC_A_states']} | {summary['A_states']} |",
             f"| A 类状态占比 | {summary['BC_A_per_state']:.1%} | {summary['multi_A_per_state']:.1%} |",
             f"| A 类动作 / 替代分支 | {summary['BC_A_per_alternative']:.1%} | {summary['multi_A_per_alternative']:.1%} |",
             f"| 分支 CPU 小时 | {summary['BC_comparison_cpu_hours']:.2f} | {summary['branch_cpu_hours']:.2f} |",
             f"| A 类状态 / CPU 小时 | {summary['BC_A_per_cpu_hour']:.2f} | {summary['multi_A_per_cpu_hour']:.2f} |", "",
             f"A 类动作 {summary['A_actions']} 个；非 BC 来源 {summary['non_BC_A_actions']} 个，Top5 来源 {summary['A_top5_actions']} 个；有 {summary['A_states_without_BC_A']} 个状态只有其他候选找到 A。满足四消和深井保持门槛 {summary['A_four_preserving']} 个，其中共同窗口 PPO 至少有两次四消的证据 {summary['A_with_observed_ppo_fours_at_least_2']} 个。A 类平均共同得分比 {summary['A_common_score_ratio_mean']:.3f}，平均四消次数比 {summary['A_four_clear_ratio_mean']:.3f}，平均多活 {summary['A_survival_gain_mean']:.1f} 块。来源交叉计数：{summary['A_sources']}。", "",
             f"在各 A 动作与 PPO 的共同存活窗口内，候选平均每千块得分 {summary['A_common_candidate_score_per_1000_mean']:,.0f}，PPO 对照 {summary['A_common_ppo_score_per_1000_mean']:,.0f}；这是逐动作比率的均值，不代表完整对局平均分。", "",
             f"A 类按 seed：{summary['A_seeds']}；按死亡距离层：{summary['A_by_tier']}；按 PPO 排名：{summary['A_rank']}；平均 PPO 原概率：{summary['A_mean_ppo_probability']:.4f}。B 类动作 {summary['B_actions']} 个，其中改变公开后续方块状态 {summary['B_future_changed']} 个。", "",
             f"第二批自动门槛：{'达到' if summary['expand'] else '未达到'}。多候选每 CPU 小时获得 A 类状态 {summary['multi_A_per_cpu_hour']:.2f}，低于 BC 对照 {summary['BC_A_per_cpu_hour']:.2f}；虽然多找到 {summary['A_states_without_BC_A']} 个 BC 没找到的状态，当前规则的计算效率不足以触发第二批。本轮停止扩大，也不启动训练。下一步可先研究 PPO Top5 的廉价危险状态重排序，再使用新种子验证；不能据离线分支宣称在线模型已改善。", "",
             "## 有 A 类动作的状态", "", "| Seed | 步数 | 距死亡层 | A 动作数 | BC 是否 A | 最优 A 的 PPO 排名 |", "|---:|---:|---|---:|---|---:|"]
    for state in states:
        hits = [(o,g) for s,o,g in a if s["seed"] == state["seed"] and s["step"] == state["step"]]
        if not hits: continue
        best = min(hits, key=lambda x:(x[0]["ppo_rank"],-x[1]["survival_delta"]))
        lines.append(f"| {state['seed']} | {state['step']} | {state['tier']} | {len(hits)} | {'是' if any('bc' in o['sources'] for o,g in hits) else '否'} | {best[0]['ppo_rank']} |")
    lines += ["", "限制：所有结果仅是 1000 块内、冻结 PPO 续控的离线反事实；按观察上限右截断。短窗提前停止的候选不能用于断言长期无益。CPU 小时是各分支实测耗时之和，不含已存在轨迹采集。改变 Hold 后的可见后续方块状态作为 B 类处理。", ""]
    (REPORTS/"ppo_multiaction_counterfactual_results.md").write_text("\n".join(lines))
    ds = ["# 多候选纠错数据说明", "", f"协议：`{VERSION}`。原始结果位于 `{RUN}`；每条候选与分支独立保存。", "",
          f"{summary['states']} 个状态、{summary['candidate_actions']} 个唯一动作、{summary['A_actions']} 个 A 类动作。`dataset_manifest.json` 保存全部状态、候选及分支结果的 SHA256。仅 A 类可考虑未来监督样本；B/C 保留为分析与对照，不自动加入训练。", "",
          "输入仅允许公开棋盘、Hold、Next 3、正式合法动作和现有 Candidate 特征。死亡距离、反事实未来和 RNG 只用于离线选样及分级。", "",
          f"A 类中 {summary['A_top5_actions']} 个来自 PPO Top5，说明部分可行替代已经在原 Actor 的高概率动作内；其余候选仍需公开风险或深井特征。A 类仅覆盖 {len(summary['A_seeds'])} 个 seed，且同一死亡轨迹上的多个状态并非独立样本。A 类均未改变可见后续方块状态；B 类有 {summary['B_future_changed']} 个发生变化。", "",
          "若以后建模，可先尝试只在危险状态重排序 PPO Top5，并保留原 PPO Top1 的正常状态监督及四消约束。需要额外特征时，仅加入公开棋盘的深井形状、出生区风险和可见下一块合法空间；不使用死亡倒计时、隐藏 Bag/RNG 或反事实未来。下一阶段须先在未参与本试点的新种子上验证数据质量。本次没有训练新模型。", ""]
    (REPORTS/"ppo_multiaction_dataset.md").write_text("\n".join(ds))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
