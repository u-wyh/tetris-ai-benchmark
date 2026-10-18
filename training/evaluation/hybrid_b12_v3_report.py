"""Paired staged report for the frozen V3-A intercept rule."""

import json
import statistics

import numpy as np

from training.train_ppo import ROOT, atomic_json

RUN = ROOT / "runs/hybrid_b12_v3"
REPORT = ROOT / "reports/experiments/hybrid_b12_v3.md"


def paired_ci(values, samples=10000):
    rng = np.random.default_rng(1431032)
    values = np.asarray(values,dtype=float)
    draws = rng.integers(0,len(values),size=(samples,len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def rows(phase,variant,seeds):
    return [json.loads((RUN/phase/variant/f"seed_{seed}.json").read_text()) for seed in seeds]


def common_score(a,b):
    index=min(a["pieces"],b["pieces"])//1000-1
    if index<0:
        return None
    return (a["score_milestones"][index]["score"],b["score_milestones"][index]["score"])


def write_report(protocol,outcome):
    phases=[("development_20k",protocol["dev"])]
    if (RUN/"development_100k_gate.json").exists():
        phases.append(("development_100k",protocol["dev"]))
    if outcome=="confirmation_completed":
        phases.append(("confirmation_100k",protocol["confirm"]))
    lines=["# Hybrid B12 V3 定向安全拦截实验","",
           "冻结 PPO 30M、BC Champion 和原 B12；V3-A 只在 B12 已选择 BC 的位置增加局部否决，不改变神经网络或 Core。开发种子 `1430000..1430015`，独立确认种子 `1431000..1431031`；达到上限按右截断处理。",
           f"模型 SHA256：PPO `{protocol['models']['ppo']}`，BC `{protocol['models']['bc']}`；V3 规则 SHA256 `{protocol['v3_rule_sha256']}`。",
           "","## 冻结规则及与旧方案的区别","",
           "V3-A 保留 B12 原触发条件。BC 已获 B12 许可时，使用公开棋盘分别模拟 PPO/BC 落点；如果 BC 落点的行转换数至少多 2，则拒绝该拦截。若两候选 Hold 用法不同，只在 BC 对可见下一块提供至少多 3 个安全落点时接受：安全落点要求不 Lock Out、最高列不超过 16 或两个首落点中更高者、洞数最多增加 1。PPO 当步 Lock Out 且 BC 不 Lock Out 时直接保留 BC 救急。",
           "C1/C2 改动拦截触发、要求更大的高度/中央改善，并用 Hold 变化或下一步原始合法动作数否决。V3-A 保持 B12 的触发和加权风险判断，另看行表面破碎程度与下一步**安全**落点；Hold 变化本身不构成否决。新增棋盘行转换为 O(200)，仅当 B12 拟用 BC 且 Hold 用法不同才搜索一次可见下一块的合法落点，约 O(合法落点 × 棋盘格)。不读 Bag、RNG 或隐藏 Next。",
           "旧 `1421007:69810` 的 Hold 变化但安全下一步收益有限、`1421013:9848` 和 `1421014:183` 的行转换增长只用于提出假设；`1421013:278` 的有益 BC 拦截并未增加行转换，也未改变 Hold。旧案例不是本轮验证数据。仅冻结 V3-A 一条新规则，未另造与 C2 重复的第二候选。",
           "","## 分阶段结果",""]
    summaries={}
    for phase,seeds in phases:
        summaries[phase]={v:json.loads((RUN/phase/v/"summary.json").read_text()) for v in protocol["variants"]}
        lines += [f"### {phase}","",
            "| 策略 | 平均存活 | P10 | 触顶 | <10K 死亡 | 平均得分（亿） | 四消占比 | BC 提议 | V3 拒绝 | 平均推理秒/局 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in protocol["variants"]:
            s=summaries[phase][v]
            lines.append(f"| {v} | {s['mean_pieces']:.0f} | {s['p10_pieces']:.0f} | {s['capped']}/{s['games']} | {s['early_under_10000']} | {s['mean_score']/1e8:.2f} | {s['four_fraction']:.1%} | {s['bc_proposed']} | {s['bc_rejected']} | {s['mean_inference_seconds']:.1f} |")
        lines.append("")
        if phase!="confirmation_100k":
            gate=json.loads((RUN/f"{phase}_gate.json").read_text())
            lines.append(f"冻结门槛：**{'通过' if gate['pass'] else '未通过'}**；逐项 {gate['checks']}。")
            lines.append("")
    final_phase=phases[-1][0]
    data={v:rows(final_phase,v,phases[-1][1]) for v in protocol["variants"]}
    v3,b12=data["V3A"],data["B12"]
    rejected=[e for row in v3 for e in row["intercepts"] if e["reason"].startswith("v3_")
              and e["reason"] not in ("v3_emergency_bc",)]
    reason_counts={}
    for e in rejected:
        reason_counts[e["reason"]]=reason_counts.get(e["reason"],0)+1
    lines += ["## 拦截行为", "",
              f"最终完成阶段 V3-A 拒绝 BC {sum(row['bc_rejected'] for row in v3)} 次；原因分布 {reason_counts}。对应事件保存在各局 `intercepts`，包含 PPO/BC 候选风险、行转换数、Hold 与下一步安全落点数。",
              "这些是明确、可重放的局部风险特征，不能单凭被拒绝次数证明长期生存提升。", ""]
    new_early=[a["seed"] for a,b in zip(v3,b12) if a["pieces"]<10000<=b["pieces"]]
    prevented_early=[a["seed"] for a,b in zip(v3,b12) if b["pieces"]<10000<=a["pieces"]]
    lines += [f"相对 B12 新增 <10K 早死 seed：{new_early}；避免 <10K 早死 seed：{prevented_early}。", ""]
    if outcome=="confirmation_completed":
        piece_d=[a["pieces"]-b["pieces"] for a,b in zip(v3,b12)]
        score_d=[a["score"]-b["score"] for a,b in zip(v3,b12)]
        common=[common_score(a,b) for a,b in zip(v3,b12)]
        common=[x for x in common if x is not None]
        ci_p=paired_ci(piece_d); ci_s=paired_ci(score_d)
        ratio=sum(a for a,_ in common)/max(1,sum(b for _,b in common))
        a,b=summaries[final_phase]["V3A"],summaries[final_phase]["B12"]
        checks={"p10":a["p10_pieces"]>b["p10_pieces"],
                "mean_survival":a["mean_pieces"]>b["mean_pieces"],
                "early_death":a["early_under_10000"]<=b["early_under_10000"],
                "score":a["mean_score"]>=.95*b["mean_score"],
                "four":a["four_fraction"]>=.9*b["four_fraction"],
                "paired_ci_positive":ci_p[0]>0}
        confirmed=all(checks.values())
        result={"checks":checks,"confirmed":confirmed,"paired_survival_mean":statistics.mean(piece_d),
                "paired_survival_ci95":ci_p,"paired_score_mean":statistics.mean(score_d),
                "paired_score_ci95":ci_s,"common_score_ratio":ratio}
        atomic_json(RUN/"confirmation_decision.json",result)
        lines += ["## 独立确认配对差异", "",
                  f"V3-A − B12：平均已观测存活差 {statistics.mean(piece_d):+.0f} 块，按 seed 配对 Bootstrap 95% 区间 [{ci_p[0]:+.0f}, {ci_p[1]:+.0f}]；平均总分差 {statistics.mean(score_d)/1e8:+.2f} 亿，区间 [{ci_s[0]/1e8:+.2f}, {ci_s[1]/1e8:+.2f}] 亿。共同存活区间累计得分比 {ratio:.3f}。",
                  f"独立确认：**{'通过' if confirmed else '未通过'}**；预设判断 {checks}。均值差区间包含零或任一指标未过门槛时，不宣称稳定改善。触顶是右截断。", ""]
        lines += ["| Seed | B12 存活/亿分 | V3-A 存活/亿分 | 差值块数 | V3 拒绝次数 |",
                  "|---:|---:|---:|---:|---:|"]
        for a,b in zip(v3,b12):
            lines.append(f"| {a['seed']} | {b['pieces']}{'+' if b['capped'] else ''}/{b['score']/1e8:.2f} | {a['pieces']}{'+' if a['capped'] else ''}/{a['score']/1e8:.2f} | {a['pieces']-b['pieces']:+d} | {a['bc_rejected']} |")
        lines.append("")
    else:
        confirmed=False
        lines += ["## 独立确认", "",
                  "开发组未通过预冻结门槛，确认种子完全未运行，也没有据此调参或更换 checkpoint。", ""]
    lines += ["## 结论", ""]
    if confirmed:
        lines.append("V3-A 在本轮预定开发与全新独立确认协议下达到综合门槛，值得作为 B12 的替代候选继续长局和网页适配验证；本次不修改网页默认 AI。")
    else:
        lines.append("V3-A 未满足本轮全部预冻结成功条件，保留原 B12。停止本轮，不自动启动 V4 或神经网络训练。")
    lines.append("本研究只测试一条冻结规则；不同策略提前死亡使平均总分与每块效率不可互换，且 100K 触顶不代表真实寿命。")
    REPORT.write_text("\n".join(lines)+"\n")
    return {"outcome":outcome,"confirmed":confirmed,"phases":[p for p,_ in phases]}
