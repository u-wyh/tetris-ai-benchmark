"""Summarize the sealed autonomous experiment without changing its protocol."""

import json
from pathlib import Path

from training.evaluation.hybrid_queue import PPO
from training.train_ppo import ROOT

RUN = ROOT / "runs/autonomous_rl_research"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    status = read(RUN / "status.json") or {}
    train = read(RUN / "training/transaction_state.json") or {}
    steps = train.get("committed_steps", 0)
    task = train.get("committed_task", 0)
    model = RUN / f"training/committed/task_{task:06d}/model.zip"
    latest = str(model.relative_to(ROOT)) if model.exists() else "当前事务模型未保留；见训练 manifest"
    blocks = []
    for label in ("smoke", "dev_20k", "dev_100k", "confirm_100k"):
        summary = read(RUN / "evaluation" / label / "summary.json")
        if summary:
            blocks += [f"\n## {label}\n",
                       "| 模型 | 平均已观测存活 | P10 | 触顶 | <10K | 平均正式分数 | 四消占比 |",
                       "|---|---:|---:|---:|---:|---:|---:|"]
            for name in ("ppo", "B12", "bc", "candidate"):
                r = summary[name]
                early = r["early_under_10000"] if label != "smoke" else "不适用"
                blocks.append(f"| {name} | {r['mean_pieces']:.1f} | {r['p10_pieces']:.1f} | "
                              f"{r['capped']}/{r['games']} | {early} | "
                              f"{r['mean_score']:.0f} | {r['four_fraction']:.1%} |")
            blocks.append(f"\n候选相对 PPO 的共同存活得分比："
                          f"{summary['candidate'].get('common_score_ratio_to_ppo')}。"
                          " 触顶按右截断，表内是已观测存活。")
    final = ROOT / "reports/experiments/autonomous_rl_final_summary.md"
    final.write_text("\n".join([
        "# 自主强化学习最终汇总", "",
        f"队列状态：`{status.get('stage', 'unknown')}`。已提交训练步数：**{steps:,}**，"
        f"共 {task} 个 4096 步事务；最后提交的新模型路径：`{latest}`。",
        "", "## V2 与发布", "",
        "历史 V2 哈希对应 `0598b9f`；后续差异只在 V2 算法评分与去重加速。"
        "Core 和评测器未变；三个 seed 各 200 步的动作/得分/消行成为固定语义回归，"
        "额外十个历史 seed 各 1000 步动作重放一致。该修复仅在研究分支，"
        "尚未合入 main；按任务要求未创建 v1.0.0 标签。",
        "两份模型在本机有正确 SHA256，但未核验到异机或独立物理介质备份；"
        "同 NVMe 的 `/data` 不能当独立备份。", "",
        "## 研究结论", "",
        "本轮选择延长 PPO 折扣与 GAE 并保留 KL 锚定，未修改奖励和游戏规则。"
        "阶段是否通过以预冻结门槛和逐 seed 结果为准；状态为停止或失败时，"
        "不得从烟测推断长期提升。", *blocks, "",
        "若候选四消占比下降超过预设 3 个百分点，按开发门槛停止；"
        "独立确认组未运行时不能宣称泛化成功。", "",
        "## 恢复与下一步", "",
        "事务状态、配置、模型、优化器、RNG、worker 状态以及逐 seed 原子结果位于 "
        "`runs/autonomous_rl_research/`。若队列因异常中止，在源码与模型哈希不变且磁盘、"
        "网页服务正常时，重新运行 `.venv/bin/python -m training.autonomous_rl_queue`；"
        "协议不符会拒绝恢复；后续工程优化已改变研究源码，若确需恢复原训练，"
        "先检出冻结提交 `17d0025` 并核验 `protocol.json`。研究设计与预设门槛见 "
        "[设计](autonomous_rl_design.md)，完整阶段报告见 "
        "[结果](autonomous_rl_results.md)。若训练路线未达门槛，下一优先方向是量化并"
        "复用训练环境的合法落点缓存，以降低 CPU rollout 成本；须先做同动作/同分数"
        "回归和配对吞吐基准，再考虑集成。", "",
        f"冻结 PPO 文件：`{PPO.relative_to(ROOT)}`。", ""
    ]))


if __name__ == "__main__":
    main()
