"""Generate cautious, reproducible reports from committed local-correction artifacts."""
import json
from pathlib import Path

from training.local_correction_data import RUN, TRAIN, VALIDATION, DEV, CONFIRM, HORIZON
from training.train_ppo import ROOT

REPORTS = ROOT / "reports/experiments"


def _read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _summary(phase, variant):
    return _read(RUN / "evaluations" / phase / variant / "summary.json")


def _table(phase, variants):
    lines = [f"### {phase}", "", "| 模型 | 平均存活 | P10 | 跑满 | 平均分 | 每千块分 | 四消占比 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in variants:
        row = _summary(phase, variant)
        if row:
            lines.append(f"| {variant} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | "
                         f"{row['capped']}/{row['games']} | {row['mean_score']:.0f} | "
                         f"{row['score_per_1000']:.0f} | {row['four_fraction']:.2%} |")
    return lines + [""]


def generate(state, choice=None):
    REPORTS.mkdir(parents=True, exist_ok=True)
    metas = [_read(RUN / "trajectories" / f"seed_{seed}.json") for seed in TRAIN + VALIDATION]
    metas = [row for row in metas if row]
    labels = [_read(RUN / "labels" / f"seed_{seed}.json") for seed in TRAIN + VALIDATION]
    labels = [row for row in labels if row]
    verified = [value for row in labels for value in row["labels"].values()]
    trusted = [value for value in verified if value["category"] == "A"]
    trigger_counts = {name: sum(value.get("trigger") == name for value in trusted)
                      for name in ("obvious_danger", "rapid_growth")}
    summary = _read(RUN / "dataset_summary.json", {})
    lines = ["# PPO 30M 局部纠错：数据与证据", "", f"状态：`{state}`。训练种子 `{TRAIN[0]}..{TRAIN[-1]}`；"
             f"独立样本验证种子 `{VALIDATION[0]}..{VALIDATION[-1]}`；完整 episode 隔离。", "",
             "冻结 PPO 30M 与 BC 的 SHA256、Action/Observation/Feature 版本见 `runs/ppo30m_local_correction/protocol.json`。"
             "在线候选只使用公开观察、正式合法动作和一步落点。死亡距离仅作事后记录。", "",
             f"已采集 {len(metas)} 局，合计 {sum(x['pieces'] for x in metas):,} 块；"
             f"风险状态观察数 {sum(x['risk_states_seen'] for x in metas):,}，"
             f"策略分歧 {sum(x['policy_disagreements'] for x in metas):,}；"
             f"保留风险候选 {sum(x['risky_candidates'] for x in metas):,}，"
             f"正常样本 {sum(x['normal_samples'] for x in metas):,}。", "",
             "优先选择 PPO 实际死亡前 1000 块内、距死亡约 500 块的风险状态；"
             "从冻结 PPO 确定性重放目标位置，并验证公开状态哈希与动作一致。", "",
             f"已验证 {sum(len(x['labels']) for x in labels):,} 个同状态分支，"
             f"A 类可信纠错 {sum(x['A'] for x in labels):,}，"
             f"B 类不确定 {sum(x['B'] for x in labels):,}。"
             f"每个分支最多跟踪 {HORIZON} 块，触及上限右截断。", "",
             f"A 类触发来源：高位危险 {trigger_counts['obvious_danger']}，"
             f"单步快速增高 {trigger_counts['rapid_growth']}；"
             f"其中改变 Hold 用法 {sum(value.get('hold_changed', False) for value in trusted)}。"
             "这是被严格筛选后的构成，不代表 PPO 所有死亡的原因分布。", "",
             f"全部已分支状态中，PPO 分支在观察窗内死亡 "
             f"{sum(v['branches']['ppo']['game_over'] for v in verified)} 次；"
             f"BC 分支比 PPO 多活至少 250 块 "
             f"{sum(v['branches']['bc']['pieces'] >= v['branches']['ppo']['pieces'] + 250 for v in verified)} 次。"
             "BC 有时确实能救局，但并非意见不同时总是更好。", "",
             "A 类必须同时满足：PPO 一步高位危险或单步最高列增长至少 3 格、"
             "BC 一步显著更安全、可见下一步空间未显著缩水、"
             "PPO 分支在观察窗内死亡、BC 至少多活 250 块、共同存活区间得分保留 ≥80%、"
             "四消次数保留 ≥75%。快速增高触发在前 4 个训练种子分支后冻结，"
             "没有使用样本验证或开发评测结果选规则；这仍只是局部证据，不等于证明 BC 动作全局最优。", "",
             f"训练 A/C 分组：{summary.get('train_A', 0)}/{summary.get('train_C', 0)}；"
             f"样本验证 A/C 分组：{summary.get('validation_A', 0)}/{summary.get('validation_C', 0)}。", "",
             "PPO 与 BC 的 Hold 可能改变实际轨迹及未来方块使用顺序；两个分支从同一完整 Core/RNG 快照出发，"
             "但长程差异仍只构成有限观察窗内的局部证据。", ""]
    (REPORTS / "ppo30m_local_correction_dataset.md").write_text("\n".join(lines))

    lines = ["# PPO 30M 局部纠错：训练", "", f"状态：`{state}`。", "",
             "从冻结 PPO 30M 初始化新 Candidate Policy，只更新 Actor。A 类用合法动作交叉熵，"
             "C 类用原 PPO→新模型的 masked KL；Critic 冻结，未加入 PPO Policy Gradient。", ""]
    for variant in ("V1", "V2"):
        history = _read(RUN / "training" / variant / "history.json", [])
        if not history:
            continue
        config = _read(RUN / "training" / variant / "config.json")
        lines.extend([f"## {variant}", "", f"保持系数 {config['keep_weight']}；设备 {config['device']}；"
                      f"已完成 {len(history)} epoch；每个 epoch 的模型、优化器、RNG 均独立保存。", "",
                      "| Epoch | 纠错损失 | 保持损失 | 验证 A 动作一致率 | 验证 C 对 PPO 一致率 | 验证 C masked KL |",
                      "|---:|---:|---:|---:|---:|---:|"])
        for row in history:
            a, c = row["validation"]["danger"], row["validation"]["normal"]
            lines.append(f"| {row['epoch']} | {row['loss_correct']:.4f} | {row['loss_keep']:.4f} | "
                         f"{a['label_top1']:.2%} | {c['ppo_agreement']:.2%} | {c['masked_kl']:.5f} |")
        lines.append("")
    if state == "insufficient_verified_corrections":
        lines.extend(["可靠 A 类样本不足，按预设门槛停止监督训练。没有用 B 类不确定样本强行给 BC 标签。", ""])
    (REPORTS / "ppo30m_local_correction_training.md").write_text("\n".join(lines))

    lines = ["# PPO 30M 局部纠错：统一比较", "", f"状态：`{state}`。", "",
             f"开发种子 `{DEV[0]}..{DEV[-1]}` 用于选模；"
             f"确认种子 `{CONFIRM[0]}..{CONFIRM[-1]}` 仅在冻结 checkpoint 后评测一次。"
             "达到上限的对局按右截断处理。", ""]
    variants = ["ppo", "bc", "B12", "V1_e1", "V1_e2", "V1_e3", "V2_e1", "V2_e2", "V2_e3"]
    for phase in ("development_20k", "development_100k", "confirmation_100k"):
        if any(_summary(phase, variant) for variant in variants):
            lines.extend(_table(phase, variants))
    if choice:
        lines.extend([f"冻结候选：`{choice['variant']}`；checkpoint SHA256 "
                      f"`{choice['checkpoint_sha256']}`。", ""])
        local = _summary("confirmation_100k", choice["variant"])
        ppo = _summary("confirmation_100k", "ppo")
        bc = _summary("confirmation_100k", "bc")
        b12 = _summary("confirmation_100k", "B12")
        if all((local, ppo, bc, b12)):
            lines.extend(["## 判断", "", f"相对纯 PPO：平均存活差 "
                          f"{local['mean_pieces'] - ppo['mean_pieces']:+.0f} 块，"
                          f"平均分差 {local['mean_score'] - ppo['mean_score']:+.0f}；"
                          f"四消事件占比 {local['four_fraction']:.2%} 对 {ppo['four_fraction']:.2%}。", "",
                          f"BC 跑满 {bc['capped']}/{bc['games']}，B12 跑满 "
                          f"{b12['capped']}/{b12['games']}，新模型跑满 "
                          f"{local['capped']}/{local['games']}。"
                          "短期得分与长期生存必须分别解释；不能挑只跑满的局宣称整体获胜。", ""])
            base = RUN / "evaluations" / "confirmation_100k"
            lines.extend(["## 独立确认逐 seed 配对", "",
                          "| Seed | PPO 存活/分数 | BC 存活/分数 | B12 存活/分数 | 局部模型存活/分数 |",
                          "|---:|---:|---:|---:|---:|"])
            common_diffs = []
            for seed in CONFIRM:
                rows = {name: _read(base / name / f"seed_{seed}.json")
                        for name in ("ppo", "bc", "B12", choice["variant"])}
                if not all(rows.values()):
                    raise RuntimeError(f"Incomplete confirmation seed {seed}")
                def cell(row):
                    return f"{row['pieces_survived']}/{row['score']}"
                lines.append(f"| {seed} | {cell(rows['ppo'])} | {cell(rows['bc'])} | "
                             f"{cell(rows['B12'])} | {cell(rows[choice['variant']])} |")
                common = min(rows["ppo"]["pieces_survived"],
                             rows[choice["variant"]]["pieces_survived"])
                mark = common // 1000 * 1000
                if mark:
                    def at(row):
                        return next(x["score"] for x in row["score_milestones"]
                                    if x["pieces"] == mark)
                    common_diffs.append(at(rows[choice["variant"]]) - at(rows["ppo"]))
            lines.extend(["", f"在双方共同存活块数向下取整到千块时，局部模型减纯 PPO 的"
                          f"逐 seed 累计分差均值为 {sum(common_diffs) / len(common_diffs):+.0f}；"
                          "这一指标与完整局平均总分分开解释。", ""])
            for name in ("ppo", "bc", "B12", choice["variant"]):
                counts = _summary("confirmation_100k", name)["clear_counts_1_to_4"]
                lines.append(f"- {name}：单消/双消/三消/四消次数 = " + "/".join(map(str, counts)))
            lines.append("")
    else:
        lines.extend(["未冻结最终候选；不使用独立确认种子推断泛化能力。", ""])
    lines.extend(["逐 seed 原始结果和每千块累计得分位于 `runs/ppo30m_local_correction/evaluations/`；"
                  "各游戏按 seed 配对，不混用开发与确认数据。", ""])
    (REPORTS / "ppo30m_local_correction_comparison.md").write_text("\n".join(lines))
