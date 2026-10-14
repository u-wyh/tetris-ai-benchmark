"""Evidence-based V1 diagnosis and immutable active-V2 dataset reports."""
import collections
import json
import statistics
from datetime import datetime

from training.evaluation.local_correction_active_v2 import SEEDS, V2
from training.evaluation.local_correction_hold_audit import _well
from training.local_correction_data import RUN, TRAIN, VALIDATION
from training.tetris_core.legal_placements import get_legal_placements
from training.train_ppo import ROOT
from training.traditional.v2 import simulate_public

REPORTS = ROOT / "reports/experiments"


def _read(path):
    return json.loads(path.read_text())


def _reasons(item):
    e = item["evidence"]
    p, b = e["branches"]["ppo"], e["branches"]["bc"]
    reasons = []
    if e["trigger"] is None:
        reasons.append("无明确危险触发")
    if not e["precondition"] and e["trigger"] is not None:
        reasons.append("BC 一步安全改善不足")
    if not e["continuation_ok"]:
        reasons.append("可见下一步空间不足")
    if not p["game_over"]:
        reasons.append("PPO 观察窗内未死亡或右截断")
    if b["pieces"] < p["pieces"] + 250:
        reasons.append("BC 未多活 250 块")
    if e["common_score_ratio"] < .8:
        reasons.append("共同区间得分低于 80%")
    if e["common_four_clears"]["bc"] < .75 * e["common_four_clears"]["ppo"]:
        reasons.append("共同区间四消低于 75%")
    return reasons


def _metric_table(rows):
    lines = ["| 数据组 | Epoch | Top-1 | Top-3 | Top-5 | 平均专家概率 | 中位排名 | 平均 NLL |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for subset in ("train", "validation"):
        selected = [r for r in rows if r["set"] == subset]
        for ep in range(4):
            vals = [r["epochs"][ep] for r in selected]
            lines.append(f"| {subset} ({len(vals)}) | {ep} | {sum(v['top1'] for v in vals)} | "
                         f"{sum(v['top3'] for v in vals)} | {sum(v['top5'] for v in vals)} | "
                         f"{statistics.mean(v['expert_probability'] for v in vals):.4f} | "
                         f"{statistics.median(v['rank'] for v in vals):.1f} | "
                         f"{statistics.mean(v['expert_nll'] for v in vals):.3f} |")
    return lines


def _group_stats(items):
    if not items:
        return "无样本"
    values = [x["evidence"] for x in items]
    return (f"{len(items)} 条；PPO/BC 一步最高列均值 "
            f"{statistics.mean(x['ppo_risk']['height'] for x in items):.1f}/"
            f"{statistics.mean(x['bc_risk']['height'] for x in items):.1f}；"
            f"新增洞均值 {statistics.mean(x['ppo_risk']['new_holes'] for x in items):.2f}/"
            f"{statistics.mean(x['bc_risk']['new_holes'] for x in items):.2f}；"
            f"可见下一步合法落点均值 "
            f"{statistics.mean(v['next_legal']['ppo'] for v in values):.1f}/"
            f"{statistics.mean(v['next_legal']['bc'] for v in values):.1f}；"
            f"共同区间得分比中位数 {statistics.median(v['common_score_ratio'] for v in values):.2f}；"
            f"PPO/BC 共同区间四消总数 "
            f"{sum(v['common_four_clears']['ppo'] for v in values)}/"
            f"{sum(v['common_four_clears']['bc'] for v in values)}。")


def generate():
    diagnosis = _read(V2 / "diagnosis.json")
    details = diagnosis["action_details"]
    trusted = diagnosis["trusted"]
    rejected = diagnosis["rejected"]
    hold_audit = _read(V2 / "hold_audit.json")["records"]
    sweep = _read(V2 / "gradient_sweep.json")
    sensitivity = _read(V2 / "loss_sensitivity.json")
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# PPO 30M 局部纠错失败诊断", "",
             "本报告审计已冻结 V1 的 18 条训练 A 类与 3 条独立样本验证 A 类。"
             "诊断性过拟合模型只保存在内存中，未参加游戏评测。", "",
             "## 专家动作的真实变化", "", *_metric_table(details), "",
             "逐条审计（每格为专家动作概率/合法动作排名/当前 Top-1 Action ID）：", "",
             "| 集合 | Seed:Step | 改变 Hold | PPO 原策略 | Epoch 1 | Epoch 2 | Epoch 3 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in details:
        cells = [f"{v['expert_probability']:.4f}/#{v['rank']}/{v['selected_action']}"
                 for v in row["epochs"]]
        lines.append(f"| {row['set']} | {row['seed']}:{row['step']} | "
                     f"{'是' if row['hold_changed'] else '否'} | " + " | ".join(cells) + " |")
    down = [r for r in details if r["epochs"][3]["expert_probability"] <
            r["epochs"][0]["expert_probability"]]
    changed = [r for r in details if r["epochs"][3]["selected_action"] !=
               r["epochs"][0]["selected_action"]]
    lines.extend(["", f"专家动作概率相对原 PPO 下降 {len(down)}/21 条：" +
                  ("、".join(f"{r['seed']}:{r['step']}" for r in down) if down else "无") +
                  f"。Top-1 选中动作改变 {len(changed)}/21 条。"
                  "验证集仅 3 条，0/3 Top-1 不能单独证明无泛化；其平均专家概率及 NLL 仍在改善。", "",
                  "按 Hold 标志分组的专家动作概率变化：", "",
                  "| Hold 标志不同 | 样本数 | 原 PPO 平均概率 | Epoch 3 平均概率 | Epoch 3 Top-1 |",
                  "|---|---:|---:|---:|---:|"])
    for flag in (False, True):
        group = [r for r in details if r["hold_changed"] == flag]
        lines.append(f"| {'是' if flag else '否'} | {len(group)} | "
                     f"{statistics.mean(r['epochs'][0]['expert_probability'] for r in group):.4f} | "
                     f"{statistics.mean(r['epochs'][3]['expert_probability'] for r in group):.4f} | "
                     f"{sum(r['epochs'][3]['top1'] for r in group)} |")
    lines.extend(["",
                  "## 梯度与损失机制", ""])
    for name in ("gradient_baseline", "gradient_epoch3"):
        g = diagnosis[name]
        lines.append(f"- {name} 固定 8A+8C batch：A/C 梯度范数 "
                     f"{g['A_gradient_norm']:.3f}/{g['C_gradient_norm']:.3f}，"
                     f"余弦 {g['gradient_cosine']:.3f}，裁剪前合成范数 "
                     f"{g['combined_gradient_preclip_norm']:.3f}，一次 Actor 更新 L2 "
                     f"{g['actor_update_l2']:.5f}；参考 PPO 冻结={g['reference_frozen']}。")
    lines.extend(["", f"另取 {len(sweep['rows'])} 个固定 8A+8C batch："
                  f"{sweep['negative_cosine_batches']}/{len(sweep['rows'])} 的 A/C 梯度余弦为负，"
                  f"余弦中位数 {sweep['median_cosine']:.3f}，"
                  f"C/A 梯度范数比中位数 {sweep['median_C_to_A_ratio']:.1%}。"
                  "保持损失确有方向冲突，但通常小于纠错梯度；单个 batch 的 2.7% 不是总体代表值。", "",
                  "A/C 各 8 条按 batch 均值相加，原始 A:C 样本数 18:11,448 不会直接把 A 梯度按此比例稀释；"
                  "少量 A 样本被反复抽取。Actor 参数确有梯度及更新，"
                  "合法动作掩码参与 log-prob 和 KL，未见交叉熵失效的实现证据。"
                  "A 梯度占主导而网络参数共享，少量纠错更新也可能改变大量正常状态；"
                  "这与验证 C 对 PPO 一致率降到 92.07%、masked KL 升至 0.04756 一致，"
                  "但固定 batch 梯度不能单独量化每个正常状态的漂移原因。", "",
                  "同一批次顺序、同一初始 PPO 下，仅在内存中做 24 次更新的 KL 系数敏感性诊断：", "",
                  "| KL 系数 | 训练 A Top-1 | 验证 A Top-1 | 验证 C 对 PPO 一致率 | 验证 C masked KL |",
                  "|---:|---:|---:|---:|---:|"])
    for item in sensitivity["results"]:
        lines.append(f"| {item['keep_weight']} | "
                     f"{item['train']['danger']['label_top1']:.2%} | "
                     f"{item['validation']['danger']['label_top1']:.2%} | "
                     f"{item['validation']['normal']['ppo_agreement']:.2%} | "
                     f"{item['validation']['normal']['masked_kl']:.5f} |")
    lines.extend(["", "系数增大略改善正常动作保持，但 24 次更新后的 A 类 Top-1 未改善；"
                  "这些只是受控短程诊断，不能据此选择正式 KL 系数或推断长局得分。", "",
                  "## 18 条小样本记忆诊断", "",
                  "| 更新次数 | Top-1/18 | 平均专家概率 | 平均 NLL |",
                  "|---:|---:|---:|---:|"])
    for row in diagnosis["overfit"]:
        lines.append(f"| {row['updates']} | {row['top1']} | {row['mean_probability']:.4f} | "
                     f"{row['mean_nll']:.3f} |")
    lines.extend(["", "取消 C 损失并用 1e-4 学习率时，80 次更新可记住全部 18 条；"
                  "正式 V1 仅三个短 epoch、每轮 8 个 batch 共 24 次更新，学习率 5e-5。"
                  "这支持训练预算与有限标签是关键限制；不能由记忆测试推断游戏表现。", "",
                  "## Hold 与非 Hold 可信标签", "",
                  "- 改变 Hold：" + _group_stats([x for x in trusted if x["hold_changed"]]),
                  "- 不改变 Hold：" + _group_stats([x for x in trusted if not x["hold_changed"]]), "",
                  f"12 条 Hold 标志不同的标签中，一步后公开 Current/Hold/Next 实际不同 "
                  f"{sum(x['future_piece_state_differs'] for x in hold_audit if x['hold_flag_differs'])} 条；"
                  "其余标志虽不同，方块类型序列未变（例如 Current 与 Hold 同种）。"
                  "全部 21 条两种动作的一步后棋盘不同。", "",
                  "作为四消通道的粗代理，邻列高度减去井列高度的最大值：Hold 标志不同组"
                  f"PPO/BC 一步后均值 {statistics.mean(x['after_ppo']['well_depth'] for x in hold_audit if x['hold_flag_differs']):.2f}/"
                  f"{statistics.mean(x['after_bc']['well_depth'] for x in hold_audit if x['hold_flag_differs']):.2f}；"
                  "非 Hold 组为 "
                  f"{statistics.mean(x['after_ppo']['well_depth'] for x in hold_audit if not x['hold_flag_differs']):.2f}/"
                  f"{statistics.mean(x['after_bc']['well_depth'] for x in hold_audit if not x['hold_flag_differs']):.2f}。"
                  "两组 BC 动作均倾向填平深井，这可能损害四消布局；该代理不是完整四消结构判据。", "",
                  "1,000 块反事实窗口的四消保留不保证长期通道完整。"
                  "现有 21 条 A 无法建立 Hold 改变导致 V1 四消下降的因果结论；"
                  "后续训练应按实际方块序列改变与否分层评估。", "",
                  "| Seed:Step | Hold 标志变 | 下一方块状态变 | PPO/BC 一步后井深 |",
                  "|---|---:|---:|---:|"])
    for row in hold_audit:
        lines.append(f"| {row['seed']}:{row['step']} | "
                     f"{'是' if row['hold_flag_differs'] else '否'} | "
                     f"{'是' if row['future_piece_state_differs'] else '否'} | "
                     f"{row['after_ppo']['well_depth']}/{row['after_bc']['well_depth']} |")
    lines.append("")
    (REPORTS / "local_correction_failure_diagnosis.md").write_text("\n".join(lines))

    primary = collections.Counter()
    overlapping = collections.Counter()
    for row in rejected:
        reasons = _reasons(row)
        if not reasons:
            raise RuntimeError("B row has no rejection reason")
        primary[reasons[0]] += 1
        overlapping.update(reasons)
    lines = ["# 局部纠错反事实与主动采样 V2", "",
             "原 V1 规则原样用于新分支；A 类要求明确危险、BC 一步显著更安全、"
             "可见下一步合法空间、PPO 分支在 1,000 块内死亡、BC 至少多活 250 块、"
             "共同得分至少 80% 与四消至少 75%。触顶按右截断。", "",
             "## V1 的 202 条 B 类拒绝原因", "",
             "原因可重叠；首因按危险触发→一步安全→下一步空间→PPO 死亡→BC 存活→得分→四消顺序判定。", "",
             "| 条件未满足 | 首因条数 | 涉及条数 |", "|---|---:|---:|"]
    for key in ("无明确危险触发", "BC 一步安全改善不足", "可见下一步空间不足",
                "PPO 观察窗内未死亡或右截断", "BC 未多活 250 块",
                "共同区间得分低于 80%", "共同区间四消低于 75%"):
        lines.append(f"| {key} | {primary[key]} | {overlapping[key]} |")
    both_censored = sum(x["evidence"]["branches"]["ppo"]["horizon_censored"] and
                        x["evidence"]["branches"]["bc"]["horizon_censored"] for x in rejected)
    lines.extend(["", f"B 类改变 Hold {sum(x['hold_changed'] for x in rejected)}/202；"
                  f"两个分支均触及观察窗右截断 {both_censored}/202。"
                  "得分和四消条件很少单独构成首因；大量样本缺少明确危险或一步安全证据，"
                  "另一部分无法在有限窗口证明 BC 真能救局。不能为了增加 A 数量放宽这些门槛。", "",
                  "## V1 可信分支示例", "",
                  "以下固定列出一个方块状态实际改变的 Hold 标签、一个仅 Hold 标志不同的标签、"
                  "一个非 Hold 标签；用于说明证据字段，不代表总体效果。", "",
                  "| Seed:Step | Hold 标志/未来方块状态变 | PPO/BC 落点高度 | PPO/BC 观察窗存活 | 共同分数比 | 共同四消 PPO/BC |",
                  "|---|---|---:|---:|---:|---:|"])
    future_changed = {(r["seed"], r["step"]): r["future_piece_state_differs"] for r in hold_audit}
    for seed, step in ((730001, 8575), (730015, 408), (730013, 79111)):
        row = next(x for x in trusted if x["seed"] == seed and x["step"] == step)
        e = row["evidence"]
        lines.append(f"| {seed}:{step} | "
                     f"{'是' if e['hold_changed'] else '否'}/"
                     f"{'是' if future_changed[(seed, step)] else '否'} | "
                     f"{row['ppo_risk']['height']}/{row['bc_risk']['height']} | "
                     f"{e['branches']['ppo']['pieces']}/{e['branches']['bc']['pieces']} | "
                     f"{e['common_score_ratio']:.2f} | "
                     f"{e['common_four_clears']['ppo']}/{e['common_four_clears']['bc']} |")
    lines.extend(["", "BC 分支达到 1,000 块时是右截断，表中不得解释成恰好存活 1,000 块后死亡。", "",
                  "## V2 主动采样与结果", ""])
    status_path = V2 / "status.json"
    status = _read(status_path) if status_path.exists() else {"state": "not_started"}
    lines.append(f"队列状态：`{status['state']}`。从 20 个 V1 轨迹种子已保存的风险候选中选择，"
                 "每局最多 30 条、相距至少 10 块，优先死亡前约 500 块及快速增高，"
                 "每局保留最多 6 条随机候选。没有生成新普通轨迹。")
    selection = [_read(V2 / "selection" / f"seed_{seed}.json") for seed in SEEDS]
    original_candidates = sum(_read(RUN / "trajectories" / f"seed_{seed}.json")["risky_candidates"]
                              for seed in SEEDS)
    lines.append(f"V1 原始保留风险候选 {original_candidates} 条；"
                 f"冻结新候选 {sum(len(x['steps']) for x in selection)} 条；"
                 f"其中随机配额 {sum(x['random_count'] for x in selection)} 条。"
                 "原 V1 223 条已分支状态被排除；少数早死局候选不足 30 条，没有用相邻状态补齐。")
    branch_files = sorted((V2 / "branches").glob("seed_*/step_*.json"))
    records = [_read(path) for path in branch_files]
    if records:
        a = [x for x in records if x["evidence"]["category"] == "A"]
        lines.extend(["", f"已原子完成新分支 {len(records)} 条：A {len(a)}、B {len(records)-len(a)}；"
                      f"{'最终' if status['state']=='completed' else '当前'}通过率 {len(a)/len(records):.2%}。"
                      f"原 V1 为 21/223 = {21/223:.2%}。" +
                      ("" if status["state"] == "completed" else
                       "队列未完成时仅代表进度，不用于判断采样效率。"), ""])
        for hold in (False, True):
            subset = [x for x in records if x["evidence"]["hold_changed"] == hold]
            if subset:
                lines.append(f"- {'Hold 改变' if hold else 'Hold 未变'}："
                             f"A {sum(x['evidence']['category']=='A' for x in subset)}/{len(subset)}。")
        lines.append("")
    if status["state"] == "completed":
        decision = _read(V2 / "readiness.json")
        finished = datetime.fromisoformat(status["updated_at"])
        protocol_mtime = datetime.fromtimestamp((V2 / "protocol.json").stat().st_mtime,
                                                tz=finished.tzinfo)
        elapsed_minutes = (finished - protocol_mtime).total_seconds() / 60
        v1_all = trusted + rejected
        def near_v1(x):
            distance = x["death_distance"]
            return distance is not None and distance <= 1000
        def near_v2(x):
            distance = x["row"]["death_distance"]
            return distance is not None and distance <= 1000
        lines.extend(["## 主动采样效率与瓶颈", "",
                      "| 数据批次 | 死亡前 ≤1000 块 A/验证 | 其他或右截断 A/验证 | 总体 A/验证 |",
                      "|---|---:|---:|---:|"])
        for label, items, pred, is_a in (("V1", v1_all, near_v1,
                                         lambda x: x["evidence"]["category"] == "A"),
                                        ("V2 新分支", records, near_v2,
                                         lambda x: x["evidence"]["category"] == "A")):
            near = [x for x in items if pred(x)]
            far = [x for x in items if not pred(x)]
            lines.append(f"| {label} | {sum(is_a(x) for x in near)}/{len(near)} | "
                         f"{sum(is_a(x) for x in far)}/{len(far)} | "
                         f"{sum(is_a(x) for x in items)}/{len(items)} |")
        v2_b = [{"evidence": x["evidence"]} for x in records if x["evidence"]["category"] == "B"]
        v2_primary = collections.Counter(_reasons(x)[0] for x in v2_b)
        v2_overlap = collections.Counter(reason for x in v2_b for reason in _reasons(x))
        lines.extend(["", f"V2 新增 {len(v2_b)} 条 B 类的拒绝原因：", "",
                      "| 条件未满足 | 首因条数 | 涉及条数 |", "|---|---:|---:|"])
        for key in ("无明确危险触发", "BC 一步安全改善不足", "可见下一步空间不足",
                    "PPO 观察窗内未死亡或右截断", "BC 未多活 250 块",
                    "共同区间得分低于 80%", "共同区间四消低于 75%"):
            lines.append(f"| {key} | {v2_primary[key]} | {v2_overlap[key]} |")
        lines.extend(["", "新批次中 429/556 条不在原 PPO 死亡前 1,000 块内或该局触顶；"
                      "这些状态 0 条获 A。临死窗口内新批次 22/127 获 A（17.32%），"
                      "仍低于 V1 同类 21/65（32.31%）。"
                      "固定每局最多 30 条、仅相隔 10 块的选择没有提高总体可信标签获取率；"
                      "两批来自相同源 episode 且非随机对照，不能把差异解释成单一规则的因果效果。"
                      f"本轮从冻结协议文件落盘到队列完成约 {elapsed_minutes:.1f} 分钟，"
                      "本批预算止于 556 条，不继续扩大低收益的远死亡状态分支。", ""])
        lines.extend([f"最终判断：`{decision['state']}`；训练/验证新增 A "
                      f"{decision['train_A_new']}/{decision['validation_A_new']}，"
                      f"与 V1 合并后为 {18+decision['train_A_new']}/{3+decision['validation_A_new']}；"
                      f"训练 A 覆盖 {decision['train_A_episodes']} 局，"
                      f"其中非 Hold {decision['train_A_non_hold']} 条。"
                      "只有训练 A ≥100、验证 A ≥30、训练覆盖 ≥8 局、非 Hold A ≥30"
                      "才允许提出后续小规模训练 smoke；这是一条只看新增样本的保守门槛。"
                      "本次不自动训练。", ""])
        if decision["state"] != "ready_for_smoke_only":
            lines.extend(["当前不具备再次训练的预设数据条件。保留 V1/V2 全部已验证分支，"
                          "下一轮应优先针对 B 类首因补充不同 episode 的真实危险状态，"
                          "并按实际方块序列改变与深井损失区分 Hold 标签；不降低 A 类安全门槛。"
                          "正常状态保持机制也需先通过动作一致率和四消布局检查。", ""])
        else:
            lines.extend(["数据数量门槛通过，但 V1 正常状态策略曾明显漂移。"
                          "只建议先设计更强的正常状态保留约束、分层 Hold 标签，"
                          "随后用少量 batch smoke 核对梯度和正常状态动作变化；"
                          "本次仍不启动正式模型训练。", ""])
    (REPORTS / "local_correction_counterfactual_v2.md").write_text("\n".join(lines))

    lines = ["# 局部纠错高质量数据集 V2", "",
             "V2 位于 `runs/ppo30m_local_correction/active_v2/`，V1 轨迹和标签均保留。"
             "训练、样本验证和未来游戏评测种子隔离。每个新分支包含源公开 Observation、"
             "合法 Action ID/候选特征、PPO/BC 动作、合法掩码下 PPO 原动作概率、"
             "一步风险、Hold 差异、1,000 块反事实证据、等级、源 SHA256 与规则版本。", "",
             "| Seed | 分组 | 新候选 | 新 A | 改变 Hold 的 A |",
             "|---:|---|---:|---:|---:|"]
    for entry in selection:
        seed = entry["seed"]
        subset = [x for x in records if x["seed"] == seed]
        a = [x for x in subset if x["evidence"]["category"] == "A"]
        lines.append(f"| {seed} | {'train' if seed in TRAIN else 'validation'} | "
                     f"{len(entry['steps'])} | {len(a)} | "
                     f"{sum(x['evidence']['hold_changed'] for x in a)} |")
    lines.extend(["", "原 V1 A/C 为训练 18/11,448、样本验证 3/1,752；"
                  "V2 只扩充风险反事实证据，正常 C 样本仍可引用 V1 冻结分片。"
                  "分支输出逐文件原子提交，完成后 `dataset_manifest.json` 记录每个文件 SHA256；"
                  "重复运行会验证模型、来源、规则和选择身份后跳过完成分支。", ""])
    meta_path = V2 / "dataset_rows.json"
    if meta_path.exists():
        meta = _read(meta_path)
        lines.extend([f"训练读取用 `dataset_rows.jsonl.gz` 含 {meta['records']} 条记录，"
                      "显式保存 Episode ID、逐条 SHA256、源文件及规则版本；"
                      f"压缩数据 SHA256：`{meta['dataset_sha256']}`。"
                      "死亡距离字段只供离线筛选，在线推理不得使用。", ""])
    if status["state"] == "completed":
        a = [x for x in records if x["evidence"]["category"] == "A"]
        train_new = sum(x["seed"] in TRAIN for x in a)
        val_new = len(a) - train_new
        one_step = []
        for item in a:
            row = item["row"]
            placements = {p["actionId"]: p for p in get_legal_placements(row["public"])}
            after = [simulate_public(row["public"], placements[row[f"{name}_action"]])[0]
                     for name in ("ppo", "bc")]
            def future(state):
                current = state["currentPiece"]
                return (current["type"] if current else None, state["hold"], tuple(state["next"]))
            one_step.append({"hold_flag": item["evidence"]["hold_changed"],
                             "future_changed": future(after[0]) != future(after[1]),
                             "ppo_well": _well(after[0]["board"]),
                             "bc_well": _well(after[1]["board"])})
        def distance_band(x):
            distance = x["row"]["death_distance"]
            return ("≤250" if distance is not None and distance <= 250 else
                    "251–1000" if distance is not None and distance <= 1000 else ">1000/截断")
        death_bins = collections.Counter(distance_band(x) for x in a)
        all_bins = collections.Counter(distance_band(x) for x in records)
        trigger = collections.Counter(x["evidence"]["trigger"] or "无可信危险触发" for x in a)
        lines.extend(["## A 类覆盖", "", f"新增 A {len(a)}/{len(records)}；改变 Hold "
                      f"{sum(x['evidence']['hold_changed'] for x in a)} 条。"
                      f"训练/验证新增 A {train_new}/{val_new}，"
                      f"连同 V1 的 18/3，分别共 {18+train_new}/{3+val_new} 条。"
                      f"死亡距离分布：≤250 为 {death_bins['≤250']}，251–1000 为 "
                      f"{death_bins['251–1000']}，>1000/截断为 {death_bins['>1000/截断']}。", "",
                      f"全部新分支的死亡距离分布：≤250 为 {all_bins['≤250']}，"
                      f"251–1000 为 {all_bins['251–1000']}，>1000/截断为 {all_bins['>1000/截断']}。"
                      f"新增 A 的触发类型：明显高危 {trigger['obvious_danger']}，"
                      f"快速增高 {trigger['rapid_growth']}，无可信危险触发 {trigger['无可信危险触发']}。", "",
                      f"新 A 中 Hold 标志不同 {sum(x['hold_flag'] for x in one_step)} 条，"
                      f"但下一步 Current/Hold/Next 方块状态实际不同 "
                      f"{sum(x['future_changed'] for x in one_step)} 条。"
                      "Hold 标志不同组的一步后深井粗代理 PPO/BC 均值 "
                      f"{statistics.mean(x['ppo_well'] for x in one_step if x['hold_flag']):.2f}/"
                      f"{statistics.mean(x['bc_well'] for x in one_step if x['hold_flag']):.2f}；"
                      "非 Hold 组为 "
                      f"{statistics.mean(x['ppo_well'] for x in one_step if not x['hold_flag']):.2f}/"
                      f"{statistics.mean(x['bc_well'] for x in one_step if not x['hold_flag']):.2f}。"
                      "深井代理仅描述一步形状，不能证明后续四消因果。", "",
                      "此数据集仅能证明固定窗口和固定续控下的局部证据；"
                      "Hold 分支可能改变方块消耗顺序，不能把两条轨迹当成完全相同的未来。", ""])
    (REPORTS / "local_correction_dataset_v2.md").write_text("\n".join(lines))


if __name__ == "__main__":
    generate()
