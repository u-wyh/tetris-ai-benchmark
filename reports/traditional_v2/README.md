# Traditional V2 实验归档

V2 使用 V1 的七项特征作为基线，并增加 row transitions、column transitions 和顶部风险。hole depth/blockades 与 V1 的 holes/covered holes 高度重叠，landing height 与 aggregate/max height 重叠，因此没有重复加入；line clear efficiency 由消行奖励和清行后的结构特征共同表达。顶部风险只对最高列超过 13 行的部分平方惩罚，不把所有高堆局面判为必死。

策略流程是：从正式 `get_legal_placements()` 枚举当前与合法 Hold 分支；用增强评估筛选第一层的前 8 个；对每个候选继续使用公开可见的 Next/Hold 状态搜索一层，按当前分数加 `0.58 ×` 后继分数选择；可见 Next 消耗完就停止前瞻。等分时按较小 Action ID 选择，固定输入始终确定。

主要入口：

```bash
.venv/bin/python -m training.evaluation.traditional_v2 \
  --seeds 100000 100001 --max-pieces 5000 --mode beam --beam-width 8 \
  --lookahead 0.58 --output runs/v2_example
.venv/bin/python -m training.evaluation.traditional_v2 \
  --output runs/v2_example --resume --seeds 100000 100001 \
  --max-pieces 5000 --mode beam --beam-width 8 --lookahead 0.58
.venv/bin/python -m training.evaluation.traditional_v2 --status --output runs/v2_example
```

`validation16_paired_20261008_01/` 保存 16 个 validation seed 的 V1/V2 配对结果和 V2 原始 JSON；`ablation_dev_20261008.md` 记录开发集消融；`highcap_paired_dev_20261008/` 保存同 seed、10,000 块上限的补充配对；`profile.md` 是同公开局面 CPU 组件采样。
