# 自主强化学习阶段结果

状态：503808 步开发短局未通过预设门槛。

CPU 训练使用加长折扣和 GAE、冻结 PPO KL 锚定；触顶右截断。

## 结论与停止依据

新模型完成 123 个原子事务，共 **503,808** 新环境步；未进入 2M 或追加阶段，独立确认组 `1492000..1492031` 完全未使用。16 个新开发 seed、20K 上限下：

| 模型 | 平均已观测存活 | P10 | 触顶 | 真实 <10K | 平均正式得分 | 四消占比 |
|---|---:|---:|---:|---:|---:|---:|
| PPO 30M | 18,682.5 | 15,991.5 | 11/16 | 1 | 5.702 亿 | 92.85% |
| Hybrid B12 | 16,172.9 | 5,092.0 | 12/16 | 4 | 4.870 亿 | 91.60% |
| BC Champion | 20,000 | 20,000 | 16/16 | 0 | 3.335 亿 | 0.03% |
| 新候选 | 18,472.7 | 15,389.0 | 13/16 | 1 | 5.606 亿 | 88.26% |

候选相对 PPO 平均已观测存活低 **1.1%**、P10 低 **3.8%**、平均总分低 **1.7%**；双方共同存活里程碑得分比 **99.36%**。四消占比低 **4.59 个百分点**，超过事前允许的 3 个百分点，因此按冻结门槛停止。20K 下大量右截断，触顶数差异不能证明真实长期寿命提升。候选 checkpoint：`runs/autonomous_rl_research/training/committed/task_000123/model.zip`，SHA256 `71a7d05d083cda8c8bd7c7517b68d1abe27516ad1e7fc54e13dce6824d06c6f6`；事务模型、优化器、RNG 与 worker 状态校验通过。它不是新冠军。

训练事务记录的墙钟合计约 **2,484 秒**；四模型开发评测逐局墙钟合计约 **3,229 秒**（并行执行，不能直接当实际经过时间）。2K 烟测四组各 2/2 触顶，`early_under_10000` 原始字段在 2K 上限下包含右截断，**不可解释成早死数量**。正式短局门槛只按 20K 结果判断。

固定历史 seed 100016 的 100 个公开状态、CPU 单线程、相同合法掩码下，模型动作评分部分的平均/P95 耗时：原 PPO **0.150/0.169 ms**，候选 **0.148/0.162 ms**。这不包含合法落点与特征构建，不能当作网页全链路延迟。原始样本汇总见 `runs/autonomous_rl_research/inference_benchmark.json`。

## 平台性能回退研究

停止训练后，对同一批合法动作测试训练环境复用已计算的正式合法落点。`TransactionFastCandidateEnv` 为新实验提供显式选择，旧配置默认行为不变。三个历史 seed 各 300 块、交替运行顺序、重复三次，普通环境 **144.6 步/秒**，缓存环境 **255.4 步/秒**，配对均值加速 **1.77 倍**；按 seed 分组的探索性 Bootstrap 95% 区间约 **1.758–1.774**。九组棋盘、分数和消行完全一致；独立 200 步测试还验证了逐步 observation/mask/reward/结束标志和快照恢复。峰值进程 RSS 约 **495 MiB**，同进程测量不能把两种环境的内存独立归因。该速度是单环境 `action_masks()+step()` 吞吐量，**不等于完整 PPO 训练可加速 1.77 倍**；本轮未用新环境重训或复用确认种子。原始数据：`runs/autonomous_rl_research/performance.json`。

## smoke

```json
{
  "ppo": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 6353400,
    "score_per_1000": 3176700.0,
    "four_fraction": 0.9418886198547215
  },
  "B12": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 6324650,
    "score_per_1000": 3162325.0,
    "four_fraction": 0.9436274509803921
  },
  "bc": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 3371900,
    "score_per_1000": 1685950.0,
    "four_fraction": 0.0
  },
  "candidate": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 6335950,
    "score_per_1000": 3167975.0,
    "four_fraction": 0.9253012048192771,
    "common_score_ratio_to_ppo": 0.9972534391034722
  }
}

```

## dev_20k

```json
{
  "ppo": {
    "games": 16,
    "mean_pieces": 18682.5,
    "p10_pieces": 15991.5,
    "capped": 11,
    "early_under_10000": 1,
    "mean_score": 570211956.25,
    "score_per_1000": 30521180.583433695,
    "four_fraction": 0.9284749034749035
  },
  "B12": {
    "games": 16,
    "mean_pieces": 16172.9375,
    "p10_pieces": 5092.0,
    "capped": 12,
    "early_under_10000": 4,
    "mean_score": 486958500,
    "score_per_1000": 30109465.271846876,
    "four_fraction": 0.9159586257850019
  },
  "bc": {
    "games": 16,
    "mean_pieces": 20000,
    "p10_pieces": 20000.0,
    "capped": 16,
    "early_under_10000": 0,
    "mean_score": 333544312.5,
    "score_per_1000": 16677215.625,
    "four_fraction": 0.00029338418659234265
  },
  "candidate": {
    "games": 16,
    "mean_pieces": 18472.6875,
    "p10_pieces": 15389.0,
    "capped": 13,
    "early_under_10000": 1,
    "mean_score": 560643043.75,
    "score_per_1000": 30349836.41389484,
    "four_fraction": 0.8825672379383533,
    "common_score_ratio_to_ppo": 0.9935890770378796
  }
}

```
