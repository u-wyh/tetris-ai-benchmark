# 模型排行榜与完整 checkpoint 清单

快照时间：2026-11-12。本文件把散落在 `runs/` 与各实验报告里的模型结果集中到一处，
所有数字都注明协议、种子集与上限；**不同协议之间不可直接比较**。

## 协议定义

| 代号 | 种子 | 每局上限 | 用途 | 说明 |
|---|---|---:|---|---|
| A1 | 1421000–1421031（32） | 100,000 | 独立确认 | 未参与训练/选模 |
| A2 | 710100–710131（32） | 100,000 | 独立确认 | 另一批独立种子 |
| B | 1400000–1400007（8） | 3,000,000 | 长局对照 | 唯一能区分 BC/传统真实寿命的协议 |
| C | 100000–100031（32） | 10,000 / 20,000 | 训练期验证 | 多用于选模，不独立 |
| D | 100016–100025（10） | 100,000 | 传统基线对照 | BC/V2/V1 同批 |
| E | 1234000–1234029（30） | 100,000 | BC 泛化 | 全新种子 |

约定：达到上限记右截断（"触顶"），**不等于**真实寿命；P10 比平均值更能暴露脆弱性；
"平均分"受局长与等级影响，短命模型的高分不能解释为更强。

## 正式排行榜（协议 A1，32 seeds × 100K）

| 模型 | 平均存活 | 中位 | P10 | 触顶 | 真实死亡 | 平均分(亿) | 四消占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BC Champion** | **97,861** | 100,000 | 100,000 | 31/32 | 1 | 81.06 | 0.0% |
| **Hybrid B12** | 71,941 | 100,000 | 11,080 | 19/32 | 13 | **102.91** | 91.1% |
| PPO 30M | 61,823 | 62,236 | 13,225 | 9/32 | 23 | 78.22 | 93.2% |
| normal_994303 | 68,869 | 90,285 | 11,876 | 15/32 | 17 | 96.13 | 93.3% |
| normal_994202 | 61,267 | 65,976 | 13,028 | 11/32 | 21 | 79.07 | 93.8% |

结论：**生存冠军是 BC**（几乎不早死）；**均分冠军是 B12**（保留 PPO 的四消风格）；PPO 30M
是纯 RL 基线；两个 30M 微调候选在 A1 上都没通过预冻结门槛（P10 或平均分不足），
因此未替换冻结模型。来源：[独立确认报告](ppo30m_independent_confirmation.md)。

## 协议 A2（32 seeds × 100K，第二批独立种子）

| 模型 | 平均存活 | 中位 | P10 | 触顶 | 平均分(亿) |
|---|---:|---:|---:|---:|---:|
| BC Champion | **100,000** | 100,000 | 100,000 | 32/32 | 83.35 |
| Hybrid B12 | 79,743 | 100,000 | 15,738 | 21/32 | **117.74** |
| PPO 30M | 71,688 | 99,016 | 17,395 | 16/32 | 100.64 |
| Hybrid C1 | 62,801 | 94,878 | 7,512 | 14/32 | 88.90 |
| Hybrid C2 | 72,567 | 100,000 | 25,070 | 17/32 | 103.61 |

注：A1 与 A2 的 PPO 结果（61,823 vs 71,688）说明**种子差异本身就很大**，单批 32 局不足以
宣称模型间的小差距；C1/C2 是在 B12 基础上加的保守拦截规则，均未超过 B12。

## 协议 B（8 seeds × 3M，唯一的长寿命对照）

| 策略 | 平均存活 | P10 | 触顶 | 平均分 |
|---|---:|---:|---:|---:|
| **BC Champion** | **2,365,529** | **1,162,680** | 5/8 | 5.50 万亿 |
| Traditional V2 Hold | 1,436,759 | 130,038 | 2/8 | 2.84 万亿 |
| Hybrid B12 | 194,285 | 36,835 | 0/8 | 897.8 亿 |
| failure-curriculum curriculum | 120,429 | 21,262 | 0/8 | 450.3 亿 |
| PPO 30M | 100,672 | 14,084 | 0/8 | 329.4 亿 |
| failure-curriculum normal | 46,761 | 1,414 | 0/8 | 71.8 亿 |

这是全项目最重要的结论：**100K 时 BC 与 V2 Hold 都触顶、看起来打平；放到 3M，
BC 是 V2 的 1.6 倍、是 PPO 的 23 倍。** 来源：`runs/post_curriculum_3m/summary.json`。

## 传统算法（无神经网络权重）

| 策略 | 协议 | 平均存活 | 触顶 | 平均分(亿) |
|---|---|---:|---:|---:|
| V1 Adapter | D（10 seeds/100K） | 26,003 | 1/10 | 12.47 |
| V2 Hold-only | D（10 seeds/100K） | 100,000 | 10/10 | 83.34 |
| V2 Beam-8 | D（10 seeds/100K） | 100,000 | 10/10 | 83.04 |
| V2 Hold-only | E（30 seeds/100K） | 97,289 | 28/30 | 79.79 |
| BC Champion | E（30 seeds/100K） | 100,000 | 30/30 | 83.34 |

BC 在全新种子上以 2:0 微弱领先 V2（差距 2,711 块），不能算稳定优势；V1 已明显过时。

## 短程对照（20K / 32 seeds，训练期选模用）

| 模型 | 平均存活 | 触顶 | 说明 |
|---|---:|---:|---|
| BC Champion | 20,000 | 32/32 | — |
| DAgger v1（epoch 2） | 20,000 | 32/32 | 只换了访问状态的数据 |
| PPO 30M | 19,626 | 31/32 | — |
| KL-PPO 2M | 19,704 | 31/32 | 从 BC 起步 + KL 锚定 |
| BC+PPO 2M（普通微调） | 15,369 | 16/32 | 无 KL 锚定，明显退化 |

## 训练过但未进入正式榜的模型

| 模型 | 协议 | 结果 | 结论 |
|---|---|---|---|
| Raw MLP PPO 10M | C（32/10K） | 198 块 | 旧架构，远弱于 Candidate 的 4,543 |
| hole V1 λ0.10 2M | C（32/10K） | 43 | 奖励塑形后崩坏 |
| hole V2 λ0.02 2M | C（32/10K） | 52 | 同上 |
| hole+height V3 2M | C（32/10K） | 53 | 同上 |
| height V4 2M | C（32/10K） | 49 | 同上 |
| overnight KL raw | 16 seeds/50K | 39,747 | 低于 PPO 41,458 |
| overnight KL risk | 16 seeds/50K | 38,149 | 同上 |
| normal_994101 | 16 seeds/100K | 46,248 | 未过开发门槛 |
| curriculum_994101 | 16 seeds/100K | 73,596 | 未过开发门槛 |
| curriculum_994202 | 16 seeds/100K | 62,235 | 未过开发门槛 |
| curriculum_994303 | 16 seeds/100K | 53,551 | 未过开发门槛 |
| head 4k 候选 | GPU 匹配 8 seeds/20K | 存活比 0.936、得分比 0.907 | 4 项质量门槛全失败 |
| head 8k 候选 | 历史质量检查 | 6 cap / 2 death | 失败 |
| **head Newton 候选** | — | **未评测** | 唯一通过短程准入（train 3/8、val 1/4），完整棋力未做 |

## 完整 checkpoint 清单（SHA256 为文件内容哈希）

| 模型 | SHA256 | 训练步数 | 路径 |
|---|---|---:|---|
| PPO 2M | `91a3048d266e549fda51d4624351fa3cc72ccee86fab1a54a4fde85f4d63cb56` | 2,002,944 | runs/ppo_candidate_raw_2m_seed42/final/model.zip |
| PPO 10M | `3efc6e23229e8a3b988b3c6be249a4520715a10a79e6502c123962da83edccae` | 10,002,432 | runs/ppo_candidate_raw_10m_seed42/final/model.zip |
| PPO 20M | `f18ac18d08c09e64459c30164043578fb240a015fc57aadd50e39d5a80856fcc` | 20,000,768 | runs/ppo_candidate_raw_20m_seed42/final/model.zip |
| **PPO 30M（v1.0.0）** | `eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34` | 30,003,200 | runs/ppo_candidate_raw_30m_seed42/final/model.zip |
| Champion 7M | `c8eba1e1629b5c70cfd9fe6cd0e28cc79da9b4e06ead62aa8fedb5e77a22a2e9` | 7,000,064 | runs/ppo_candidate_champion/model.zip |
| Raw MLP 10M | `a3f88a41f27d7d041ba50af3a19be2dd9efb677af18b56f3d0b1f25cf1a66d71` | 10,002,432 | runs/ppo_raw_10m_seed42/final/model.zip |
| hole V1 λ.10 | `66e7dd8efeb11ef8f0e375f0310afb9ee04a709a9915c9ed6ed4538efbdb4bd5` | 2,002,944 | runs/ppo_hole_v1_2m_seed42_lambda010/final/model.zip |
| hole V2 λ.02 | `b7d6356a4cc3b0c59c86a095b53cadb6e32fc66e6facb8b584ee32f8b74113dc` | 2,002,944 | runs/ppo_hole_v2_2m_seed42_lambda002/final/model.zip |
| hole+height V3 | `545238526b8cdc991f03f1cda6df84b0598c4d4024b7be1719a2bdea886747e4` | 2,002,944 | runs/ppo_hole_height_v3_2m_seed42/final/model.zip |
| height V4 | `c5170503368f1946a7a5b3cb9b6a1c0ad2830411181cad738662928f1dda48cb` | 2,002,944 | runs/ppo_height_v4_2m_seed42/final/model.zip |
| normal_994101 | `be9a3601c26a59435b5ea3efa555fe7ad2ac6baed6ae7923988a26941e6229c8` | 503,808 | runs/ppo30m_multiseed_followup/training/normal_994101/committed/task_000123/model.zip |
| curriculum_994101 | `d06657fc2332fecbacb97f4feb2082005da464513589bf0a90d6e2bebb945fea` | 503,808 | runs/ppo30m_multiseed_followup/training/curriculum_994101/committed/task_000123/model.zip |
| normal_994202 | `13a0ab5bd5d1503f3974ac2ca9558a6eff8817701c8b36fb89bf9a80fc261702` | 503,808 | runs/ppo30m_multiseed_followup/training/normal_994202/committed/task_000123/model.zip |
| curriculum_994202 | `cacc80e3da3a3f7aeeb5084f4d3013d9ded842213b844650927392ebf3975a55` | 503,808 | runs/ppo30m_multiseed_followup/training/curriculum_994202/committed/task_000123/model.zip |
| normal_994303 | `5cd1d3d2bc19e10e2e90d252db06108c81bad307129caeeb315e238dc0123d2d` | 503,808 | runs/ppo30m_multiseed_followup/training/normal_994303/committed/task_000123/model.zip |
| curriculum_994303 | `6def89ea6e73aa9b774cde23b7cfe86751b6b9a17c71fa099886607d9909e3e8` | 503,808 | runs/ppo30m_multiseed_followup/training/curriculum_994303/committed/task_000123/model.zip |
| overnight raw | `66f8eff2a6e05f6cd035cea92ae40cf7ea5075c7beb71b5c799de2f8476f1e14` | 503,808 | runs/ppo30m_overnight_finetune/raw/committed/task_000123/model.zip |
| overnight risk | `fb805f87ea7f88edf142b8dc329817bac23fe2af8af677f28c73de19085a7287` | 503,808 | runs/ppo30m_overnight_finetune/risk/committed/task_000123/model.zip |
| failure-curriculum normal | `369674b1d481f5030fd9be8214094ae9331d177c7ddc888a8e5a55d5a3cb91bb` | 503,808 | runs/ppo30m_failure_curriculum/normal/committed/task_000123/model.zip |
| failure-curriculum curriculum | `0d9fe361c19edec8c66b6b1609ba4aedf5552144c1aae67c2a9c1883420a9c60` | 503,808 | runs/ppo30m_failure_curriculum/curriculum/committed/task_000123/model.zip |
| BC epoch 1 | `9de9d5fc0803027fbd3867567c90b428a32a3a1756d44cdacabd93cb9e1ceed1` | 20M PPO + BC e1 | runs/imitation_v2_hold_20m/bc/epoch_01.zip |
| BC epoch 2 | `6e1f183362c8afb4d6031f6de3f251a57a12db19a939d294f7c5e2a70d351878` | 20M PPO + BC e2 | runs/imitation_v2_hold_20m/bc/epoch_02.zip |
| **BC Champion（e3）** | `5f5613cb5eb4e80a02406b70902ad3d5c571eb8fe38bec4b40f73b34289ca663` | 20M PPO + BC e3 | runs/candidate_bc_next_stage/champion/model.zip |
| DAgger epoch 1 | `307e4a6efcfb33b8258799f8958d4726b5ad99b09f85dfab2beff433581d1d76` | 25k+20k 数据 | runs/candidate_bc_dagger_v1/bc/epoch_01.zip |
| DAgger epoch 2 | `0bd8635fd028e50478f79c0f375a05d686da2464d2988ada29cdddb2541a6d4b` | 25k+20k 数据 | runs/candidate_bc_dagger_v1/bc/epoch_02.zip |
| KL-PPO 2M | `70794b10776104fb3c7631a7eddcdbac157418eeb4e7e473187b75b8c509d1e1` | 2,002,944 | runs/candidate_bc_klppo_2m/final/model.zip |
| BC+PPO 2M | `32abd0792b3a080bdac6b5e0cb3e818c99b1d4721ba1a13c3681e76181e4a49c` | 2,002,944 | runs/imitation_v2_hold_20m/ppo_finetune_2m/final/model.zip |
| head 4k | `9a11ce79e7ef32c8a2dfa88a713422503cd6635082ef9d4b244eaa854dc65b3d` | 4,096 | runs/autonomous_actor_context_probe_20261106/training/committed/task_000001/model.zip |
| head 8k | `016b6be1bdc320ca1ab2ca31c677924b2511f7884ec3932dc1d73c389af33673` | 8,192 | runs/autonomous_actor_context_probe_20261106/training/committed/task_000002/model.zip |
| head Newton 候选 | `5453af80d37d4539eae1258a54fe8cc8443fda5875278cf5876e80b4d0b473cb` | 源 30M + 5 head 步 | runs/autonomous_head_newton_20261112/probe/evaluation_candidate.zip |

## 使用限制

1. **协议不可混用**：A1/A2/B/C/D/E 的种子、上限、目的都不同；例如 Champion 7M 的
   100 seeds/50K 平均 20,247 不能与 A1 的 61,823 比较。
2. **右截断**：100K/3M 触顶只是"至少活到上限"；B 协议里 BC 仍有 5/8 触顶，真实排序未知。
3. **样本量**：B 协议只有 8 局；A1/A2 各 32 局，仍不足以分辨小幅差异（参见 PPO 两批的 61,823/71,688）。
4. **验证集已被反复使用**：C 与 6 微调模型的开发组只用于选模，不能当独立泛化结论。
5. **模型文件不在 Git**：权重位于被忽略的 `runs/`，只有本清单里的 SHA256 可用于核对。
6. **B12/C1/C2 没有独立权重**：它们是 PPO 30M + BC 的组合策略，SHA 表中不出现。

## 复现入口

```bash
.venv/bin/python -m pytest tests/test_candidate_vs_v2_hold.py -q
.venv/bin/python -m unittest discover -s tests/parity -v
```

逐 seed 原始结果：`runs/ppo30m_independent_confirmation/`、`runs/ppo_candidate_champion/`、
`runs/ppo30m_multiseed_followup/evaluation/`、`runs/post_curriculum_3m/`、
`reports/traditional_longlife/`。
