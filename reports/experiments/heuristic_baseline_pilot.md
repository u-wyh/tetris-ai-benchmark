# `legacy_heuristic` Benchmark v1.0 Pilot

## 冻结的算法与评测口径

这是网页 V1 的 **Python 合法落点适配版**，不是原网页 AI 的原样实测。公开策略名为 `legacy_heuristic`，实现为 `training/traditional/v1.py` 的 `V1Adapter`（版本 `v1-adapter-1`）；`heuristic_hold` 本轮未实现。保留 `boardValue()` 七项权重：消行 +8.2、总高度 -0.47、洞 -4.2、covered holes -0.16、凹凸度 -0.28、最高列 -0.18、井深 -0.08；下一块最佳局面乘 0.58。保留固定 seed 的独立 Mulberry32 微扰（幅度 0.002），不消耗 Gameplay RNG。

当前块与下一块候选由正式 `get_legal_placements()` 搜索，最终动作经 1840 位 Action Mask 验证；不使用 Hold，只读取公开棋盘、当前块和 `Next[0]`。不可达的旧版几何落点被排除，SRS、Top Out 和计分均由 Python TetrisCore 执行。网页 V1 不考虑真实路径，且会对 Down 操作加 Soft Drop 分；本适配版使用 Core 的终点级动作与正式计分，故不能把两者的网页得分直接相等看待。原版模拟拒绝负 y 格；若全部候选都被拒绝，则选择第一个合法非 Hold 动作避免停滞。

Pilot 运行时 Git HEAD 为 `8d0599d351cf391dfd1bffdf853b3cd7bbea2ec4`；适配器与评测脚本随后提交于 `bd4b999`；算法与评测脚本 SHA-256 为 `728bbd0f631b2ce33e7b86e2704ab7cedd5ee77a884cbf3ff680347823dc1a6e`。开发 seeds `300000..300004` 与训练 `42..49`、validation `100000..100031`、final test `200000..200099` 不重合。所有逐 seed 结果原子保存在被 Git 忽略的 `runs/legacy_heuristic_pilot_*_20261008/`；记录动作、耗时、终止类型和正式游戏分数。

## 开发集 Pilot

| 上限 | 完成局数 | 平均存活块 | 平均消行 | 平均分数 | 触及上限 | 决策/秒 | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 5 | 1,000 | 398.6 | 906,380 | 5/5 | 18.80 | 各局 53.9–61.2 秒 |
| 10,000 | 5 | 6,639.2 | 2,649.2 | 46,188,480 | 2/5 | 19.76 | 各局 184.9–545.0 秒 |

| 10,000 块组 seed | 存活块 | 消行 | 得分 | 结束原因 |
| ---: | ---: | ---: | ---: | --- |
| 300000 | 10,000 | 3,998 | 89,067,600 | 到达上限 |
| 300001 | 4,298 | 1,713 | 16,424,000 | Game Over |
| 300002 | 3,416 | 1,355 | 10,403,200 | Game Over |
| 300003 | 10,000 | 3,998 | 88,312,000 | 到达上限 |
| 300004 | 5,482 | 2,182 | 26,735,600 | Game Over |

1,000 块组 5 局和 10,000 块组 2 局的存活均值被上限截断，分别只能解释为至少 1,000 和至少 6,639.2 块的样本均值；两组使用同一批开发 seed，10,000 块局的前 1,000 个动作与对应短局逐项一致。相同五个开发 seed 上，PPO-Raw best/final 分别平均存活 213.6/175.8 块，平均消行 73.6/58.8，平均得分 49,920/27,000；这只是 Pilot。正式 PPO-Raw final 在独立 100 seeds 上为 166.02 块，不能与开发集截断均值直接作正式配对结论。`heuristic_hold` 未实现，因此没有该版本的速度或成绩。

## 成本与正式评测准备

一次决策的耗时统计包含当前合法落点、Mask 和一层 Next 前瞻，不含 Core.step 与记录 I/O。单步 profile 中，9 次下一块合法落点搜索占约 28/33 ms 的策略时间，是主要瓶颈；权重和前瞻深度均未因此更改。10,000 块组实际合计 33,196 次决策，策略速度 19.76 次/秒，端到端速度 18.55 块/秒，串行 CPU 运行时间约 29.8 分钟；其中两局触及上限。若正式 100-seed 对局全部达到 50,000 块，按目前约 18–20 次/秒和每 10,000 块约 232 KiB 的记录量估算，串行 CPU 时间约 **70–77 小时**，逐 seed JSON 约 **113 MiB**；这是全局触顶的成本上界估计，实际可能因提前 Game Over 缩短。按这五个开发 seed 的平均长度简单外推约 9.3 小时，但不能据此预测正式 seeds 的寿命。正式评测已可使用与 PPO-Raw 相同的 `200000..200099` 共 100 seeds、每局最多 50,000 块，按相同的 TetrisCore 规则配对；算法在 final-test 前冻结，本阶段**不自动启动**。既有 PPO-Raw best/final 正式结果可直接作为配对基准；传统 AI 正式结果尚无。

正式评测的固定 seed 列表已在 `training/evaluation/seeds.json` 的 `final_test` 中，执行时调用 `training.evaluation.traditional.run(output, seeds, 50000)`；PPO-Raw best/final 的同 seed 成绩见 `runs/ppo_raw_10m_seed42_analysis/final/{best,final}.json`。保存目录不可复用，执行前应确认不会覆盖旧结果。

正确性测试覆盖网页公式、合法动作、公开信息边界、固定 seed 复现、无 Hold、Top Out、上限截断、分片兼容与异常后恢复。完整 Python 测试为 54 passed、29 parity subtests passed；JS 的 SRS/落点/规则回归为 58 passed。`TetrisCore`、Reward、Observation、Action Space 和 PPO 训练均未改动。
