# Traditional V1 Adapter 实测报告

这是合法动作适配版实测，不能视为原始网页 V1 的实测。

配置：7 seeds，最多 10000 块；CPU；代码版本 v1-adapter-1。
Git 基线：`308bc90427d62bb9fde954072f79503101fb07e9`；实际代码 SHA-256：`728bbd0f631b2ce33e7b86e2704ab7cedd5ee77a884cbf3ff680347823dc1a6e`。

保留原 V1 全部七项权重、下一块 0.58 系数及独立 Mulberry32 的 0.002 扰动。
当前与下一块候选均用正式 BFS，禁止 Hold；原版几何枚举不验证路径。候选集合/顺序变化使 RNG 消耗及最终选择可能不同。
策略 RNG 用 CLI seed 初始化，与游戏 RNG 隔离；策略只接收公开 Observation，不读取 Bag、RNG 或 Next 4+。
原网页 Down 使用 softDrop(true) 每格 +1；Core BFS Down 无加分，仅剩余 HardDrop 每格 +2，消行按正式等级计分。
保留原 V1 对任何负 y 格子的模拟拒绝；全被拒绝时选最小合法非 Hold 动作避免停滞，原网页可能停滞。
前瞻无候选时仍只用当前评分，未额外添加死亡惩罚。Core 的 partial lock/lock out/block out 规则不变。

决策时间包含当前合法动作及 mask 构造、全部一块前瞻和策略选择；不含 Core.step 与记录 I/O。每秒决策数为总决策次数/总决策时间。
P90/P95 使用 nearest-rank；逐 seed 文件包含完整动作及逐决策耗时。存活块计数沿用 Env 的已执行 placement 次数（含最后致死块）。

| 指标 | 实测 |
|---|---:|
| num_seeds | 7 |
| mean_pieces | 8386.428571 |
| median_pieces | 10000 |
| p90_pieces | 10000 |
| mean_lines | 3350.714286 |
| median_lines | 3999 |
| p90_lines | 3999 |
| mean_score | 68261800 |
| median_score | 88540700 |
| p90_score | 89339300 |
| game_over_rate | 0.285714 |
| cap_rate | 0.714286 |
| mean_decision_ms | 74.586080 |
| p95_decision_ms | 126.375025 |
| decisions_per_second | 13.407328 |
| end_to_end_pieces_per_second | 12.594712 |
| lines_per_piece | 0.399540 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |

| seed | 块数 | 消行 | 正式分数 | Game Over | 截断 |
|---|---:|---:|---:|---|---|
| 100019 | 5024 | 1998 | 22386900 | True | False |
| 100020 | 10000 | 3999 | 88255800 | False | True |
| 100021 | 10000 | 3999 | 88540700 | False | True |
| 100022 | 10000 | 3998 | 88612900 | False | True |
| 100023 | 10000 | 3999 | 88621400 | False | True |
| 100024 | 10000 | 3999 | 89339300 | False | True |
| 100025 | 3681 | 1463 | 12075600 | True | False |

达到上限是截断，不能将上限解释为真实寿命。至少半数截断，评测已饱和；应使用独立 validation 种子提高上限后再判断寿命。
本阶段未运行原网页 V1 或 V2，不能推断它们的成绩或算法提升。16 seeds 也不足以描述所有出块分布。
