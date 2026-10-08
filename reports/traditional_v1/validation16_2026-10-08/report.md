# Traditional V1 Adapter 实测报告

这是合法动作适配版实测，不能视为原始网页 V1 的实测。

配置：16 seeds，最多 5000 块；CPU；代码版本 v1-adapter-1。
Git 基线：`8d0599d351cf391dfd1bffdf853b3cd7bbea2ec4`；实际代码 SHA-256：`728bbd0f631b2ce33e7b86e2704ab7cedd5ee77a884cbf3ff680347823dc1a6e`。

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
| num_seeds | 16 |
| mean_pieces | 4044.687500 |
| median_pieces | 5000.000000 |
| p90_pieces | 5000 |
| mean_lines | 1612.937500 |
| median_lines | 1997.000000 |
| p90_lines | 1999 |
| mean_score | 16216331.250000 |
| median_score | 22050400.000000 |
| p90_score | 22463900 |
| game_over_rate | 0.375000 |
| cap_rate | 0.625000 |
| mean_decision_ms | 50.616429 |
| p95_decision_ms | 80.255648 |
| decisions_per_second | 19.756431 |
| end_to_end_pieces_per_second | 18.542691 |
| lines_per_piece | 0.398779 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |

| seed | 块数 | 消行 | 正式分数 | Game Over | 截断 |
|---|---:|---:|---:|---|---|
| 100000 | 5000 | 1997 | 21885000 | False | True |
| 100001 | 2952 | 1168 | 7885100 | True | False |
| 100002 | 1648 | 649 | 2358700 | True | False |
| 100003 | 5000 | 1996 | 22464100 | False | True |
| 100004 | 5000 | 1997 | 22463900 | False | True |
| 100005 | 5000 | 1997 | 22187800 | False | True |
| 100006 | 5000 | 1999 | 22291000 | False | True |
| 100007 | 5000 | 1998 | 22318600 | False | True |
| 100008 | 5000 | 1997 | 22121700 | False | True |
| 100009 | 2407 | 953 | 5045700 | True | False |
| 100010 | 1312 | 517 | 1478500 | True | False |
| 100011 | 5000 | 1999 | 22223100 | False | True |
| 100012 | 5000 | 1999 | 22032800 | False | True |
| 100013 | 5000 | 2000 | 22068000 | False | True |
| 100014 | 1942 | 768 | 3230100 | True | False |
| 100015 | 4454 | 1773 | 17407200 | True | False |

达到上限是截断，不能将上限解释为真实寿命。至少半数截断，评测已饱和；应使用独立 validation 种子提高上限后再判断寿命。
本阶段未运行原网页 V1 或 V2，不能推断它们的成绩或算法提升。16 seeds 也不足以描述所有出块分布。

## 并行执行与结果核对

主进程完成 seeds 100000–100003；另外三个独立 tmux/CPU 进程分别完成 100004–100007、100008–100011、100012–100015。分片均使用相同代码 SHA-256、规则、策略和 5000 块上限；合并时校验配置并拒绝覆盖已有 seed。主进程在 seed 100004 的半局被有意中断，半局未计入结果；`error_*.json` 保留该中断记录。`merge_manifest.json` 记录各 seed 来源。决策耗时是这些并行 CPU 进程运行时的墙钟时间，受同机负载影响；和其他实验比较计算成本时需保持相同负载条件。

Smoke Test 在 3 个固定 validation seeds（100000–100002）各运行 200 块，均截断；独立重跑首个 seed 的动作序列、分数、消行和终止状态相同。正式 16 局中有 10 局在上限截断，当前上限不足以估计它们的最终存活寿命；如需比较长寿命策略，应只使用独立 validation seeds 提高上限，继续保留 final test seeds 不参与调参。
