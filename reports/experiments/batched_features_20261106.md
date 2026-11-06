# 批量NumPy特征：一致性通过，静态速度未过门

隔离worktree `/home/wyh/projects/tetris-batch-feature-validation`，分支codex/batched-candidate-features，代码`1524460`，训练来源28。main27及准确归档43cc26f逐项SHA相同，推理20在隔离/main/归档逐项同；没有采用本实现或更改训练模板。

## 实现与验证

显式bitboard_batched后端、要求compact搜索。一次处理全部候选行/列位图，使用整数位操作与256项byte bit_count/bit_length查表，避免浮点log2。清行候选仍调用原参考函数；异常/非典型输入回退以保留原输出或异常语义。

131项受保护差分/配置测试通过，覆盖随机20-bit列与2的幂边界、完整矩阵/mask/非法行零、多/非连续清行、已有满行、不可见/负y、空或变长cells、负action索引、重复/溢出/异常类型、200步真实轨迹及快照恢复。未验证真实GPU错误feature恢复，因为速度门未过。

固定原3份快照24状态加3边界，共27状态，预先固定落点，只计特征。30组AB/BA配对，CPU100%/2GiB/原温控；38输入未改。

| 指标 | 结果 |
| --- | ---: |
| 配对特征速度比中位数 | **1.110909** |
| 最小配对速度比 | 1.043306 |
| 预声明中位门槛 | **1.8，失败** |

所有feature/mask一致且每对更快，但eligible_for_gpu_benchmark=false。没有重跑计时、放宽门槛、启动CUDA8吞吐或完整上下文恢复，也未把静态提升称为训练提速。旧delta/fused负结果保持。

## 一次有界函数剖析

拒绝后另行预声明：同27状态/公开观察SHA、20轮，仅剖析该内核，不改变速度资格。14:59:38 passed，保护器14:59:40 completed；41输入未改。cProfile process_time总1.08795秒，包含较大剖析开销，累计项相互嵌套，不能当普通运行耗时/吞吐。

| 函数 | 调用数 | 累计CPU秒 |
| --- | ---: | ---: |
| batch candidate_features | 540 | 1.08612 |
| 原参考candidate_features回退 | 360 | 0.35037 |
| batch column_statistics | 1080 | 0.28450 |

参考总调用包含清行/空候选等回退，未分别计数，不能将360次全部精确归为清行。column_statistics每次候选构造调用两遍，包括仅有一个底板的统计。以上为下一实现提供线索，尚未证明具体优化收益。

## 保护与下一步

测试CPU峰60℃/GPU35℃/内存300.7MiB；静态41.5℃/37℃/15.4MiB；剖析56.25℃/37℃/280.2MiB，全100%/2GiB与原温控/GPU遥测。没有环境推进或模型/优化器更新。

保留本实现/负结果，不合入1524460、不启动其GPU验证。下一在新隔离来源研究完整向量化清行压缩及直接底板统计，避免重复参考回退与单元素数组调用；不是修补本版后重计时。仍需新固定差分、预声明静态和真实GPU门槛，未证明下一方案正确或更快。

证据：runs/autonomous_batched_features_20261106/runtime_archive.json、unit_guard/、micro/design.json、micro/microbenchmark.json、micro/qualification.json、profile/design.json、profile/feature_profile.json、resource_audit.json、inference_audit.json。原生goal首工具14:51:58真实执行见native_goal_batched_features_execution_20261106.json。
