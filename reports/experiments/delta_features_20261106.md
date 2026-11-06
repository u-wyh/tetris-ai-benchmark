# 增量候选特征：一致性通过，静态速度未过门

隔离worktree `/home/wyh/projects/tetris-delta-feature-validation`，分支codex/delta-candidate-features，代码提交`52afbfd`。main27训练来源改动前已归档至`/home/wyh/projects/tetris-runtime-43cc26f`，27个SHA与当前已冻结数据来源逐项同。

## 新算法与保护

显式bitboard_delta后端、要求compact搜索，新增training-only模块，隔离训练来源28。预计算底板列高/孔洞/顶部占格/相邻高度差/井深；无清行时只更新触及列与其邻居，通过1024种column mask预计算索引。仅检查触及行是否新满行；已有满行和清行完整重算，完全不可见落点保持旧“不清行”语义。

main与旧fused副本未改；原bitboard默认仍可用。推理20个源码SHA与main/准确归档全部相同。没有更改原模型、训练模板或旧断点身份。

123项受保护差分/配置测试通过，包括1024种列掩码与独立逐格board_features参考、84种piece/board/hold组合、多行/非连续清行、已有满行、全部不可见/负y/顶出、重复action/溢出拒绝，以及200步真实轨迹与快照恢复。只测试了后端配置拒绝，尚未执行真实GPU错误feature恢复验收。

## 静态结果

固定3份提交快照共24状态，加3个清行/顶出边界，共27状态。落点提前固定，只计特征；30对交替AB/BA，同CPU100%/2GiB/原温控。38个输入完成后逐项未改。

| 指标 | 结果 |
| --- | ---: |
| 当前bitboard特征每批CPU时间中位 | 0.0122773秒 |
| delta候选每批CPU时间中位 | 0.0072487秒 |
| 配对速度比中位数 | **1.690273** |
| 最小配对速度比 | 1.598473 |
| 预声明中位门槛 | **1.8，未通过** |

所有feature/mask逐项一致且每对更快，但不足以进入下一阶段。**eligible_for_gpu_benchmark=false**：未启动真实CUDA8吞吐对照、错误feature恢复或完整上下文扩展/故障恢复；这些均未测，不能称通过。没有重计时、放宽门槛或将静态提升当成训练提速。

因此保留代码和负结果，main继续使用已验证row masks+原bitboard特征，不合入52afbfd。旧fused的4.91%端到端负结果也保留。

## 资源与后续

静态保护器14:46:29 completed；静态CPU峰41.375℃/GPU37℃/内存13.2MiB，全100%。单元差分测试CPU峰60.875℃/GPU36℃/内存299.1MiB。温控与GPU遥测保持，无训练/优化器更新或CPU训练fallback。

该版本仍逐候选执行Python字典/循环；本次没有做逐函数剖析，不能确定其具体开销占比。下一可研究不同的批量NumPy整数计算：一次处理所有候选的行/列与特征，减少逐候选Python调用；这须在新的隔离副本/来源身份下实现并重新预声明验证，不能直接修补本版并重复计时。尚未证明批量方案正确或更快。

证据：runs/autonomous_delta_features_20261106/runtime_archive.json、unit_guard/、micro/design.json、micro/microbenchmark.json、micro/qualification.json、micro/guard/、resource_audit.json、inference_audit.json。实际原生goal执行见native_goal_delta_features_execution_20261106.json。
