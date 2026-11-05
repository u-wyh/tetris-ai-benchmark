# 完整上下文显式降学习率

## 新假设和边界

此前原LR完整上下文262144步长局筛选7/8触顶，未过门槛。新单臂固定原Adam、8worker、全局RNG、PPO更新计数、奖励、gamma0.995、GAE0.95等，仅在真实源模型恢复后将LR从0.0003明确降为0.00003，不使用KL或重新生成环境。历史失败实验只作背景，不冒充本轮严格因果对照。

`initial_context.learning_rate_after_restore`必须显式声明有限正数，且只支持不超过源LR的值。原config/metadata/task manifest照常冻结；先验证实际源模型配方，再改学习率标量、常数schedule和Adam组/default的LR，随后验证目标配方。未声明时仍要求完全相同，其他PPO字段、来源offset、版本等不放宽。报告分别记录源/目标LR及原PPO更新数。

## 验证

- 42项受保护初始化、来源及实际CUDA模型测试通过。
- 独立手工恢复后改LR的正式环境参考，对照新入口bitboard环境；两方各8192步。
- 4096和8192检查点policy、Adam、全部Python/NumPy/CPU/CUDA RNG及8worker完整状态精确一致，只规范化预声明的30003200步偏移。
- 新入口任务2真实退出后在另一进程恢复；Adam从586000继续递增、PPO从36625继续更新，所有组LR保持0.00003。原文件及独立来源副本SHA不变。
- `runs/autonomous_context_lr_20261105/probe_summary.json`于22:10:50通过，父服务22:10:54完成。该验证不是棋力提升结论，旧来源缺SHA的限制仍保留。

## 无人值守筛选

已验证8192断点复制到独立 `runs/autonomous_context_lr_screen_20261105/full_context_low_lr/training`，由持久参数筛选服务继续至262144新增步，再用bitboard执行8×20000配对门槛。

复用 `runs/autonomous_bitboard_baseline_20261105/baseline.json`。训练器改动后，实际推理20文件身份和运行库仍匹配，已验证，因此没有重复长基线或重标旧结果。

接力 `tetris-parameter-followup@autonomous_context_lr_screen_20261105.service`等待完整筛选；不合格自动止损，合格才启动额外8种子配对。`extension_enabled=false`，本轮上限262144，不自动2M或发布模型。所有数值和支持输入已冻结，运行期间不修改。

GPU8、总CPU250%/6GiB与原温控保持。评测固定2worker及较低CPU配额。全部服务、工作副本、断点和小时续办留在lightbook；用户暂停取消或硬件不可用时按既有保护停止和保留进度。
