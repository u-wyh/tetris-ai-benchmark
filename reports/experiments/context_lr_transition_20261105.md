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

## 实际结果

2026-11-05 22:29:33筛选完成，22:29:36接力确认`no_eligible_candidate`。候选4/8达到20000块；平均存活13747.875块，相对基线0.68739375，观测得分比0.5898042639。提前结束的种子为100000=6779、100004=5811、100005=3449、100006=13944。未启动额外8种子、扩训或模型替换。证据：`runs/autonomous_context_lr_screen_20261105/summary.json`及对应followup目录。

本轮低LR完整上下文仍未过预声明长局门槛；不能把恢复正确或训练步数增加当作棋力提升。后续不重复此配置。训练采样最高CPU78.875℃/GPU55℃、配额250%；候选评测CPU79.375℃/GPU50℃、配额150%。

## 完成后扩展的恢复回归

新增`tests/test_extend_candidate.py`的完整上下文降LR分支，与原新初始化分支一起验证：独立小型CUDA8源断点→完成→新目录扩展→故意中断→恢复。两者均与不中断控制组比较全部policy张量、Adam状态、PPO更新计数、trainer RNG及8worker状态；全上下文分支还检查源步数偏移与累计步数。2项测试通过（180.60秒），验证输入SHA前后相同。这是合成小预算恢复验证，没有扩展被拒绝候选，也不证明原30M候选获得提升。

证据：`runs/autonomous_context_extension_verify_20261105/summary.json`。首试发现位图夹具漏写`fast_candidate_env`后主动中断；保留原证据，修正后独立`guard_retry01`通过。通过轮采样最高CPU78.375℃/GPU53℃、4079.49MiB，配额250%。
