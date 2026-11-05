# Adam状态迁移的有界开发试验

## 依据与限制

2026-11-05的LR/参考KL筛选均未通过长局门槛；beta0.3虽在64个既有正常状态保持62个原动作，长局仍仅4/8触顶。不能据此继续盲目增加KL或步数。

核对原PPO30M的config及checkpoint序列化数据，除计划LR外，PPO超参、gamma/GAE、奖励和单局上限一致。原checkpoint保留完整Adam矩与步计数586000，先前迁移只加载权重并重置Adam。新试验单独检验迁移Adam状态的影响，**这是新假设，不是已证明的退化原因**。

| 项目 | 冻结配置 |
|---|---|
| 初始化与参考 | 同PPO30M SHA eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34 |
| 新变量 | initial_optimizer=restore_state；原强KL组为fresh |
| 其余 | KL0.3、LR3e-5、gamma0.995、GAE0.95、训练单局10000、原网络/奖励 |
| 训练种子 | 1600000–1600007，与历史强KL组相同 |
| 预算 | 先262144步（试运行计入），目标最多2097152，串行控制器总3小时 |
| 开发评测 | 同100000–100007，8×20000块，匹配冻结基线 |
| 初筛门槛 | 8局全触顶且平均已观察得分至少冻结模型99% |
| 后续 | 合格才额外100008–100015配对；通过才按524288/1048576/1572864/2097152分段扩展，退化即停止 |

历史fresh组结果直接复用，避免重跑已完成训练；其数值源码留在5d690fd。新源码仅增加显式初始化选择、配置/恢复校验及来源记录，默认fresh继续保持原初始化语义。该比较属于历史开发控制，不是同步重复试验或独立确认；重复使用开发种子存在选择适配风险。最终测试集不使用，正式模型不替换。

## 实现与验证

新增`training/initialization.py`，restore_state复制权重、Adam矩和计数，但保留新run配置的学习率及其他optimizer选项；深复制避免与冻结参考共享可变状态，拒绝空/非有限状态或不兼容参数。环境、RNG和PPO步数仍是新run，不能称为精确续接原30M训练。正常事务恢复继续保存全部新run状态，不再重复初始化。

初始化模式写入配置并在resume核验；初始化实现加入训练来源哈希，共21文件。旧来源事务不会静默迁移到新代码。12项定向测试通过，覆盖矩/计数/LR、源状态独立、fresh默认、非法输入、来源变化和参数筛选。

真实CUDA8：4096步→退出→resume8192步验证通过，8worker快照齐全、18份Adam参数状态计数586000→586160、LR3e-5、参考SHA不变。证据`runs/autonomous_optimizer_transfer_20261105/smoke_verification.json`。

首次测试命令因错误文件名未运行，已保留失败后修正；第一次训练试运行在元数据阶段因临时systemd PATH缺Node退出，尚未开始训练或提交。原日志与config保留，重试使用与持久服务一致的PATH并成功。不是额度或GPU故障。

## 接手位置

- 控制器：`tetris-optimizer-control-20261105.service`，目录`runs/autonomous_optimizer_transfer_20261105/`。
- 接力：`tetris-optimizer-followup-20261105.service`，目录`runs/autonomous_optimizer_followup_20261105/`，当前等待比较。
- 合格后的新目录：`runs/autonomous_optimizer_extra_20261105/`及`extension/`。
- 保持lightbook、CUDA8、总CPU250%/内存6GiB、原温控与有限时间预算。查看实际MainPID/status/guard，避免重复启动；运行中来源与支持文件冻结。

尚无本组长局结果，不能宣称恢复Adam已经改善棋力。

## 首轮结果与实际自动接力（17:18）

262144步、同8×20000开发比较完成：8/8触顶，平均已观察块数比1.0、得分比1.0019875537，得分6胜2负，通过冻结门槛。历史fresh强KL组4/8触顶、得分比0.684014；这里只支持本次保留Adam的候选继续研究，不证明Adam重置是所有退化的原因，也不构成独立泛化结论。双方触顶均右截断，不宣称真实寿命提高；约0.20%得分差异也未做独立确认。

17:17:05接力服务自动复核候选SHA、来源、262144步和门槛后写`handed_off`，自动启动额外8种子配对与条件续训服务。实测核验：接力MainPID0/Result=success；pair实例MainPID93337在baseline阶段；extension实例MainPID93341正常等待上游状态。不是需要人工确认的暂停，尚未开始额外训练。后续必须完成100008–100015的基线及候选配对并通过原门槛，再恢复原事务；失败自动止损。

证据：`runs/autonomous_optimizer_transfer_20261105/summary.json`、`restored_optimizer/stage1_8x20k.json`；`runs/autonomous_optimizer_followup_20261105/summary.json`；`runs/autonomous_optimizer_extra_20261105/status.json`及`extension/status.json`。当前源码与支持文件继续冻结，不在活动试验中改评测器。
