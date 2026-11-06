# Actor Adam一阶动量隔离

3d8ed0f独立副本诊断于2026-11-06 09:30:35 passed；7项保护测试通过。原上下文CUDA8同4096 rollout、16批标量摘要、64训练态probe均与上一轨迹相同，原模型0更新，原/低LR各80步。

|学习率|最终KL比（对原动量）|峰值KL比|贪心一致率变化|
|---|---:|---:|---:|
|3e-4|0.970971|0.361875|0|
|3e-5|0.414461|0.414461|0|

仅清零12个actor参数的Adam exp_avg，保留exp_avg_sq、step、group及所有critic状态。达到预声明“最终/峰值KL均降且动作一致率不降”诊断门槛，不代表棋力改善或自动采用。选择低LR作为下一8192步质量筛选起点，是依据更保守的训练态扰动；没有查看新质量结果后选参数。

CPU峰70℃、GPU50℃、cgroup3553.06MiB，全程250%，51冻结输入完成时未改。完整25数值源码已归档本机tetris-runtime-3d8ed0f；旧来源不静默迁移。

后续实现显式initial_context.actor_momentum_after_restore=zero_first_moment，默认preserve；新增training/optimizer_transition.py纳入26来源。必须先验证原源上下文独立手工转换与新事务4096/8192跨进程恢复完全一致，再做合成完成→扩展→故障恢复，且错误preset恢复拒绝/原提交字节不变，才运行8×20000开发门槛。原30M参考继续冻结。

证据：runs/autonomous_actor_momentum_20261106/actor_momentum.json、adoption_diagnostic_gate.json、guard/。

## 显式转换与真实恢复验收

44306c5实现可选initial_context.actor_momentum_after_restore=zero_first_moment，默认preserve；26数值来源包含新optimizer_transition.py。生产转换与诊断转换状态精确同，57项保护回归通过。原25完整源码已归档本机tetris-runtime-3d8ed0f，旧断点不迁移。

原Task7325完整上下文手工参考与新事务4096/8192跨进程恢复于09:40:53通过：policy/Adam/PPO更新/全部RNG/8worker状态精确相同，原文件未变。初始化18个Adam计数仍586000，只转换12个actor一阶状态。GPU8训练段CPU250%，验证段CPU150%；CPU峰81℃、GPU55℃、内存3574.46MiB，未到温控软阈值，冻结输入未变。

09:42:26合成完成→扩展→任务2故意中断→恢复通过，1 passed/3 deselected（87.03秒），完整状态与连续控制相同；错误动量preset恢复被拒绝且config/metadata/transaction_state字节不变。CPU73.5℃、GPU51℃、内存3919.58MiB，采样全250%。两服务均终态success，无重复启动。

## 预声明质量筛选已启动

09:45:38开始评测已完成8192步候选SHA016b6be1bdc320ca1ab2ca31c677924b2511f7884ec3932dc1d73c389af33673（累计30011392）。8×20000验证种子100000–100007，复用同推理20身份的冻结基线原JSON，不重新计算或重标。均存活与得分比≥0.99、失败不多于基线、各seed存活比≥0.9才通过；若通过仍需额外8种子，无追加训练/最终测试/推广。评测2worker/CPU150%/6GiB，原GPU温控/内存保护和1800秒预算；全局STOP/禁用/错误主机检查已补入控制器，5项保护回归通过。

当前结果待完成，不能以诊断/恢复验证宣称棋力改善。证据runs/autonomous_actor_context_probe_20261106/、runs/autonomous_actor_context_extension_20261106/、runs/autonomous_actor_quality_8k_20261106/protocol.json。
