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
