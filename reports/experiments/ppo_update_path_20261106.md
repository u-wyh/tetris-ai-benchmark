# 真实 PPO 顺序更新与 KL 提前停止诊断

## 协议与证据

2026-11-06 09:25:53 passed，控制器09:25:56 completed/MainPID0success。代码682f88e；49输入冻结含实际安装的sb3_contrib/ppo_mask.py，25训练/20推理来源未改。历史原训练metadata缺源码SHA限制保留。

CUDA＋8环境采集同一原上下文4096 rollout，SHA及16批全部标量摘要与原梯度审计相同。实际调用MaskablePPO.train，各副本最多5epoch/80步，共同保留原policy/Adam/RNG；探针为首批64训练状态，选取不消耗首次训练置换。4种LR/targetKL之外，另跑原LR无追踪控制；追踪组与控制的policy、Adam、PPO更新数精确相同。原模型优化器0步，原policy/Adam/PPO计数不变，无普通检查点或棋力评测。

9项保护测试覆盖参数/Adam双向隔离、RNG保留、buffer顺序、失败后RNG/原模型恢复。实际CUDA运行也验证追踪无数值影响与原rollout/标量摘要一致。

## 单个rollout结果

|学习率|target_kl|实际步数|提前停止|64状态最终KL|峰值KL|最终贪心一致率|
|---|---:|---:|---|---:|---:|---:|
|3e-4|无|80|否|0.00551154|0.0183125|0.953125|
|3e-4|0.003|5|是|0.00868135|0.00868135|0.9375|
|3e-5|无|80|否|0.00116270|0.00116270|1.0|
|3e-5|0.003|80|否|0.00116270|0.00116270|1.0|

原学习率轨迹峰值在第15步，之后回落；提前停止组第6个minibatch的近似KL0.00517944超过原生阈值1.5×0.003，故只更新5步。它降低本次路径最大扰动，但停下位置比80步终态更远，不能视为质量改善。低LR的门槛未触发，观察指标相同；不因该诊断重复已失败的低LR训练。

这些KL来自固定训练态，动作一致率不代表长局存活。原生minibatch近似KL也不是严格的全局KL约束，门槛与probe KL不可等同。不能以此证明长期退化原因。

## 下一步与资源

不直接采纳targetKL。结合上一单步诊断，下一独立副本诊断可仅清零actor Adam的exp_avg，保留exp_avg_sq/step/所有critic状态，对比两种学习率完整更新路径，检查能否减少早期动量偏移而避免fresh Adam的归一化突变。先在副本验证，不改原模型或训练配置；有支持证据才考虑完整可恢复训练/质量门槛。

共享CPU250%/6GiB/swap0，原温控；CPU最高71.875℃、GPU52℃、cgroup内存3553.89MiB，采样全250%，49输入前后不变。证据runs/autonomous_ppo_update_path_20261106/update_path.json、update_steps.jsonl、update_cases.json、resource_audit.json及guard/。
