# 固定原上下文 critic 校准与持久样本

2026-11-06 14:02:35诊断passed，14:02:37保护器/父控制器completed；不是新训练。原完整上下文task7325、CUDA＋8环境、4096样本，历史rollout SHA `800cc7e3d629cb86840e66608313a3224a49280b51d4a8dc055acf6f83e5e7b0`精确相同，原policy/Adam/PPO更新数不变。当前row masks/27训练来源，52冻结输入完成后逐项未改；历史原始训练代码未保存SHA，不能声称当时源码已知。

## 可复用数据

`runs/autonomous_critic_calibration_20261106/rollout.npz`约2.2MiB压缩数据，SHA `124e0495b682d249942bc535a2b21667e2ad72b6a74934eb95f84bd9050ef3c1`。包含完整观测、动作/mask、reward/value/return/advantage/log_prob、episode_starts、末值/dones、逐step终止/截断与新孔洞/高度标记；轴顺序time,worker。另存环境原奖励，buffer奖励保留SB3 timeout bootstrap语义。

不允许覆盖既有样本。无pickle回读、逐数组shape/dtype/SHA验证，并用已安装SB3精确重算advantages/returns；6项受保护测试覆盖常偏差与方差、终止/末值/奖励损坏拒绝和持久化拒绝覆盖。元数据保存来源、配置哈希、运行库及全输入哈希。后续分析直接复用，不再次采样该4096窗口。

## 结果

残差定义为GAE return减当前value。

| 指标 | 数值 |
| --- | ---: |
| 残差均值 | -0.896633 |
| 残差标准差 | 7.168626 |
| RMSE | 7.224482 |
| 解释方差 | 0.482116 |
| 均值偏差平方/MSE | 1.5403% |
| 新孔洞step | 46/4096 |
| 真终止/截断 | 1/0 |

保存后的只读探索性分解14:03:36 passed：worker4贡献97.4008%平方误差，唯一终止在该worker的step237；含终止及同episode前31步的32条样本贡献88.1861%。误差最大的ceil(1%)=41条贡献92.6619%；46条新孔洞step贡献29.0397%，这些分组重叠，不能相加。其余4064条RMSE2.492912。其余7个worker的RMSE约0.67–2.22。

这不支持“整体常数偏差主导”的解释。误差明显集中在一次失败附近，尚不能将这种集中直接等同于critic梯度占比或PPO质量下降原因；梯度取决于网络Jacobian，Adam和全局裁剪还有额外影响。暂不采用全局bias修正、critic预热或robust loss，也不启动新PPO/GAE/LR网格。

局限：单个固定窗口、时序相关、仅一次真终止且无截断；GAE目标有bootstrap，不是无偏完整Monte Carlo回报。终止窗口与误差top1%是探索性描述，无独立验证或因果结论；终止分组只标当前transition。不能据此宣称学习质量提升。

## 保护与下一步

采样CPU峰65.125℃、GPU41℃、组内存3670.1MiB，采样全250%无降载，6GiB/swap0/原温控不变。轻量尾部分析CPU100%/2GiB，保持GPU遥测，是数据分析而非CPU训练fallback。全部计算现已结束，无checkpoint或质量推广。

下一先检查既有梯度审计能否将失败尾部与critic梯度对应；缺少对应证据时，用本持久样本做有界只读梯度分解，区分样本平方误差集中与实际梯度/裁剪作用。原policy/Adam/PPO0更新，不重采样、不跑质量候选；先有证据再选择学习方案。

证据：同目录design.json、dataset.json、calibration.json、tail_analysis.json、resource_audit.json、summary.json、unit_guard/、guard/和tail_guard/。无人值守本次真实小时执行另见runs/autonomous_development_20261105/hourly_critic_execution_20261106.json。
