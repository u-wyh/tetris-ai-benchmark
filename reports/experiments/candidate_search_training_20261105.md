# 位图落点搜索：真实CUDA8训练验证

## 设计

原训练使用正式BFS及位图特征。新选项仅将搜索切换为既有`training/evaluation/fast_legal.py`，特征、PPO、初始化模型/Adam/KL配方、奖励与资源预算相同。独立lightbook副本先预检188项，再按正式/位图、位图/正式两组顺序，每组每后端8192步，4096后真正退出并跨进程恢复。种子1900000、1900100；全部GPU8、CPU250%/6GiB/原温控。

事前要求两组均更快、几何平均≥1.05，且没有温控降载；完整policy、Adam、PPO更新计数、trainer RNG、8worker状态及除`wall_seconds`/`gpu_peak_allocated_mib`外的指标必须精确相同。只作性能和恢复验证，不是新棋力实验。

## 结果

| 种子 | 正式搜索训练秒 | 位图搜索训练秒 | 吞吐倍数 |
| --- | ---: | ---: | ---: |
| 1900000 | 15.881 | 14.530 | 1.092980 |
| 1900100 | 15.611 | 14.626 | 1.067346 |

23:40:52全部通过，几何平均**1.0800869（约+8.01%）**。计时为训练metrics中的任务时间，包含学习；不含进程冷启动/监控等待，不能当作整命令或长期固定收益。所有要求的状态精确相同；错误搜索preset恢复被拒绝，原config/metadata/提交状态不变。训练采样最高CPU77.5℃/GPU54℃，进程组内存3583.05MiB，所有CPU配额样本250%。

随后单独验证合成正式搜索源断点→完整上下文显式降LR/位图搜索→完成→扩展→故意中断→恢复，与连续控制组精确一致，覆盖offset/lifetime计数。23:46:58完成，1项通过/2项未选择；85.04秒测试，采样最高CPU75.75℃/GPU53℃、3920.77MiB、配额250%。没有扩展此前被拒绝的候选。

## 采纳与来源

采用显式`candidate_search_backend: bitboard`，需要普通fast Candidate环境和`candidate_feature_backend: bitboard`；默认formal保持。配置、metadata及preset恢复固定搜索模式。新训练来源闭包25文件；主副本集成后与实测副本逐项SHA比对，不靠Git提交号相近推断一致。

原23数值/20推理代码归档在`/home/wyh/projects/tetris-runtime-65fd092`，带本机venv/runs链接，旧实验继续使用原代码身份，不能用新25来源直接恢复旧23断点或重标旧缓存。推理20文件及运行库没有变化，原bitboard长局基线身份保持，未重复评测基线。新实验可显式选择已验证搜索模式；该性能验证不构成模型提升或旧失败配方通过质量门槛的证据。

证据：`runs/autonomous_candidate_search_20261105/gpu_design.json`、`gpu_summary.json`、`gpu_resource_audit.json`、`adoption_gate.json`；`runs/autonomous_candidate_search_context_20261105/`。主副本接入的确切SHA核验另记`integration_audit.json`。
