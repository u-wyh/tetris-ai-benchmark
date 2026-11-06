# 当前行位图后端：真实8worker CPU剖析

2026-11-06 10:37:10 passed/completed，8worker各512step，共4096样本，自动reset共1次。GPU8原上下文、原模型/Adam/PPO0更新，采样SHA800cc7e3d629cb86840e66608313a3224a49280b51d4a8dc055acf6f83e5e7b0与历史审计完全同，57输入前后未改。未保存质量模型/普通检查点。

cProfile使用每worker process_time，统计step与自动reset，跨worker CPU和110.7084秒，含很大的逐函数CPU计时开销；不能与无剖析采样6秒比较吞吐，不能把CPU和当墙钟。累计项相互嵌套，排除RPC/序列化/启动/父进程/GPU。

| 函数 | 累计CPU秒 | 调用数 |
| --- | ---: | ---: |
| FastCandidateEnv.step | 110.2339 | 4096 |
| row masks get_legal_placements | 53.8496 | 4097 |
| bitboard candidate_features | 50.4645 | 4097 |
| builtins.sum（各调用者合计） | 35.0566 | 1166977 |

落点与特征均为主要worker CPU工作。特征实现每个candidate分别遍历10列计算holes/aggregate/height/bumpiness/wells/top_four，带多个generator/sum/min/max；范围校验逐项与NumPy float32标量比较。下一在隔离副本尝试合并列统计循环、使用等价的Python整数上限；推理20参考实现保持，训练必须显式后端选择并锁定新源码身份，所有特征/溢出边界/轨迹及GPU完整状态一致才采用。此剖析只提出性能假设，未证明新实现收益。

证据：runs/autonomous_worker_profile_20261106/design.json、worker_profile.json、resource_audit.json、summary.json、guard/。
