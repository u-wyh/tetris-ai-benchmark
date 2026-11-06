# 当前位图搜索构建剖析与存储测量

2026-11-06受保护CPU100%/2GiB静态测量，无训练。两个已保存8环境观测的candidate均为uint8，形状[8,1840,16]；4096观测的密集candidate矩阵为115MiB。已安装SB3的DictRolloutBuffer.reset按observation_space对应dtype分配，不能误按float32估成460MiB。两个batch中非零特征行每环境26–68，仅为快照稀疏度，不是独立验证的合法mask计数。相对实际训练3.5GiB左右组峰内存，这不足以优先投入完整packed-buffer重写，尚无实测吞吐收益。

更新profile_candidate_construction.py支持显式--search-backend bitboard，默认formal保留。原源task7325与新候选task2共16状态、30次重复、480次构建，cProfile CPU2.14578秒：fast_legal._branch累计1.43313秒（约66.8%），其中自身0.86220秒；位图特征累计0.67120秒。它包含剖析开销，排除env推进/IPC/策略/PPO，不能当训练吞吐结论。

下一隔离假设：使用紧凑的visited状态表并跳过已访问邻居的重复碰撞查询，保留BFS邻居顺序、SRS踢墙、越界/负y、action排序/去重和路径。以当前bitboard为对照，新增显式bitboard_compact训练搜索后端；默认formal与原bitboard保留，推理20不变。先全部placements/action/path/features/masks差分与边界回归、预声明静态ABBA，再实际CUDA8两seed AB/BA各8192跨4096进程恢复。policy/Adam/PPO/RNG/8worker/非计时指标必须精确同；几何吞吐≥1.05且每对更快、无温控降载才考虑采用。若收益不足保留结果，不把cProfile占比或内存理论值当性能采纳依据。

证据：runs/autonomous_rollout_storage_profile_20261106/footprint.json及guard/；runs/autonomous_bitboard_construction_profile_20261106/profile.json及guard/。此前actor一阶重置候选长局得分比0.8658487295、6/8触顶，已拒绝，无extra8/追加训练/推广。
