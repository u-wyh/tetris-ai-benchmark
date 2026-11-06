# 实际CUDA8采样与PPO阶段剖析

2026-11-06 10:17:05 profile passed，父控制器10:17:08 completed。保持当前main26来源、旧bitboard搜索，原30M task7325完整环境/Adam/RNG上下文；一次4096样本、独立无剖析/剖析两副本各实际PPO.train。原模型/Adam/PPO更新数保持，0原模型优化器步，无普通检查点/质量候选。

采样指纹800cc7e3d629cb86840e66608313a3224a49280b51d4a8dc055acf6f83e5e7b0与已完成原4096只读审计精确相同；两PPO副本policy/Adam/epoch计数精确一致。真实train backward各80次。55冻结输入未改，GPU8/CPU250%/6GiB/原温控，180秒命令/600秒队列。

## 观察

| 阶段/函数 | 剖析墙钟秒 | 说明 |
| --- | ---: | --- |
| 4096样本采样（512向量步） | 6.00647 | 阶段边界同步CUDA |
| 一次5epoch PPO.train | 0.72780 | 固定同rollout，80次backward |
| SubprocVecEnv.step_wait累计 | 3.52975 | 包含等待worker计算及传输，不能当纯通信开销 |
| posix.read自身 | 3.41699 | 16400次，跨调用嵌套，不能与上述累计相加 |
| Candidate policy.forward累计 | 1.60005 | 512次，包含主机端CUDA等待 |
| 单独get_action_masks累计 | 0.22692 | 512次，仅约剖析总阶段3.4%，不是主要剩余收益 |
| PPO数据_get_samples累计 | 0.15768 | 80次 |

cProfile有开销，只剖析主进程、不含worker内部；累计项嵌套，CUDA同步的阶段墙钟不等于GPU繁忙时间。此测量用于选择下一假设，不能与其他无剖析运行直接比较速度，也不证明棋力。原完整上下文虽按当前代码严格加载，但原30M历史metadata缺训练来源SHA的既有限制保持，不能声称原历史代码完全复现。

## 决定

不优先重写packed rollout buffer（密集uint8仅115MiB）、也不单独缓存mask RPC去追≥5%收益。此前compact搜索静态+7.1%而实际+2.1%，不采用，main维持旧bitboard。

下一隔离尝试每个旋转/纵坐标用10列位图预计算可用横坐标，减少逐位置碰撞函数与LRU查询。必须保留搜索顺序、动作/path、初始负y/旋转kick边界；先归档旧9144a36的准确27文件运行代码，使用新目录和冻结身份，不能重标旧断点。静态差分/ABBA充分后才执行新的预声明GPU8两seed2100000/2100100，几何≥1.05且每对更快、全状态精确及完整上下文扩展/故障恢复才考虑采用；未开始新训练。

证据：runs/autonomous_training_stages_20261106/design.json、training_profile.json、resource_audit.json、summary.json及guard/。
