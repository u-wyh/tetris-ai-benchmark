# 相同硬件预算下的候选特征后端

## 从实际负载选择优化目标

只读采样区间为2026-11-05T17:56:51–17:57:11+08:00，共20.0013秒。CUDA8运行时，8个环境worker消耗36.92 CPU秒、主训练7.54 CPU秒，环境占受保护任务CPU用时83.02%。所有PID及starttime一致，期间提交131072→139264步。这个短区间用于选择进一步测量方向，不是函数级剖析，也不是连续GPU利用率估计。

复核已有实现后选择重用training/evaluation/fast_candidate_features.py的位图特征；正式合法落点生成器保持相同。27个真实训练/清行/出界边界状态，逐项uint8特征及掩码完全一致。30次AB/BA配对CPU微基准中位加速1.7516倍；只计预先给定正式落点后的特征构造，不包含搜索/IPC/推理/更新，不能当作整体训练加速。

## 真实CUDA8训练与恢复

设计在执行前冻结，种子1800000与1800100，分别按official→bitboard和反序运行。每个case4096步→退出→恢复8192步，总32768步；配置target仍为2M，但只运行2个事务，未触发周期/最终测试。

| 种子 | official训练环节两任务用时 | bitboard用时 | 吞吐倍数 |
|---|---:|---:|---:|
| 1800000 | 17.323秒 | 16.267秒 | 1.064917 |
| 1800100 | 17.478秒 | 16.392秒 | 1.066252 |

两个检查点4096/8192在两组种子下的policy全部张量、Adam状态、Python/NumPy/CPU/CUDA RNG、PPO最后观测、8worker完整状态及非计时指标均精确一致。计时指标预声明排除wall_seconds、gpu_peak_allocated_mib；保留各自ZIP SHA，不把ZIP元数据差异当参数差异。

两对几何平均吞吐1.065584，即本机短对照的**训练环节吞吐约提高6.56%**。计时取训练metrics的wall_seconds之和，包含任务内学习，未把进程冷启动/监控轮询当作训练计时；不是长期命令总耗时或通用固定加速保证。两对全程CPU250%、无降载，满足执行前≥1.05且每对更快的门槛。

保护样本峰值CPU76℃、GPU53℃、内存3582MiB，最低系统可用内存8228MiB。所有训练CPU总250%/内存6GiB/swap0、GPU8、原温控保持，未增加资源配额。

## 采用与限制

新增显式candidate_feature_backend=bitboard，配置及resume核验该模式，数值来源新增后端及位图实现，共23文件。默认official保持可对照；后续新实验优先采用已过门槛的bitboard。推荐准备配置configs/ppo30m_guarded_bitboard.json保留CUDA8、原gamma/GAE/奖励、LR3e-5/KL0.3/Adam状态及正确RNG初始化。这是资源与速度已验证的研究起点，不代表该训练配方已通过长局棋力门槛；必须由开发控制器按门槛执行，不直接盲跑2M或重跑失败轨迹。

21项环境逐步/快照/边界/来源测试通过。完整GPU验证服务Result=success，全部释放；未发布新模型、未使用最终测试集。旧PPO30M和所有历史模型保持冻结。

证据：runs/autonomous_rng_isolation_20261105/cpu_process_profile.json；runs/autonomous_feature_cpu_20261105/feature_microbenchmark.json、gpu_design.json、gpu_summary.json、gpu_guards/、gpu_pairs/。

后续优先核查原PPO30M最后完整事务，而非再重复权重迁移试验。已确认task_007325保留trainer/vector快照文件及manifest/worker_digests，原任务记录8个环境，model.zip与冻结final一致，原config为CUDA。原metadata缺少训练来源SHA；必须先审计原运行版本、配置及恢复观测一致性，再决定独立副本的完整上下文续接协议，不能静默迁移或提前声称精确续接。
