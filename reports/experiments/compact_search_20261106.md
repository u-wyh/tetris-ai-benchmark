# Compact visited bitboard search（隔离验证中）

## 范围与预声明

lightbook本机隔离副本`/home/wyh/projects/tetris-search-compact-validation`，实现提交9144a36。显式`candidate_search_backend=bitboard_compact`保留formal默认与旧bitboard。使用有界bytearray访问表，先排除已经访问的邻居再查询碰撞；BFS Left/Right/RotateCW/Down顺序、动作去重/排序与路径保持。超出访问表范围的罕见初始坐标转原搜索。新增模块纳入27训练来源，当前主副本26与推理20未改。

两组seed2000000/2000100，旧bitboard与compact按AB/BA，每臂8192、4096跨进程恢复。要求policy/Adam/PPO更新/全部trainer RNG/8worker/非计时指标精确一致，只有wall_seconds及gpu_peak_allocated_mib按既有协议排除。实际训练吞吐几何平均≥1.05且两对都更快，全程CPU250%无温控降载；不达门槛不采用。GPU8/6GiB/原温控/命令300秒、队列1800秒，47冻结输入见gpu_design.json。训练目标本身8192，无额外扩训/棋力推广。

## 已完成

- 初轮222项差分/边界/配置/原环境回归通过，包括3136组墙/负y位姿；后续36项含新增compact实际环境逐步trajectory/特征/奖励/状态/快照与配置回归通过。
- 27状态（原30M8worker、新actor8K8worker、失败低LR8worker、3边界）×30 ABBA静态完整构造：中位1.0713309765、最小1.0488263335。所有有序placement/path/features/mask精确。CPU100%/2GiB保护。
- 静态独立design构建tuple/list类型错误导致未生成；shell随后执行了受保护基准。门槛此前工具调用已声明、输出源码/快照SHA事后独立复核，micro_audit如实记录。不能称该微基准完整事前冻结，更不能把其7.1%当实际训练收益。
- 首次两个transient服务遗漏用户Node PATH，metadata初始化FileNotFoundError(node)，没有提交训练事务。partial归档failed_initialization_missing_node，保留原guard；补标准PATH后的retry01沿用不变数值/配方/门槛，未删除失败。

## 当前

真实对照`tetris-compact-search-benchmark-retry01.service`执行中；`tetris-compact-search-extension-retry01.service`在独立retry01目录串行等待，然后验证完整上下文完成→扩展→故障恢复。只在完整summary通过后报告结论，不用启动/accepted代替完成。

证据：`runs/autonomous_compact_search_20261106/`、`runs/autonomous_compact_search_extension_20261106_retry01/`。

## 终态：不采用

GPU对照10:13:24 passed一致性，eligible=false：seed2000000旧14.748秒/compact14.490秒，1.01780538；seed2000100旧14.566秒/compact14.220秒，1.02433193。几何1.02106344，即约+2.1%，低于预声明≥5%，不采用到main。policy/Adam/PPO计数/全部RNG/8worker/非计时指标均精确，错误backend恢复拒绝、config/metadata/transaction_state字节未变。47冻结输入未改。

后续合成扩展验收刚于10:13:29启动，因采用门槛已经失败，于10:13:45主动停止，guard10:13:46 interrupted、服务MainPID0，保留现场；不声称完整上下文扩展通过，也不为未采用方案继续耗费GPU。先前服务Node失败记录保持。主副本26来源/推理20和旧bitboard保持，隔离实现9144a36与全部证据保留。

静态搜索构造约+7.1%未转化为端到端≥5%。后续若继续性能工作，应先量实际rollout/IPC等待/策略/PPO各阶段，而非根据静态构造百分比继续堆叠搜索修改；没有棋力改善结论或新候选推广。
