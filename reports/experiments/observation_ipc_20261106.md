# 保存观测的纯通信测量：暂不实现共享内存

2026-11-06 14:37:51诊断passed，父控制器/保护器14:37:52 completed、MainPID0/Resultsuccess。**eligible_for_isolated_transport_prototype=false**；没有更改实际环境传输或训练配置。

## 协议

复用已保存原4096样本中time=0,32,…,480的16个观测/worker，共128个观测。8个真实forkserver进程只重放固定数据，不计算环境、特征、策略或PPO。每对交替AB/BA，8对，每种协议512次vector往返，父进程使用已安装SB3的_stack_obs堆叠结果；进程启动、预检和结果校验在计时外。

full发送观测字典/保存的奖励和部分info/空reset_info；cached仅把观测换成索引，父进程用事先已有数组还原。两者其余元数据相同。所有16个索引、所有worker均在计时前逐项验证内容/类型/堆叠一致，计时后再检查最后返回值。选定数据没有终止，测量不包括terminal/reset额外载荷、mask RPC或完整游戏info。cached是**乐观缓存对照，不是共享内存实现**。

1项受保护真实多进程测试验证两种协议的奖励/info/数组dtype及SB3堆叠一致，子进程全部正常释放；58个冻结输入完成后逐项未改。

## 结果

| 项目 | 数值 |
| --- | ---: |
| 每worker观测数组大小 | 30388 bytes |
| full回复pickle body平均 | 30747.94 bytes |
| cached回复pickle body平均 | 126.94 bytes |
| full每512次vector往返时间中位数 | 0.146333秒 |
| cached对应时间中位数 | 0.074413秒 |
| 配对节省时间中位数 | **0.072565秒** |
| 预声明工程潜力门槛 | **至少0.4秒，未过** |

8对均省时，但绝对节省不足以支持当前优先构建复杂的共享内存生命周期/恢复协议。因此停止此transport实现方向，不重复计时。不能把纯wire的相对速度当作完整训练加速，也不能据此精确分摊旧step_wait/posix.read里的时间；旧等待含worker计算，实际流水线还存在重叠和不同载荷。

## 保护与下一步

CPU峰58.25℃、GPU37℃、组内存428.9MiB，全250%无降载，6GiB/swap0/原温控及GPU遥测保持。无游戏推进、模型更新或CPU训练fallback；原30M/main27/推理20/row masks保持。

下一有实质算法变化的候选是增量特征统计：当前每个落点都复制/扫描20行并重新统计10列。可预计算底板统计，无清行时只更新被方块触及的列与相邻高度项；清行、已有满行或特殊边界保留完整重算。必须先隔离实现和逐项差分，后静态与真实CUDA8端到端门槛；未证明新实现正确或更快。旧fused的4.91%负结果保留，不重新计时或直接采用它。

证据：runs/autonomous_observation_ipc_20261106/design.json、pairs.jsonl、ipc_summary.json、resource_audit.json、summary.json、unit_guard/和guard/。原生goal轮首工具14:33:06真实执行见native_goal_ipc_execution_20261106.json。
