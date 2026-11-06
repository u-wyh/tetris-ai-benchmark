# 首步失去四行后的有限分支对照

全部15个既有上下文，每个克隆两臂，首步分别强制保存的原策略动作（清4）和8k策略动作（清0）；之后**两臂都跟随同一个冻结原30M CUDA greedy策略**。固定256步含首步，共30条分支，未调整horizon或挑选子集。

## 协议与实现

GPU＋8环境，每批固定4对及槽序，尾批3对/6 active槽。通过诊断env_method直接推进active worker，死亡/TimeLimit/horizon后停止，空槽不初始化/不参与模型forward；不调用自动reset的vector step。每臂初始完整Core/bag/episode/numpy/action RNG与保存上下文精确相同，hold后队列差异属于首动作影响。

每pair原子保存trace、终端完整状态和身份。整批receipt完成后才可跳过；若中断留下partial pair，重放同一完整批次并比较已有结果/state，不覆盖。这是为保持CUDA float32批次稳定；不能恢复时把余下pair重新分组。采用最高float32 matmul精度、TF32关闭、greedy不消耗采样RNG。5项定向测试通过。

## 结果

| 共同256步结果 | 数量/差值 |
| --- | ---: |
| 候选首步分支清行更少 | 4对 |
| 清行相同 | **10对** |
| 清行更多 | 1对 |
| 候选−原累计清行 | **−7行** |
| 累计清四行事件差 | −1次 |
| 原环境累计reward差 | 约−14 |
| 观察到死亡/TimeLimit | 两臂均0 |
| 到达horizon删失 | 两臂各15 |

15个首步合计即时少60行，到256步时差距合计为7行。这个局部协议下，原策略后续经常补回首步清行损失；仍有4对清行更少，不能称首步无害。没有观察到死亡，也不能证明更长生存等价。上下文来自原策略随机采样轨迹、经greedy分歧/失去四行筛选，彼此时序相关；不代表一般状态分布。

**这不是让8k策略持续决策的对照。** 因而不能把整体8k退化归因于一次漏清行，也不能据此排除持续动作漂移的累积影响。未推广模型、调整奖励、开启PPO或延长本次horizon。

## 执行与恢复证据

进程388410完成首批4对（2048环境步/255 GPU预测批次），进程388487复用首批完整文件，再完成其余11对（5632步/765预测批次）。首四pair和group receipt SHA未改。总计**7680实际模拟环境步、1020次GPU预测、0优化器步**；每组源模型policy/完整Adam/PPO计数及CPU/CUDA sampling RNG不变，计数30003200/_n_updates36625。

86冻结输入未改，32.812秒；branch_summary16:32:19 passed、restart_receipt16:32:20 passed、父guard/controller16:32:23 completed。CPU峰72.75℃/GPU49℃/内存3431.32MiB，全250%无降载、6GiB与原温控/时间预算。单测100%/2GiB，CPU37.875℃。主训练31、推理20和固定编译产物保持。

## 下一步

复用本次A/B结果（两种首步后均原策略续走），保持相同15上下文/批次/256步，新增C/D两臂：同样两种首步，但之后都由冻结8k策略续走。使用同一未修改runner、只显式换固定续走模型，检查第一步物理结果仍4/0及模型状态不变。

按相同首步分别比较A与C、B与D，在两臂共同步数上比较reward/清行/四行事件，并单列观察到的死亡与删失。这能隔离该局部实验中的“首动作”与“后续策略”，仍不等于整体训练质量证明。当前C/D尚未开始，不重复A/B或prefix。

证据：runs/autonomous_rank_branches_20261106/{design.json,unit/guard/command.log,branch_summary.json,pairs/,group_00.json…group_03.json,process_388410.json,process_388487.json,restart_receipt.json,resource_audit.json}；本原生goal续轮首工具16:19:47见native_goal_rank_branches_execution_20261106.json。
