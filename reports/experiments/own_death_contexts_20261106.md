# 自身死亡轨迹：最后模式分歧与完整状态恢复

## 无人值守续办已实际执行

19:04:55小时投递`01a1b088-7bec-7902-af0c-36c1a33c5358`在19:37:17被新轮`01a1b0a6-18c7-73e3-96f2-6bc8108c19b6`消费。新userMessage.clientId与原queue的clientUserMessageId一致；queue已空，rollout中的新turn_context后19:37:28.139出现真实exec调用`call_0352af07bc624082a3f68bf0e4c7b100`。

这比accepted提供了更进一步的真实执行证据，不外推无限额度。后台13827→1、四个驻留/续办/防休眠单元正常、原goal active，执行machine-id为指定lightbook；未重新测试OS重启或短时暂停。证据：`runs/autonomous_development_20261105/hourly_own_death_context_execution_20261106.json`。上一轮`ced99ee`的144936步完整捕获没有重跑。

## 选择规则与静态结果

读取已通过完整审计的8k自身轨迹，**对每一个真实死亡，选择死亡前最后一个候选动作不同于原30M旁观动作的决策**。包括全部两次死亡，排除6个cap删失轨迹；不是挑选原策略15处漏四行状态。分段数组仅物化一次，未执行新环境步、策略forward或optimizer。

| seed | 最后分歧动作step | 前状态round/pieces | 原死亡step | 从当前动作到死亡（含当前） | 原片段/前快照 | 8k/旁观action |
|---|---:|---:|---:|---:|---|---|
| 100002 | 17823 | 17822 | 17994 | 172 | 70/69 | 1510/382 |
| 100005 | 6921 | 6920 | 6942 | 22 | 28/27 | 1665/102 |

两处候选动作使用hold，旁观动作不使用hold；两边预测均清0行，孔洞和aggregate高度分别相同。候选/旁观最高概率：seed100002为0.439899/0.422639，旁观赋予已执行动作0.417513；seed100005为0.236652/0.220189、已执行动作0.207723。hold、使用piece和落点一起变化，尚未隔离hold的因果作用。

选择器3项定向校验通过：全部死亡不遗漏、最新分歧顺序/1-based动作和前状态偏移、当前动作即死亡的边界。静态选择结果与保护记录见`runs/autonomous_own_death_targets_20261106/`。这些描述不证明改动作能救活，也不推广8k模型。

## 恢复协议与验证

新增独立`restore_own_death_contexts.py`，不修改已冻结的捕获helper/runner/design或训练31文件。外部冻结manifest SHA和state/rows文件SHA先校验，再加载自己的pickle；从chunk27/69的全部8worker快照恢复，包括已死槽、Core/bag/游戏/NumPy/action RNG和active观测/mask。

只重放原保存的candidate动作；每步核对观测/mask摘要、实际选中与旁观特征、reward/终止/seed/pieces/孔洞/高度。所有目标在当前轮任何槽推进前提取，状态必须是step−1；整段末全部worker状态和active观测/mask与原capture快照精确，才能提交完整单worker上下文。模型不加载，不称为GPU策略推理；CUDA必须可用并保持遥测，无CPU训练或其他主机fallback。

3项恢复定向校验通过，包括真实forkserver重建、真实死亡槽保持停止、完整状态RNG、坏manifest/数据在pickle前拒绝、观测/mask和预算拒绝。正式两片段预声明最多512轮/4096实际模拟步；先独立进程重放chunk28，再第二进程复用其context/receipt，仅重放chunk70并校验首批SHA未改。250%/6GiB/原温控、180秒命令/420秒wall；0新策略采样/forward/optimizer。

19:46:09正式恢复启动；完整结果待保护器和父controller实际结束后追加，不预写通过。证据：`runs/autonomous_own_death_contexts_20261106/`。

## 已完成结果

上下文19:46:29保存、restart19:46:30通过；保护器19:46:32完成、父controller19:46:33完成。进程422039重放chunk28的1822步，进程422089复用其receipt/context，只重放chunk70的1792步；共3614实际环境步/512轮、0策略forward/新动作采样/optimizer。首批文件SHA未改，全部153冻结输入未改。

每步观测/mask摘要、实际与旁观所选特征、reward/终止/episode seed/pieces/孔洞/高度精确；两段末完整8worker/Core/bag/游戏及NumPy/action RNG、active观测/mask与原快照逐项一致，父采样RNG不变。chunk28真实重现seed100005死亡，剩余轮停止该槽；chunk70恢复并保留此前已死槽，没有自动重开局。

恢复材料为`contexts/seed_100002_step_17823/`及`contexts/seed_100005_step_6921/`中的完整`state.pkl`与外部manifest。可通过已固定`restore_rank_contexts.load_context(directory, expected_identity)`校验SHA后加载，expected_identity为`contexts_summary.json.identity`再加对应`target`。两者状态的max_pieces均为20000，后续分支环境必须相同，不能误用默认10000。

| 阶段 | 保护器完成 | 秒 | CPU配额/内存上限 | 冻结输入 | 采样CPU℃ | 采样GPU℃ | 采样内存MiB |
|---|---|---:|---|---:|---:|---:|---:|
| 静态选择和3测试 | 19:40:28 | 4.121 | 100%/2GiB | 137 | 54.875 | 37 | 235.69 |
| 恢复3测试 | 19:44:44 | 8.191 | 100%/2GiB | 140 | 54 | 38 | 741.05 |
| 正式8worker重放 | 19:46:32 | 20.603 | 250%/6GiB | 153 | 73.5 | 45 | 2846.84 |

全部完成、无温控降载，原温控/内存/时间保护未放宽。表内是2秒间隔的采样峰值，不是连续真实峰值。GPU在本重放中只用于可用性/RNG恢复与遥测，没有GPU策略推理。

19:48:05后台仍healthy、四单元enabled+active、linger yes、sleep masked、proxy为空；19:48:25 queue为空。本轮小时消息已实际消费并完成上述开发，不能把此前19:33仍排队的旧时刻与当前状态混同。未新启动训练，也没有局部因果/全局质量改进结论。

下一可在这两个**自身**状态预声明四臂有限干预，固定首动作分别为原旁观/8k动作，再分别由原/8k续走；必须校验源8k路径的重现，区分局部死亡、256步删失与全局质量门，不重复之前15处参考轨迹分支。
