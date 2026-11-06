# 动作变化几何与清行核查

这是同状态排序诊断之后预声明的描述分析，只读取已保存431处动作变化及对应底板观测。没有模型forward、环境推进、采样或优化器更新；原排序分区、两个模型和输出未改。91项独立坐标/正式落点/hold/负y/旋转边界/坏数据与占格守恒测试通过。

## 几何还原

1840动作中的x/y是occupied bounding box左上角，不是旋转矩阵原点；依据piece/rotation的CELLS先减min_x/min_y再平移。逐行核对action解码与保存feature元数据，并核对chunk/worker/global_step/episode_seed与原数据。

| 431处变化 | 数量 |
| --- | ---: |
| visible cells不同 | **421** |
| visible/full cells相同，但hold不同 | **10** |
| visible cells、piece、hold全部相同 | **0** |
| hold相同 | 192 |
| piece相同 | 204 |

联合分组：227处同时换hold和piece且占格不同；2处piece相同但hold/占格不同；192处piece/hold相同但占格不同；10处piece/占格相同但hold不同。不能把全部431处归为相同落点的旋转编号变化。占格相同也不证明路径、drop score、queue/hold或完整轨迹等价。

## 即时清行方向

| 候选−原策略清行 | 状态数 |
| --- | ---: |
| −4 | **15** |
| −3 | 2 |
| −2 | 2 |
| −1 | 4 |
| 0 | 402 |
| +1 | 4 |
| +3 | 1 |
| +4 | 1 |

23处减少、6处增加、402处相同，整数总量**−63行**。其中15处原选项清四行、候选选项不清行。这些均为相同原策略采样底板上的合法第一步特征，不是已执行反事实轨迹，不能声称长期净损失63行或更早死亡。

## 特征依赖的精确核对

从保存obs_state前200个binary occupancy格直接数底板块数，逐行验证两种动作：

`aggregate_after = holes_after + board_blocks_before + visible_new_cells - 10*cleared`

431行全部精确。候选−原的整数总量为aggregate +634、holes +4、visible count 0、cleared −63；因此 **634 = 4 − 10×(−63) + 0**。上一报告的更高aggregate主要已由更少即时清行解释，不能把这两项作为独立退化证据叠加，也不能直接推断增加了634个块。

## 资源与边界

测试及分析均CPU100%/2GiB、原温控与GPU遥测；没有CPU训练。静态分析15:59:50 passed，保护器15:59:52 completed、父controller15:59:53完成，41冻结输入未改。主训练31与既有采用审计逐项同，推理20及C产物保持。

现有独立数据还保存initial与每chunk的完整worker状态及hash。这允许下一阶段从已核验快照重放保存动作、逐步核对观测，恢复这15个“失去四行”上下文，再做有限分支对照；无需重新采样一套原策略轨迹。重放和分支都会推进模拟环境，届时必须单独计数，不能沿用本轮0env声明。

下一先验证恢复/重放与每步obs/mask/终止及seed一致，再固定全部15上下文，以原选择/候选选择作为不同的第一步，之后两臂都跟随同一个冻结原策略。该反事实只能研究首步在原策略续走下的局部影响，不能直接归因整套8k策略；固定有限horizon，终止后不自动重开episode，避免挑最好上下文或根据结果更改horizon。未开始该重放/分支工作，未启动训练或调整奖励。

证据：runs/autonomous_rank_geometry_20261106/{design.json,unit/guard/command.log,geometry_summary.json,flipped_geometry.npz,rows_manifest.json,resource_audit.json}。本轮执行见native_goal_rank_geometry_execution_20261106.json。
