# 15个失去四行上下文的完整恢复

复用已提交initial与4段末端worker快照，恢复几何审计中**全部15个delta_cleared=-4**时刻。没有按结果挑选、重采样策略动作或训练。

## 恢复方法及校验

只反序列化本机自有、checksum验证后的快照。新独立诊断使用当前准确31来源与固定native产物，不改原27来源checkpoint metadata，也不把它作为新来源训练断点恢复；只恢复worker状态。

每段从其之前的initial/段末快照装入8环境，重放原保存actions；这里自动reset是为了精确重现原采样episode边界。逐步核对完整state/candidate观测、合法mask、原环境reward、done/truncation、episode_seed、pieces、清行/孔洞/高度；每段末完整worker状态（Core、bag/gameplay RNG、numpy/action RNG、episode计数等）与原末端快照精确一致。

每个目标在执行该步前保存完整worker状态、行身份、观测/mask摘要、state SHA和设计来源；两动作合法且候选特征分别清4/清0。部分输出保留、错误身份拒绝、已提交上下文不覆盖；保存后pickle往返一致。

4项定向检查通过，包括真实forkserver重放、3次自动episode reset和完整RNG、校验和在反序列化前拒绝、上下文不可覆盖与选择行对齐。

## 实际执行和续办

| 进程 | 新重放段 | 复用段 | 环境步 |
| --- | --- | --- | ---: |
| 385456 | 1、2 | 无 | 8192 |
| 385535 | 3、4 | 1、2 | 8192 |

原前两段receipt和上下文文件SHA保持，15个上下文全部校验成功。**实际重放环境步16384，策略forward为0、优化器更新为0**；不能称零环境计算，也没有推进任何训练模型的计数。重放是原动作的模拟复现，未产生新策略采样。

## 资源及无人值守

| 阶段 | CPU峰℃ | GPU峰℃ | 内存峰MiB | 配额 |
| --- | ---: | ---: | ---: | ---: |
| 单测 | 55.25 | 35 | 566.40 | 100% |
| 8环境重放 | 69 | 48 | 2743.89 | 250% |

原温控/6GiB/无swap/时间预算保持；执行前CUDA不可用即拒绝。重放不做GPU模型forward，GPU数值为遥测；没有转CPU训练。28.855秒，16:15:29上下文与重启回执passed，16:15:30保护器及父controller完成，69冻结输入未改。主训练31、推理20和编译产物保持。

本次16:03:05小时请求对应新轮01a1afe4-436b-72e3-a1c6-b941ec5825b5，首工具16:05:47实际执行、queue清空。匹配依据是唯一待办/相同提示/新轮顺序，未保留submission/clientId相等映射。后台PID13827→1、keeper、4服务定时器驻留/linger/防休眠正常、桌面proxy空；不把accepted当执行，不声称OS重启或整夜已另测。

## 后续分支

上下文已准备好，下一从这15个相同完整状态各克隆两臂，第一步固定保存的reference/candidate动作，之后都跟随同一冻结原30M CUDA greedy策略，固定总horizon256（含首步），最多7680环境步。8环境分批，每批最多4对；终止/尾批空槽停住，不自动重开episode。

需要独立预声明和定向测试inactive/终止处理、状态克隆、首步4/0、逐对cache、模型/Adam/PPO计数及sampling RNG不变。结果只表示首动作在原策略续走下的局部影响，不等同于8k策略整体退化原因。当前**尚未开始分支**，未启动长训练或改奖励。

证据：runs/autonomous_rank_contexts_20261106/{design.json,unit/guard/command.log,contexts_summary.json,contexts/,chunk_0001.json…chunk_0004.json,process_385456.json,process_385535.json,restart_receipt.json,resource_audit.json}；小时执行见runs/autonomous_development_20261105/hourly_prefix_replay_execution_20261106.json。
