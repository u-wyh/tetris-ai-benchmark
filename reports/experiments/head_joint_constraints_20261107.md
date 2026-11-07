# 末层公开表示已封存；单次联合约束求解超时

2026-11-07，lightbook；分类progress。结果为 **unresolved**，不能称可行或不可行，没有候选系数、策略更新或棋力提升结论。

## 实际执行与额度间断

小时12:14:05 submission `01a1b436-af53-7763-9385-b74a4161e43a` 由turn `01a1b439-dd7b-7c82-9c49-ad8e1ddade20` 消费，client `01a1b436-af53-70d2-81b7-524b0fe58ffc`；首工具12:17:53.284+08/call_be05833e77b547f881bc23a1c95bcd99。该轮完成缓存与21测试后因额度中断。

12:32:48旧lp任务在strict require_authorization拒绝，尚未写execution.json、读取真实缓存或调用数值求解器；0真实LP。保留失败guard及完整缓存。15:15:55原小时机制确认ordinary_usage_allowed并恢复原goal；新原生turn `01a1b4dd-4bc2-79f0-b96e-4ea40e50a1bc` 首工具15:16:23.680+08/call_32dd7446e3ab4d37b37d4e5bcb510c6f，未手动修改目标/账户/付费或绕过限制。中间不是持续开发。

15:15:55新submission `01a1b4dd-4bee-7de0-a1d7-1a2415387678` 排队时thread_busy；本轮为原生恢复，不将该accepted说成已消费。health15:18:45 healthy：四个unit enabled/active、linger yes、sleep masked、proxy空；实际后台13827→1。全部工作在machine-id 6b1b552131804ebc918f30005b4d0204，未重启或转移主机，不扩大OS重启/未来额度保证。

## 缓存及来源

新增scripts/freeze_head_representation.py，14项定向测试通过（0.17秒）：外部SHA先于数组加载、封闭公开数组schema、mask/合法动作、来源计数、整个episode训练/验证隔离。正式source599703于12:27:14完成/controller16 passed，只加载原30M模型一次，实际30003200/36625，SHA eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34。

固定CUDA/最高float32、TF32false、batch8；8配对初始公开状态加632正常状态，共80真实forward，无预热、0新环境、0策略优化器。每batch同一forward同时导出最后Linear64→1的输入h/rawscore/真实概率mode；30264合法动作、h形状(30264,64)。全部640 mode匹配预先冻结teacher；四对重复h/rawscore差均0；最大重建误差4.269137640733334e-6，小于事前1e-4。完整policy/Adam/计数/加载后采样RNG和源ZIP不变。

632正常状态来自原30M GPU完整轨迹chunk0..78，80个manifest/state/rows链先验SHA验证、公开观测/mask匹配snapshot及下一chunk首轮teacher；固定train seed100000..3与validation100004..7，各316状态/4episode。同局相关，全部已探索，非独立棋力确认。4个新GPU强有益pair仅训练，seed992000/004/006/008，每pair只取一次，不把重复初始lane加权，也不推广其他173旧标签。只含公开观测h，没有把私有RNG当模型特征。

cache/cache_metadata.json SHA e10cdf221a4da15783cdb3b5e8a488f654e7292da840008771361f0a742ed72c；features.npz SHA 2a992048409c2691a67af4a6d2fcc125054f0ef5cb4c9088cb0cb46c7963d411。320事前输入及controller新增design共321项，完成后SHA均相符，31训练/20推理源未变。

## 隔离求解器与事前协议

本机原venv/系统缺数值LP依赖，先于真实求解记录实现调整：从官方PyPI取highspy1.15.1 wheel，仅解包到本实验solver/artifact，未pip install或升级原模型依赖。wheel5034168字节，SHA 9730647160a6481426729f46d9989a0507d05f3cf96f9fb180f4ab9891bea67b；外部manifest及15个文件逐项SHA、实际import位置/二进制版本验证。原venv包名/版本列表完成后再次精确相同。

[官方Python接口](https://ergo-code.github.io/HiGHS/stable/interfaces/python/)与[PyPI工件](https://pypi.org/project/highspy/)是API/下载来源；本次版本、SHA及执行结果来自本地solver/solver_artifact.json和guard。

新增scripts/solve_head_constraints.py及7测试（0.12秒、3个tiny合成LP），覆盖正负位移最小L1、冲突不可行与超时区别、validation及重复lane不进入训练、pair超过base不等于全局Top1、工件SHA拒绝。正式累计只有1个真实LP，不把合成测试算真实拟合。

冻结原w0和bias，64维位移d拆成128非负变量，最小L1。全部316train正常state的14769个合法竞争动作要求 `(w0+d)·(h_ref-h_a) >= min(w0·(h_ref-h_a),1e-3)`，另4个新GPU good/base要求margin>=1e-3，总14773行。316validation不拟合、不选择目标或阈值。固定simplex、单线程、parallel off、presolve on、seed0、60秒、primal/dual1e-9、small_matrix1e-12、独立残差门槛1e-7。bias固定，无逐pair重试或参数搜索。

## 真实结果与保护

旧lp未入求解；lp_resumed源628048于15:21:25实际开始，15:22:25结果写出，controller15:22:27 passed，unit inactive/success/MainPID0。**HiGHS kTimeLimit，217 simplex iterations，unresolved**；API返回1个warning，未静默丢弃。controller通过仅表示进程正常记录结束，不是LP可行性或质量通过。没有返回/使用未收敛系数、没有预测审计或普通model.zip，eligible_for_training=false。

| 阶段 | CPU/内存限制 | guard墙时 | 采样次数 | CPU最高℃ | GPU最高℃ | cgroup最高MiB |
|---|---|---:|---:|---:|---:|---:|
| schema14测试 | 100%/2GiB | 2.087s | 1 | 38.875 | 34 | 13.625 |
| CUDA缓存 | 250%/6GiB | 6.148s | 3 | 66.75 | 37 | 897.074 |
| LP7测试 | 100%/2GiB | 2.086s | 1 | 39.625 | 35 | 13.355 |
| 额度拒绝旧lp | 250%/6GiB | 2.119s | 1 | 40.75 | 36 | 15.520 |
| 真实单次LP | 250%/6GiB | 63.348s | 31 | 80.5 | 49 | 505.793 |

以上为采样最大值，非连续峰值。全部decision hold，CPU82/90℃、GPU75/85℃、swap0/最低可用内存1GiB/六核亲和性/聚合250%与6GiB/目标STOP/主机/GPU可用性检查保持；原120秒command600秒wall。纯数值LP CPU计算不构造CPU模型；缓存推理全部CUDA。缓存321、LP测试24、两LP设计各33项完成后SHA均相符。

## 结论与下一步

80次forward已持久化，不重跑。单次LP预算已用完，不延长时间、换solver/目标/margin或逐pair重试来追结果。当前没有“末层无解”证据，原4k/8k完整棋力失败结论不变。

下一先对这份固定缓存做一次有界数据诊断：正常竞争margin分布、近零约束行/特征尺度、训练矩阵数值秩/条件与4个pair方向的投影；validation只报告、不选参数。0模型load/forward/环境/优化器/LP求解，不应用系数或放宽棋力门槛。先查重并固定定义；如提出新学习协议须另事前封存，不能把数值结构诊断称可行性证明。

证据根runs/autonomous_head_constraints_20261107/{completion_audit.json,unit,cache,solver,lp_unit,lp,lp_resumed}；跨轮回执runs/autonomous_development_20261105/{hourly_head_constraints_execution_20261107.json,native_goal_head_constraints_resume_20261107.json}。整体21测试通过、80GPU forward、1原模型load、1真实LP、3tiny合成LP、0环境步/策略优化器/模型更新。
