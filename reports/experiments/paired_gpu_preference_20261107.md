# 原30M四对偏好的当前GPU复核

2026-11-07，lightbook。本轮分类 **progress**：完成真实8进程配对恢复、跨进程恢复/复用验收、5000窗口分段执行和完整记录审计。没有策略训练或模型推广。

## 实际自动执行

原生续轮 `01a1b41c-7ae8-7a12-b391-1bc0067e41be`，首工具11:45:37.535+08 / `call_e1f142527cee45e58480ca11e1e23df2`；前轮ec5f162是progress，不重跑其几何诊断。本轮不是新hourly消息消费。证据 `runs/autonomous_development_20261105/native_goal_paired_branch_execution_20261107.json`。

接手无live guard，后台/goal正常。有限串行驱动器进程594893于轻薄本后台自动执行phase03..11；12:06:15终止、systemd inactive/success，9阶段自行完成。各阶段均核对本机machine-id、control/STOP、goal严格active、CUDA、实际kernel资源限额、独立温控guardian与heartbeat watchdog。没有桌面通信暂停或OS重启新验收，也不声称未来额度始终可用。

## 固定输入和协议

只用封存原30M `runs/ppo_candidate_raw_30m_seed42/final/model.zip`，SHA256 `eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34`，真实30003200步/36625次PPO更新。CUDA、batch为原顺序活跃分支、最高float32、TF32关闭；单个分支停止后不auto-reset。

从已审计的177个旧Top-K强有益偏好中，事前固定train seed992000/992004/992006/992008，各按(step,state_id,action)升序取首对。不能根据本轮GPU结果替换。此选择仍依赖已知旧正例，不是无偏采样；四seed也不是独立质量确认集。

| seed | 源step / state_id | base动作 | alternative动作 |
|---|---:|---:|---:|
| 992000 | 24434 / 90 | 249 | 1047 |
| 992004 | 79538 / 91 | 110 | 40 |
| 992006 | 40709 / 92 | 616 | 63 |
| 992008 | 84013 / 93 | 1555 | 152 |

四对分别复制相同完整Core、私有bag/游戏RNG、NumPy/action RNG、episode/计数/奖励状态，共8个真实工作进程。每对只首动作不同，之后共同使用冻结原PPO的实际float32概率mode。模型输入仅公开observation和legal mask；私有状态仅用于离线恢复。

重复episode seed由独立branch_id区分；没有修改原ordered GPU8接口要求unique seed的规则。固定每分支最多5000新步、全job≤40000唯一环境步；每phase最多2×256轮/4096新步，120秒command/600秒wall。driver最多9个后续阶段，1800秒总runtime。所有阶段逐次核对显式起点manifest SHA/计数，外部hash在pickle前检查，完整worker/输入摘要随每个chunk持久化。

## 首次探针失败与兼容修复

- v1 unit16passed/1.15秒；第一次真实恢复探针进程592786被完整字段断言拒绝，guard11:53:50 command_failed/8.267秒。**0模型加载、0forward、0环境、0优化器**，无chunk提交，进程已释放；原capture/oracle空job写STOP封闭。
- 四个旧快照均包含完整core/NumPy/action RNG。唯一缺失字段为后来新增的`reward_profile`。查证suite commit b382a44b旧get_state确实未写该字段，原placement-reward-v1和当前default奖励公式相同；旧源封存 `r2/legacy_tetris_env_b382.py`。
- 仅当只缺这一字段、原reward_version和零shaping都匹配时，显式增加default metadata。缺任何RNG/轨迹字段、未知reward均拒绝；没有合成随机状态。新r2重新冻结67→68输入和独立目录，没有覆盖旧失败。
- r2 unit **19passed/1.16秒**；实际8进程恢复探针进程593284，11:56:09 receipt probe_passed / 11:56:12 controller passed，0模型/forward/环境/优化器。成对完整状态及公开输入相等。

## 真实恢复和去重验收

生产进程593628推进256轮/2048步后，另进程593894恢复至512轮/4096步。独立oracle进程594155一次推进512轮/4096步。12:01:44只读verifier确认0/256/512三个checkpoint的**全部worker/Core/NumPy/action RNG/公开view摘要和全部逐步数组精确相等**。oracle已STOP封闭，不继续5000窗口。

12:02:14重复已完成phase02 CLI，进程594621明确reused，原进程593894/receipt SHA及整个job文件集合/字节均不变；新增模型加载/forward/环境/优化器均0。没有以投递或controller success代替实际计算证据。

随后driver串行完成phase03..11：生产全job **28000环境步/5000次GPU前向**，21个checkpoint，四base都2000步死亡、四alternative都5000步右删失。每阶段完整policy、Adam、真实计数和模型加载/worker安装后的全局采样RNG前后不变。

正式独立计算总量为生产28000＋oracle4096＝**32096环境步，5512次GPU前向，12次原模型加载，0策略优化器步**。恢复验收的8192步是其中两个前缀的合计，不能再次相加。另v1/r2单测各2个compact环境逻辑步，共4步，无CPU策略训练；监控单测无真实环境。

## 固定规则结果

完整窗口审计12:09:21 passed，controller12:09:23 passed；外部封存161输入，消费全部trace并核对连续活槽/branch_id/seed/首动作/后续natural action、完整snapshot计数/终点、模型SHA/计数、phase实际工作量及checkpoint公开输入。

规则复用原compare：base真死、至少多250步且≥1.5倍、共同窗口得分≥.95、base四消≥3时四消比≥.9；保留右删失语义。

| seed | base步数/终点 | alternative步数/终点 | 共同2000步得分比 | 四消比 | 标签 |
|---|---|---|---:|---:|---|
| 992000 | 2000 / 真死 | 5000 / 右删失 | 1.0093172503 | 1.0051020408 | 强有益 |
| 992004 | 2000 / 真死 | 5000 / 右删失 | 1.0116454233 | 1.0211640212 | 强有益 |
| 992006 | 2000 / 真死 | 5000 / 右删失 | 1.0156715238 | 1.0106382979 | 强有益 |
| 992008 | 2000 / 真死 | 5000 / 右删失 | 1.0115582734 | 1.0157068063 | 强有益 |

四对均强有益，观测生存增加3000步；不能推出5000之后的生存。8条新分支都有显式原模型SHA及当前运行来源；其步数/终止、逐步得分与累计四消记录也全部匹配对应旧CPU分支。旧记录不含完整动作序列/全部observation tensor和逐branch模型SHA，故不能声称补出历史模型/runtime身份或逐位全轨迹相同。

只补齐这4对的当前GPU证据，**不把其余173对一并认定为已复核**。没有新策略，不能据此声称整体棋力提高，`eligible_for_training=false`及原4k/8k质量失败结论保持。

## 资源、故障与代码

证据根 `runs/autonomous_paired_branch_20261107/`，当前有效实验在`r2/`：probe、phase01..11、oracle_phase、restart、reuse、pipeline、window、driver_test及completion_audit/resource_summary。

- 所有真实GPU阶段CPU250%/6GiB/swap0/六物理核affinity，共用tetris-compute.slice限额；原CPU82/90℃、GPU75/85℃、内存/心跳/STOP保护保持。18个r2 guard均completed、采样决策全hold；采样最高CPU **70.125℃**、GPU **50℃**、单guard cgroup **3201.742MiB**，仅为采样最高而非连续真实峰值。
- 数据/单测/只读验收100%/2GiB，仍有GPU遥测和原温控；单采样不作内存峰值证据。
- window准备脚本一次str/Path异常发生在design生成前；初次controller因缺design终止，未启动guard或任何分支。核实后修正准备，以新controller验收，没有重跑GPU计算。
- 已完成driver原源在修改前冻结校验通过，并归档`pipeline/driver_executed.py`及`at_finish_frozen_audit.json`。随后增加瞬时systemctl查询失败重查同一unit的处理；1项集成测试12:09:49 passed/1.19秒，确认不误停、不重复启动，**不是本次真实pipeline曾经历该故障**。当前共有20项相关测试通过，v1旧16项不重复计数。
- 新增paired_branch_capture/paired_branch_guard、prepare/run/continue阶段工具、restart/window审计与测试；原31训练源/20推理源未改变。r2核心68冻结输入前后不变，driver后续修订与执行归档明确区分。

下一先封存这4对当前GPU标签及公开表示，再检验是否存在同时保持正常动作的末层调整；不沿上轮失败的均值方向直接训练。原模型和质量门槛保持。
