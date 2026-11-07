# 当前GPU偏好验证集：4个预定案例均强有益

2026-11-07，lightbook；本轮progress。补充4个validation案例/3个完整episode，未进行策略训练或模型推广。

## 无人值守实际执行

上一dc0dc4a完成数值结构审计，非等待或状态复述。本轮原生turn01a1b4eb-4404-7953-b98c-997a1ff2e8b8，首工具15:31:28.454+08/call_cd50de3dfd9c4cb19d211c24d0835f58，回执native_goal_paired_validation_execution_20261107.json；非新hourly消费。

health15:29:05 healthy：四unit enabled/active、linger yes、sleep masked、proxy空、goal active、实际后台13827→1、固定lightbook machine-id6b1b552131804ebc918f30005b4d0204，无既存live compute。没有重启后台/转移机器，不扩大OS重启或未来额度保证。

新driver631238在本机独立后台串行10阶段，真实完成后unit inactive/success/MainPID0；完整5000round/21checkpoint。并非仅启动accepted或依靠人工逐阶段续开。原4train5000、512恢复oracle、80forward缓存/单次LP超时和matrix审计均未重跑。

## 事前选择与必要修复

选择继承旧Top-K seed划分：val992011/992016/992027，先在旧beneficial/ppo_top中每seed按(step,state_id,action)取最早1，再从余项按(seed,step,state_id,action)取最早1。protocol.json及capture/design.json在新GPU结果前封存，无失败替换/改变种子/延长窗口。4个train seed992000/004/006/008未混入val。旧CPU标签均已探索，4case仅3episode且992011两个case相关，不能称4个独立局或独立最终棋力确认。

旧branch ID seed_role在同seed多个case时冲突。paired_branch_capture现在只在case重复seed时加入step/action，8个独立ID支持同源state不同动作；旧4unique-seed训练选择和IDs兼容。selector显式case_split并拒绝缺seed/不足case，driver读取有限first/last/max_phase及独立unit命名、旧默认3..11兼容，新1..10/专属tetris-paired-validation。auditor记录validation用途。改动前三脚本及SHA存before_validation_changes；已完成旧job用其at-finish证据，改后路径不冒充旧执行源码。

11项定向测试1.15秒通过，包括旧选择/ID兼容、validation整个episode隔离及顺序不依赖输入排列、重复seed/同state不同动作独立ID、重复case拒绝、缺seed/第四case拒绝、阶段预算/namespace拒绝、瞬时systemctl超时只重查同handle。测试没有模型或环境计算。

source631063于15:36:15完成8真实native worker初始化probe：完整core/NumPy/action RNG/公开输入成对一致，所有first action合法，0模型load/forward/env/optimizer。旧b382仅缺reward_profile按已查原公式映射default metadata，所有RNG必需，不合成缺失状态。

## 实际GPU工作与标签

原模型SHA eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34，实际30003200/36625，CUDA最高float32/TF32false、8配对branch、greedy冻结继续。第一轮自然动作全部匹配封存baseline。每phase≤2×256round/4096新步、5000/branch/40000job，终止branch不auto-reset。10个源进程631275/631380/631496/631635/631739/631838/631953/632085/632175/632328。

实际 **26900唯一环境步、5000 GPU forward、10次原模型load、0策略优化器**。10phase完整policy/Adam/计数/加载及恢复后的全局采样RNG均未变；新8branch显式当前模型SHA。没有新增512 oracle，不能加之前32096train/oracle工作进本轮。

| state / seed / snapshot step | base→alternative | base新步/终止 | alternative新步/终止 | 共同窗口得分比 | 四消比 | 观测生存增益 |
|---|---|---|---|---:|---:|---:|
| 95 / 992011 / 1398 | 613→107 | 2000 / 真死亡 | 5000 / 右删失 | 1.015278650 | 1.015544041 | 3000 |
| 99 / 992016 / 51350 | 1169→84 | 2000 / 真死亡 | 5000 / 右删失 | 1.007930446 | 0.989690722 | 3000 |
| 106 / 992027 / 4217 | 272→134 | 2000 / 真死亡 | 5000 / 右删失 | 1.012694548 | 1.010256410 | 3000 |
| 84 / 992011 / 2498 | 1169→336 | 900 / 真死亡 | 5000 / 右删失 | 1.039921530 | 1.046511628 | 4100 |

4/4按原preference规则strong beneficial；seed992016四消比0.98969仍高于原0.9阈值，未改门槛。前三共同2000、末个共同900步；替代均5000右删失，观察增益3000或4100不是窗口外生存预测。4个标签仅validation，不回流训练。

source632577完成全窗口审计15:40:36、controller15:40:39 passed；完整21chunk manifest链/row消费/独立branch身份/完整worker状态、初始core/RNG、checkpoint公开输入/动作/死亡/截断与全部source receipt核验。全部8个旧已记录outcome步数/得分/四消/终止也相等；旧branch均缺model_sha256，这不能补历史逐branch模型/runtime或完整动作/tensor证明。当前labels不证明其他173旧偏好或新策略棋力，eligible_for_training=false。

## 保护与证据

13guard（unit/probe/10phase/window）全部completed，全部采样hold。真实GPU阶段250%CPU/6GiB/swap0、六核亲和性、聚合slice250%6GiB、CPU82/90℃GPU75/85℃、最低可用1GiB/GPU telemetry/120command600wall/host与goal/STOP保护保持；unit和纯数据window100%2GiB。完整窗口入口增加strict授权前后检查，0新模型/forward/env/optimizer。

采样最高CPU71℃、GPU51℃、单guard cgroup3202.363MiB；非连续真实峰值。91job输入、96driver输入、各phase递增冻结提交及171window输入在完成后SHA均相符。原模型ZIP与31训练/20推理源不变。completion_audit逐guard保留墙时/PID/采样范围，窗口branches SHA a8bef3ecceb842871fd2a6225c0b10df57dbb2be9c691d9e8d067f1e252a8d4a，window_result SHA9ffc8375966cc34891ed837ef7d4b8a9bdd45ded5f19afff604b28777189674a。

证据根runs/autonomous_paired_validation_20261107/{protocol.json,capture,probe,pipeline,phase_01..10,window,completion_audit.json}，跨轮回执native_goal_paired_validation_execution_20261107.json。

## 下一步：单次有验证集的末层GPU偏好试验

当前有4train/4validation currentGPU强有益标签，验证按3episode聚合；源policy模块确认合法动作raw score直接进masked categorical，可据缓存rawscore定义概率目标。先做新的限定准入审计和事前单次协议，明确旧eligible_for_training=false不授权自动PPO或推广；不能直接将旧177标签加入训练。

只补新validation初态公开h（原CUDA模型一次batch8前向/完整状态不变），复用原640缓存不重提取。新试验仅64维末层权重位移d，bias及其余网络固定，GPU fresh Adam、lr0.01/betas(.9,.999)/eps1e-8、固定256次更新，不选择epoch或重扫参数。每次batch8=4train pair初态＋4train正常state（seed100000..3分别取chunk=t mod79），loss=4pair平均softplus(-(原good-base rawscore差＋d·h差))＋4normal平均KL(softmax缓存rawscore || softmax(缓存rawscore+h·d))＋0.01||d||²，T1/β1固定。缓存目标是定义的logit教师，有限精度与实际原CUDA概率需披露，不声称逐位相同。d初始0、GPU float64代数，最终映射float32末层；256是head optimizer步，不伪造PPO环境步/更新计数。

validation pair/normal全部只在固定终点报告，pair按每seed内mean再3seed等权，不用于loss、调参、early stopping或重选候选。保留rawscore重建误差≤1e-4与模型/数据SHA/计数/RNG检查。事前声明候选实际全网络CUDA复核预算：原validation1forward＋更新后4train/4val共2forward＋632normal79forward=至多82次full-policy forward，另256个batch8 head-training forward，不混算。原模型ZIP/Adam/30003200与36625不得改变，完整候选/新head optimizer/元数据单列。

先定向测试loss/梯度、episode隔离、fresh optimizer、参数作用范围/计数与有限精度，再真实有限GPU试验；所有计算保持原250%6GiB/温控/120command600wall/STOP/额度保护。前述准入审计或保护不通过则保留，禁止CPU模型fallback/换主机。只评固定终点，不重试已封闭LP或旧方向。

若最终实际GPU在train与val各至少1个good为全局Top1，且val按episode等权pair loss较原下降，方准备后续独立有界完整棋力评测；其余结果如实失败并停止本试验，不扫参数补救。所有normal动作变化/KL都完整报告，动作变化不等于棋力下降，不另立正常Top1全相同的质量门槛。普通PPO恢复资格不自动授予；候选仅供评测，原最终≥.99pieces/.99score、死亡不增、所有seed≥.9等质量门槛保持。原始计数、额外head256步和后续评测环境步分开记录，GPU/host/额度不足保存等原每小时机制。
