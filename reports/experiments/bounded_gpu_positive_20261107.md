# 有限GPU阶段实际计算路径验收

## 结果

2026-11-07新sealed job通过实际CUDA8 phase入口，原30M控制/已有4k旁观，固定无序seed [100015,100008,100014,100009,100013,100010,100012,100011]、真实max20000/chunk256、phase上限256轮/2048步。

- 直接受保护CLI源进程 **583634**，申请20段，实际保守限制为**1段/256轮/2048新环境步**，两模型共**512 GPU forward/0optimizer**。旧read_chain额外边界检查2次，含原capture锁内；真实guardian/watchdog/kernel/授权/profile检查成立。
- 初始chunk0与chunk1的**全部rows数组**、完整worker/Core/bag/游戏/reset/action RNG、父采样RNG、active views与计数，与原有序GPU串行oracle冻结前256轮**精确一致**。双方保持各自design/cache identity；原source policy/完整Adam/实际计数30003200/36625、旁观4096/36630未变。新保护查询没有扰动结果。
- source process JSON与phase receipt一致，guard watchdog child_pid核验到实际计算进程，不能仅凭controller passed/accepted推断执行。
- 重复同attempt真实CLI **583886**：只返回原receipt/reused=true，源PID/原收据SHA及所有job文件/集合不变，新增计算0。复用receipt中的2048是原attempt历史累计，不能再次计作新增。
- 新attempt绑定真实256轮边界，CLI **583891**：preserved/chunks0，0环境/forward/optimizer，无新started/process/chunk。job状态仍resumable_partial、can_resume=false/阶段预算满、completed=false。完整20k评测没有完成，此功能验收job保持结束，不延长预算。

## 保护与证据

计算guard/controller11:05:12 completed，10.273秒/77冻结输入，实际CPU250%/6GiB/swap0/原温控、主机/目标/STOP/内存/180秒command600秒wall保护；5次采样最高CPU63.75℃、GPU42℃、3220.98MiB，始终250%/hold。纯只读比对与复用/拒绝guard/controller11:06:06 completed，6.153秒/94冻结输入/100%2GiB，3次采样最高CPU63℃/GPU43℃/575.88MiB，hold。均为采样值，不声称连续真实峰。所有有限service终态，原后台/驻留/每小时/防休眠设置维持。

证据根 `runs/autonomous_bounded_gpu_phase_acceptance_20261107/`：事前protocol、job新sealed design/chunk0/1/process/progress、compute request/started/phase_receipt与guard/controller，verify/acceptance_receipt与guard/controller、budget_refusal/phase_receipt、completion_audit。新 `scripts/verify_bounded_gpu_phase.py` 实施实际进程/两身份精确完整比对/幂等与阶段拒绝；本轮没有额外单测或复测旧矩阵，实际集成是目标验证。

原生轮01a1b3f5-1d9c-7ee1-a415-04bee1f10331首工具11:02:35.573+08/call_9ab89abfec8a4a32b990dd8c6bd917d0；native_goal_bounded_positive_execution_20261107.json记录实际开始和结果。前轮6bffb78为progress，本轮补齐此前未验的positive CUDA计算，旧512轮作业未追加、旧冻结runner/helper/主31/推理20/训练模板未改。

## 使用范围

`eligible_for_future_bounded_phase_launch=true`：现在有真实正向CUDA执行、精确源结果、真实预算拒绝/复用及完整保护证据。未来有授权的GPU8片段可用新sealed job和冻结attempt/controller运行，严格按真实缓存断点，不把旧attempt receipt当当前执行或额度恢复证明。主机/GPU不可用保留，无CPU训练或他机fallback；额度受限保存等每小时，暂停取消停止。

这是基础设施实际计算验收，**不是新棋力/速度门或4k资格评测**，没有PPO训练、普通checkpoint或推广；原30M保持冻结，已有4k/8k质量失败结论不改。下一回到学习数据/信用分配瓶颈，避免继续叠加状态工具或重复失败超参。
