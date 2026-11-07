# 有限GPU评测阶段启动器与真实保护验收

## 功能

新 `scripts/run_gpu_evaluation_phase.py --attempt-dir <原job之外的尝试目录>` 读取冻结request.json，显式version=bounded-gpu8-phase-v1/mode=capture或guard_probe、job目录/design SHA、expected_start（last_manifest_sha256/committed_chunks/round/actual_unique_environment_steps）、max_chunks1–20、controller_design和guard_dir。计算请求对应的新job必须声明bounded_phase_version并在frozen_files封存phase/guard/inspector源码；旧cache不能补字段迁移。新helper `gpu_phase_guard.py` 和 `verify_gpu_phase_refusal.py` 配套。

- 状态检查invalid/blocked、已达有限预算、已审计终态直接拒绝或保留。每256轮按当前活槽数计算保守工作上界（死亡只会减少，不autoreset）；phase剩余轮/步不足一整个chunk时不向上凑，原20000全局末段仅32轮正确处理。max_chunks被夹到实际允许数量。全停缺summary仅允许0环境finalize，必须已有真实process工作量完整，不能补造。
- 只在外层检查存在TOCTOU并发窗口。因此临时包装原模块read_chain/check_inputs：先执行原有全部校验、原样返回snapshot/identity，再增加expected边界/保守预算拒绝；原runner在prepare和持有原capture.lock后均调用read_chain。改变边界即在模型/环境启动前失败。不是绕过协议或更改数值计算，原文件未改；finally总还原绑定，独立CLI无并发线程共享绑定。额外check_inputs还检查原始源输入、主机/目标/STOP/当前保护状态。
- attempt锁和request SHA保证重复调用复用原receipt，未完成started尝试不盲重跑，须核对已提交断点后创建新的冻结attempt。guard_probe只核验保护，不进入capture或加载策略。receipt/退出0都不代表整个评测或棋力已验收；保留initial/final检查状态，完整终态仍需保护收据passed后的全trace审计。
- 失败计算量：启动capture前为0；可能开始后直到核实原runner的process receipt前为null/unknown，防止中断后的未提交工作被伪标0。optimizer始终0。中断或额度恢复时新attempt按真实缓存边界创建，不把旧blocked/reused收据当当前可运行证据。

## 实际保护证据

不是只信配置/环境变量。执行前要求lightbook固定machine-id、用户control启用且无STOP、真实本机目标严格active（usageLimited/暂停取消均拒绝）。保护控制器冻结request/job/profile/相关源码且command精确对应当前launcher，profile与原CPU250%/6GiB/82–90℃CPU和75–85℃GPU配置一致。

核验当前cgroup属于匹配live service，kernel cpu.max≤2.5核、memory.max≤6GiB、swap0、六个实际CPU affinity；KillMode=control-group/OOMPolicy=stop/实际RuntimeMax有限且≤180秒。验证当前进程是guarded_process MainPID后代、guardian实际resource_guard命令/heartbeat路径与fresh心跳匹配，CUDA可用；缺失/不符均不转CPU或其他主机。暂停由现有控制器持续检查，额外门在段边界也核验授权/保护；底层guardian继续每2秒温控，watchdog在guardian退出或心跳过期后结束子进程。

当前r2真实探针进程**582268**，guardian**582238**、watchdog**582264**，真实cgroup位于tetris-compute.slice，cpu=2.5、memory=6442450944、swap0、affinity[4,6,8,10,12,14]、120秒，CUDA可用。这是live保护正向验收，不冒充真实8环境已运行。

## 测试、实际拒绝及范围

- r1 45项/1.30秒、一次真实guard探针与两作业拒绝均通过。随后修复未知计算量误标0风险，保留r1目录而不伪称当前源码未改。
- **r2 46项/1.32秒通过**：预算边界/不足段/死槽上界/最后32轮、非法max_chunks、满预算不进guard/compute、幂等/中断与陈旧断点、源协议不绕过、真实文件锁内并发推进拒绝与绑定恢复、缺process计数不补造、活writer、guard失效/no CUDA/错误内核/heartbeat/profile/source、暂停/额度/STOP/host拒绝、未知计算量null。模型和内核证据单测是synthetic/mocked；不冒充真实GPU正向capture。
- r2真实两个已有512轮job各两次独立CLI：oracle **582540→582545**、parallel **582565→582570**，第一调用preserved/chunks0，第二仅reused_receipt，均无started/新process/chunk，全部原输入SHA/集合未变。request想申请20chunk也不能扩展已经满预算的旧功能验收。
- **本轮所有r1/r2合计0新环境步、0GPU forward、0optimizer**。不加载策略，不构造环境，不续写两个旧job；原policy/Adam/计数、主31/推理20和冻结capture/helper/模板均不改，既有4k/8k质量失败保持。

r2 unit guard/controller10:56:50 completed，2.099秒/70冻结输入/100%2GiB；probe guard10:57:23/controller24 completed，2.089秒/88输入/250%6GiB；refusal guard10:59:06/controller07 completed，8.178秒/107输入/100%2GiB。原温控/内存/时间/主机目标保护保持。unit/probe各只有1采样，不能把15.53/13.63MiB当真实峰；refusal4采样最高CPU65℃GPU42℃573.86MiB，仅采样值。均hold。所有有限service终态。

证据根 `runs/autonomous_gpu_phase_20261107/`：当前unit_r2/guard_probe_r2/refusal_r2、completion_audit；r1 unit/guard_probe/refusal保留历史。原生轮01a1b3e4-d974-7e73-8a13-282ce743ca04首工具10:44:48.822+08/call_76753b9cbfa9497088a42893c94996e1，记录于native_goal_gpu_phase_execution_20261107.json。

## 尚需的正向CUDA集成

真实live guard和真实拒绝已通过；本轮没有通过新增钩子的正向CUDA capture，不能扩大验收范围。下一仅做一个新封存job的有限256轮/≤2048步正向集成，对照已冻结oracle前256轮全部trace/state/RNG/views；原有512轮job不延长。验证max_chunks20真实被夹为1、实际source counts/Adam冻结、重复attempt不重算及新attempt满预算拒绝。此有限基础设施验收不算新的质量资格/速度门，不启动PPO或推广失败候选。
