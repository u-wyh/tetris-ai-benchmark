# GPU评测只读状态判定

## 已完成

新增 `scripts/inspect_gpu_evaluation.py`，命令行 `--run-dir <GPU8作业目录>`，完整终态检查另需 `--upstream-summary <保护控制器收据>`。JSON状态 fresh、resumable_partial、terminal_verified、invalid、blocked；退出码0为有效状态，2为invalid，3为blocked。必须同时看status/completed/can_resume及阶段预算，不能把退出0或控制器passed当评测已完成。

复用未改的capture_gpu_evaluation.prepare（schema、模型/源码/runtime/native/frozen输入身份、manifest链/所有文件SHA先于pickle）和validate_snapshot；检查初始reset/seed、末snapshot的全部8槽、实际轮/unique环境步以及每段冻结policy/Adam/计数/RNG审计。完整终态还必须有准确summary/全部8个endpoint、保护收据passed，且原audit_own_policy_capture的完整trace、独立worker endpoint、源计数、父/worker采样RNG与process实际工作量审计全部通过。没有summary而全部槽已停止时仍是partial，只建议补终态摘要、0新增环境步。坏摘要不能触发重算。

为保留冻结入口和旧作业，输入检查在临时symlink视图完成，先把design/已提交chunk交给prepare；summary仅在完整审计阶段可见，compute entry从未调用，也未猴补任何协议校验。原auditor只写临时输出，不在原job创建trajectory_audit。共享已有capture.lock（只读、不创建），写进程持锁时返回blocked；检查前后完整输入文件集合/SHA及lock inode，变化则撤回可恢复/完成判断。pending路径单独列出，不当作已提交工作。

阶段上限来自design phase_round_limit/phase_environment_step_limit（缺省20000/160000）。达到有限阶段预算的活槽partial缓存can_resume=false，建议保留结果、不自动延长。此工具只报告缓存/预算状态，没有调度或CUDA可用性承诺，也没有直接替代计算入口的预算约束；裸调用原runner仍须由调用方遵守phase预算，下一步补受保护的有限phase启动器。

## 验证与范围

- 21项定向测试/1.73秒：fresh、controller passed仍partial、有限阶段不自动延期、缺summary补摘要、缺保护收据blocked、伪造/错身份/不足8endpoint/错误终态counter、模型审计与坏cache/换seed、坏hash在pickle前拒绝、真实frozen终态auditor拒绝重hash后feature/outcome错配和缺process工作量；持锁写者和检查中输入变化均blocked。
- 正面terminal_verified测试是合成一致fixture，通过真实旧auditor；preflight模型/runtime/native采用mock。它证明完成判定协议，不冒充真实GPU训练或新整局终态。
- 新 `verify_gpu_evaluation_status.py` 在真实两个有序GPU评测缓存调用独立CLI：579361/oracle、579366/parallel，均3提交/512轮/4096 unique已发生步/8活槽，保护收据passed但输出resumable_partial/completed=false/phase_budget_reached=true/can_resume=false，全部原文件SHA/集合未变。没有兼容的新通用协议完整终态GPU作业，本轮不为状态工具另开完整评测或迁移旧身份。
- 本轮总0新环境步、0GPU forward、0optimizer，不加载策略ZIP，不构造worker，不训练/推广；原30M与4k计数及旧结果维持原值。旧fixed/generic runner/helper/主31推理20未改。

单测guard10:40:11/controller10:40:12 completed，4.112秒/64冻结输入；真实CLI guard/controller10:41:39 completed，4.119秒/93冻结输入。均CPU100%/2GiB/swap0、原温控/GPU遥测/内存/主机/目标/STOP/时间检查；实际GPU8训练评测仍上限250%/6GiB。CLI仅2次采样最高CPU57.5℃/GPU37℃/573.63MiB，采样值不是连续真实峰；unit2次49.375℃/35℃/298.68MiB，均hold。

证据 `runs/autonomous_gpu_status_20261107/`：unit/run设计、guard日志与采样、inspection_receipt、completion_audit。原生持续轮01a1b3dc-05a3-7d70-9052-ed82be097124首工具10:35:11.490+08/call_95ad890243fc4e31ab1b8090aeb4bc2a；不是新hourly消息。`runs/autonomous_development_20261105/native_goal_gpu_status_execution_20261107.json`记录实际开始/结果。
