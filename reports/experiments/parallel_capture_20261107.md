# 完整GPU8评测的并发worker派发

记录：2026-11-07T10:18:43+08:00。**固定ABBA评测与原子提交阶段几何提速1.877311倍，正常进程恢复通过，已提供后续GPU评测入口。** 不声称PPO训练加速或棋力提高，原模型不变。

## 实现与采用范围

旧own_policy_capture.advance_round按lane同步等待capture_step；新ParallelNativeEnv新增seed→action路由，通过一次SubprocVecEnv.env_method先发送全部active worker，再按原lane顺序接收。两模型GPU forward、active batch形状、最高float32/TF32false、自然概率mode和row字段不变，死亡/封顶槽停住、无autoreset。要求8个不同episode_seed，**不能用于同seed的8个历史context**。

旧冻结helper、capture runner、原job/design/缓存及主31/推理20源码未改。新capture_parallel_trajectories.py显式要求dispatch_version=unique-seed-parallel-v1；复用原capture的事务/完整状态/RNG恢复实现，只在独立进程临时绑定并发worker和step，finally恢复。dispatch字段随整个新design SHA进入缓存身份，拒绝旧serial job静默迁移、换模型复用旧cache、坏快照及错seed；新输入先验验证在锁/数值推进前完成，所有pickle先查链和文件hash。

## 一次预声明ABBA

来自已完成原GPU基线chunk0000完整8worker，开发seed100000..100007/max20000，一局无reset，原30M实际控环境/已有4k只旁观。固定A/B/B/A、每arm512轮；每次恢复同全部worker/active views/父RNG。每arm仅GPU预热2forward、0环境步，然后同步CUDA计时。包含两GPU模型、全部环境推进/返回、摘要/trace组装、完整状态导出、256轮原子提交/压缩/fsync和模型/RNG审计；不含启动/恢复/初始提交/预热/事后source比较hash。

| arm | 派发 | 计时秒 | 实际新环境步 |
|---|---|---:|---:|
| 1 | serial | 9.302913 | 4096 |
| 2 | parallel | 5.160840 | 4096 |
| 3 | parallel | 4.874013 | 4096 |
| 4 | serial | 9.529291 | 4096 |

配对A1/B1=1.802596677、A2/B2=1.955122448，几何1.877311170，超过先验≥1.05且两pair方向一致的门。不是无误差的普适速度保证，也未重复计时选最好结果。四臂逐数组自然action/probability/margin/obs-mask digest/selected feature/outcome与原冻结前512轮全部精确；每段完整worker/Core/bag/游戏及reset/action RNG、父CPU/CUDA RNG、active views和计数都与原snapshot相同，模型policy/完整Adam/计数未变。八真实worker PID [445132, 445133, 445134, 445135, 445136, 445137, 445138, 445139]。

正式ABBA：16384新环境步，原模型与旁观各2048 GPU批次，另8次预热，共4104 GPU forward；0optimizer。不是训练步增加。

## 入口与恢复验收

routing10项测试passed/5.71秒：错误seed/映射推进前拒绝、真实死亡、真实cap停槽、两真实worker跨新进程恢复。entry5项passed/1.18秒：旧serial设计不能迁移、checksum一致但变更模型身份不能复用旧cache、坏snapshot在pickle前拒绝、缓存worker seed拒绝、异常后模块绑定恢复。两组测试合10个小型真实环境步、0optimizer；checksum身份fixture不是有效PPO ZIP语义测试，实际PPO加载由下面真实CUDA验收覆盖。

verify_parallel_capture_restart.py驱动**574285→574341**，两进程各256轮/2048新步；第二个直接恢复初始+首段两个提交目录，不重采前缀。初始和首段全部文件SHA未改，三个新chunk的trace和完整状态与已校准B1逐项精确，源policy/Adam、计数原30003200/36625和旁观4096/36630、采样RNG未动。源模型ZIP SHA eb859b…/9a11ce…保留历史训练来源，当前31/20只标评测运行源码。

恢复验收共4096新环境步、两模型各512 GPU批次/共1024，0optimizer；正常退出→新进程恢复已测，OS重启/任意断电范围未扩大。这个512轮恢复根已完成，**不要把它追加成另一完整质量评测**。本开发正式合20480新步/5128 GPU forward，另unit10环境步。

## GPU/CPU保护

全程lightbook，GPU8/原CPU250%/6GiB/swap0/温控与GPU遥测；每个正式command180秒/wall600秒。单测100%/2GiB。GPU或主机不可用拒绝，无CPU训练/他机fallback。

| 阶段 | guard完成 | 秒 | 配额 | 采样CPU℃ | 采样GPU℃ | 采样内存MiB |
|---|---|---:|---:|---:|---:|---:|
| routing_unit | 2026-11-06T21:40:18+08:00 | 8.191 | 100% | 52.875 | 36 | 737.75 |
| abba | 2026-11-06T21:41:46+08:00 | 36.766 | 250% | 75.375 | 52 | 3233.46 |
| entry_unit | 2026-11-06T21:46:40+08:00 | 2.084 | 100% | 40.000 | 34 | 13.42 |
| recovery | 2026-11-07T10:14:31+08:00 | 20.601 | 250% | 66.500 | 45 | 3501.40 |

全部冻结输入完成后未改，全部hold无降载；正式阶段峰值采样CPU75.375℃、GPU52℃、内存3501.40MiB。2秒采样不是连续真实峰值；entry_unit仅1样本，13.42MiB不能当真实进程峰值。

## 无人值守与额度间断

原生goal轮01a1b111-6105-7352-96bc-ef4df50f2474首工具11月6日21:34:41.228，已完成benchmark及entry unit，尚未启动恢复时额度受限。11月7日10:11:45仍usage_limited；10:12:55计时器记录ordinary_usage_allowed=true并恢复原目标，未消耗额度重置券或改账户。新原生轮01a1b3c7-c7fa-7223-8240-4b935624c203首实际工具10:13:12.226，检查之前未启动、文件未变后才算剩余4096步。**不把等待额度的夜间说成持续开发，也不声称额度未来保证可用。**

10:14:27本机healthy/goalactive/4驻留和小时单元enabled active、linger yes/睡眠masked/proxy空/后台13827→1。10:12:55新hourly submission01a1b3c7-c846-7e13-80e9-70ffe9780b5f（client01a1b3c7-c845-7ba1-a28d-2b765eedf430）仍pending/thread_busy，本轮是原生目标恢复，**未声称该投递已消费**。源与恢复新旧证据均在native_goal_parallel_capture_execution_20261106.json，后台自动投递与实际工具严格区分。

## 后续使用

新模型/新评测必须新建job目录，参考原完整GPU基线design的31/20/runtime/native/frozen_files结构，增加显式dispatch_version，固定实际模型path/SHA/counts、源码和资源profile。frozen_files须包含新入口与parallel_capture、旧capture实现和依赖、模型及native产物；不要复制恢复验收专属expected_chunks/expected_identity或512轮预算来跑完整质量任务。

通过现有verify_context_guards_when_idle受保护controller，将validation_command设为新capture_parallel_trajectories.py --run-dir NEW_JOB --max-chunks 20；保留goal/control/机器检查及250%6GiB/原温控/时间预算。每阶段最多5120轮，恢复时原design/模型/缓存身份不变，使用新guard输出目录。当前明确支持固定100000..100007/20000协议，额外不同8seed需要单独显式通用入口，不能篡改旧来源绕过硬编码。

4k7cap/1死9797、8k6cap/2死及KL离线门失败均保持，不为性能验收重训/扩2M。持续目标active，暂停取消即停，额度不足保存等待每小时重试。证据：runs/autonomous_parallel_capture_20261106的benchmark/arms/completion_audit，以及autonomous_parallel_capture_recovery_20261106的restart_receipt、process、chunk与guard/controller。
