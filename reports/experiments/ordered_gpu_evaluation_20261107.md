# 任意8个有序seed的可恢复GPU评测入口

## 实际结果

2026-11-07在lightbook完成新独立入口 `scripts/capture_gpu_evaluation.py`、定向测试和真实GPU进程验收。原固定seed入口及冻结旧作业未改。当前主31/推理20源码身份维持原值；这是评测时来源校验，不代表原30M历史训练来源已知。

- 26项定向检查通过（1.34秒）。覆盖任意顺序/uint32上界、非法/重复seed、CPU设备拒绝、schema、模型与源码/运行环境/native身份、旧cache换seed/模型或坏checksum在pickle前拒绝、错snapshot在模型/锁前拒绝、CUDA不可用拒绝启动。模型/runtime预检的单测使用mock，不冒充实际ZIP/CUDA验证；正式验收使用真实两ZIP、native库和完整运行环境。
- 单测另16个真实compact环境步，用first-legal合成推理验证8个无序/边界seed reset→完整恢复→封顶停槽/不autoreset；非GPU或棋力测试。真实死亡/跨子进程路由另有上一轮冻结测试，本轮不重复运行。
- 固定开发seed `[100015,100008,100014,100009,100013,100010,100012,100011]`。原30M实际控制，已有4k仅旁观。两checkpoint在CUDA加载、env=None，随后显式逐lane重置，防ZIP seed42覆盖；构造用base_seed=0安全占位，支持不连续/无序和uint32边界。
- 串行oracle进程577330：512轮/4096步；并行577424：256轮/2048步，退出后577485复用2个已提交chunk，仅计算下一256轮/2048步。首段全部SHA未变。
- 初始和两段提交的全部rows数组（action/probability/margin/obs-mask摘要/双方selected features/reward/终止/seed等）、完整worker/Core/bag/游戏及reset/action RNG、父采样RNG、active views/计数均精确一致。各worker实际episode_seed与design及trace逐一核对，不能只靠两臂同错通过。
- 正式共8192新环境步、2048 GPU forward、0optimizer；原policy/完整Adam/实际计数30003200/36625、旁观4096/36630不变。没有训练checkpoint或策略推广。

## 入口协议与限制

`design.json`显式声明evaluation_version=ordered-gpu8-v1、dispatch_version=unique-seed-parallel-v1、dispatch_mode（parallel或serial_oracle）、device=cuda、n_envs=8、8个不同uint32 seeds、max_pieces=20000、chunk_rounds=256、两冻结模型路径/SHA/实际counts、native运行库、主/推理/运行环境及入口/helper源码身份、frozen_files。

整个design SHA和实际入口/helper源码进入新缓存身份。每次检查输入后才读取经manifest链和所有文件SHA验证的本地pickle，锁内再次检查。每256轮完整原子提交，死/cap槽不reset。自然概率mode、最高精度/TF32禁用和原greedy实现不变；无CUDA直接失败，不转CPU。恢复保持全部8槽及源policy/Adam/RNG，提交后用全snapshot/实际步数约束核对。一次CLI只算1–20个新chunk：`--run-dir <新作业目录> --max-chunks N`；真实运行须由现有resource_guard及目标/STOP/主机控制器包裹，不能裸跑无保护训练。

本轮两个512轮功能作业已验收结束，**不得追加到20k当另一完整质量评测**。serial_oracle是显式CUDA环境派发对照，不是CPU推理fallback。不是新的速度门、候选4k额外8资格评测、独立未触碰确认集或棋力改善；4k/8k已失败质量结论不变。正常进程边界恢复通过不等于OS重启/任意断电已测。

## 资源保护与证据

正式guard 10:31:42/controller10:31:43 completed，38.930秒，69个冻结输入在验收后仍全部一致；GPU8、CPU250%/6GiB、swap0、原82/90℃CPU与75/85℃GPU温控、内存/时间/主机/目标限制保留。19次采样最高CPU72.75℃、GPU52℃、3506.16MiB，实际配额始终250%；decision含hold/increase，increase发生于上限，没有实际升配额或热降载。单测100%/2GiB/2.058秒，仅1次采样，不能把9.91MiB当真实内存峰值。

证据根：`runs/autonomous_ordered_gpu_eval_20261107/`，含protocol、两个design/chunk/progress/process、restart_receipt、completion_audit、unit/run controller与guard日志/采样。新 `verify_ordered_gpu_evaluation.py` 拒绝已有验收标记或已有chunk的重复验收，失败保留进度；需要人工/后续Codex先核对断点而非重复启动。

小时续办：10:12:55消息由新turn01a1b3cf-a4b3-7ba0-b8c3-31d608a3aa11实际消费，首工具10:21:41.867/call_d419daf984ab40e1acced8fd12946894，client精确且queue空。收据 `runs/autonomous_development_20261105/hourly_ordered_gpu_execution_20261107.json` 记录实际开发和上述验收，不用accepted或token显示推断执行/未来额度。驻留/每小时/防休眠配置延续已验收状态；没有扩大OS重启验证范围。
