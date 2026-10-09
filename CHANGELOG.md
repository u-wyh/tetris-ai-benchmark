# 修改日志

## 2026-10-09

- 实现 Candidate-Scoring MaskablePPO：公开 Dict Observation、精确 uint8 的 16 维落点特征、共享候选评分网络、合法动作 gather/scatter、事务式 8-worker CUDA 训练与自动配对评测流水线。
- 完成五个 2M PPO checkpoint 的统一 32-seed 棋盘诊断与逐 seed 配对 bootstrap 区间；V3 减少中央最高列，但未稳定超过 Raw 存活。
- 写出 Candidate-Scoring MaskablePPO 架构设计与首轮独立 2M 对照计划；本次未启动新训练。

## 2026-10-08

- 夜间 PPO 队列补充可恢复的 `extended_training` 选择阶段：统一比较 Raw/V1/V2/V3/V4，仅在 V3/V4 达到保守存活门槛且无明显 Game Over 风险时创建独立 10M run。
- 新增独立 PPO-Shaped V2（新增洞惩罚系数 0.02）2M 配置；保留 V1 全部训练参数，并在 Step-0 同时校验与 Raw、V1 的初始权重一致。
- 完成 PPO-Raw/Shaped 2M 的 32-seed 逐落点死亡诊断；Shaped 少造洞但在中央形成尖塔，终局总高度 83.22、最高列 19.31，31/32 局的最高列在中央两列。新增带存活样本数的高度/洞曲线、死亡前 20 步曲线与报告；未训练新模型。
- 第四阶段网页集成：增加 Human、V1 Adapter、V2 Hold-only（默认）和 V2 Beam-8 模式。策略只接收公开 Observation，由独立 Web Worker 使用合法 Placement BFS 搜索；主线程按标准路径逐步执行 Hold、移动、旋转、无人工加分的 Down 和 Hard Drop。保留原始网页 V1 函数和 Human 规则。
- 页面新增固定 Seed、演示速度、开始/暂停/重新开始、方块放置数量、存活时间及第 6 版可恢复存档。固定种子的网页策略与 Python V1/V2 在动作和正式 Core 状态上逐步一致；AI 搜索不会读取隐藏队列、Bag 或 RNG。
- 验证：Python 78 tests、35 subtests passed；JS 61 tests passed；JS/Python Parity 6 tests passed（含三模式固定 seed 动作序列、Board/Score/Lines/Level/Hold/Next/Game Over、Worker 几何与主游戏一致）。Tailscale 8080 服务及所需脚本均通过 HTTP 200 检查。
- 本次提交信息：`feat(web): integrate parity-tested traditional AI modes`。
- 新增独立传统 V2（`v2-1`）：公开观察边界内的增强启发式、合法 Hold、两层 Beam Search（宽度 8、前瞻系数 0.58）和顶部风险项；提供纯局面模拟、确定性 tie-break、组件耗时统计、可恢复 benchmark、配对比较与 bootstrap/sign-test 分析。
- V2 开发集消融（310100–310103）中，5000 块上限下无 Hold 平均 4734.25 块、Hold-only 与 Beam-8 均 4/4 截断；无 Hold 决策 4.294 ms、Hold-only 5.662 ms、Beam-8 53.825 ms。独立 10000 块开发组中无 Hold 两局在 6173/7421 块死亡，Hold-only 与 Beam-8 两局均达到上限；这些结果用于冻结配置，不能替代正式 validation。
- V2 validation 配对评测使用与 V1 完全相同的 16 seeds 和 5000 块上限：V2 平均 5000 块、1998.9375 行、20,878,968.75 分，16/16 截断；正式分片平均决策 64.396 ms、P95 84.689 ms、15.529 次/秒。V1 为平均 4044.6875 块、10/16 截断；V2 在 6 个 V1 Game Over seed 上均存活到上限，但 10 对双方都截断，寿命差受右截断影响，不能直接宣称无偏最终寿命提升。
- 同局面 CPU profile：V1 总决策约 66.16 ms，V2 约 65.46 ms；V2 成本主要为未来合法 BFS 34.17 ms 和模拟/特征 25.12 ms。V2 报告、逐 seed 原始 JSON、配置和合并清单归档在 `reports/traditional_v2/`。
- 新增可配置 PPO-Shaped V1 Reward：每个净新增洞扣 0.10，Raw 默认系数 0 保持兼容；事务元数据、恢复校验、Task 洞指标及独立 2M 配对评测入口同步接入，不改变游戏规则和 Observation。
- PPO-Shaped V1 已完成 2,002,944 步；32 个相同 validation seeds 上相对 Raw 2M 每 100 块新增洞从 151.24 降至 36.94，但平均存活从 57.94 降至 43.06。配对结果见 `reports/experiments/ppo_hole_v1_2m_comparison.md`，未启动 10M。
- 新增独立传统 V1 合法落点适配器与可恢复的 CPU Benchmark。保留网页 V1 原代码、七项启发式权重、一块 Next 前瞻系数和微小随机扰动；适配器使用 Core 的 BFS 落点及 Action Mask，禁用 Hold，固定 seed 复现选择。
- 每个 seed 完成后原子保存动作序列、逐决策耗时和正式游戏结果；记录代码 SHA-256、Git 基线、配置、异常与进度，生成聚合 JSON 和 Markdown 报告。中断后只重跑未完成对局，输出目录禁止覆盖。
- 3-seed × 200 块 Smoke Test 完成，独立重跑动作、分数、消行及截断状态完全一致；修订版 Smoke Test 保持相同动作与结果。新增 V1 公式、合法性、公开信息边界、Top Out、截断、确定性及异常恢复测试。
- `legacy_heuristic` 开发集 Pilot：5 seeds × 1,000 块均触及上限；5 seeds × 10,000 块平均执行 6,639.2 块、消除 2,649.2 行，2 局触及上限，平均 19.76 次决策/秒。未启动正式 100-seed 评测。
- 验证：Python 测试 48 passed；JS 测试 58 passed；JS/Python Parity Test 4 passed。
- 16 个 validation seeds（100000–100015）、每局最多 5000 块实测：平均 4044.6875 块、1612.9375 行、16,216,331.25 分；6 局 Game Over，10 局截断（62.5%，寿命指标饱和）；平均决策 50.616 ms、P95 80.256 ms、19.756 次决策/秒。并行 CPU 负载影响墙钟计时，详见归档报告。新增合并校验测试后 V1 专项测试 8 passed。
- 逐 seed 原始动作与决策耗时、配置、聚合、合并来源、异常记录和报告归档在 `reports/traditional_v1/validation16_2026-10-08/`；原网页 V1 算法和网页界面未改动。

## 2026-10-07

- 正式长训在 Task 1 前保存并验证同一随机初始化的 Step-0 PPO 模型；baseline 评测独立于训练状态，可从 Step-0 中断恢复，0 步结果进入 validation 曲线与 best 比较。
- Validation 记录请求阈值和实际 committed 步数；正式 run metadata 补充 run 名称与 NVIDIA driver，状态显示最近 milestone、validation、best 和训练速度。
- 长期事务训练仅保留最近 3 个普通 checkpoint；每跨 1M 步独立保存 milestone，并准备完整 Task 边界的 10M 正式配置。
- 增加固定种子 periodic/milestone validation、预留独立 final test 种子集；best 统一按 periodic 协议的平均存活方块、消行、分数原子切换，避免不同上限评测混用。
- 评测在独立进程运行，结果原子发布；中断后从已提交 Task 补做评测，再安全执行 retention。
- 事务式训练接入 CUDA + 8 worker `SubprocVecEnv`；每 Task 保存并校验 8 份环境状态、主进程 CUDA RNG、模型/optimizer，恢复时逐 worker 核对 observation/mask digest。
- 故障注入验证未提交 Task 回滚、损坏 checkpoint 拒绝加载及中断/不中断训练的参数、optimizer、RNG、环境状态完全一致；稳态吞吐约 550 steps/s，checkpoint 约 8.8 MB。
- 增加 `forkserver` 模式的 `SubprocVecEnv` 和 `base_seed + worker_rank` 种子管理；通过多 worker MaskablePPO、mask 与自动重置测试。
- 完成 CPU/CUDA 各 1/2/4/8 worker 的固定 4096 样本 rollout 基准；最快为 CUDA + 8 worker（566.83 steps/s），作为后续正式训练的推荐目标配置；现有事务训练默认设备暂不改变。
- 在 RTX 3050 Laptop 上验证官方 PyTorch 2.10.0 CUDA 13.0 build，并完成 CPU/CUDA MaskablePPO 短基准；整体吞吐几乎相同，正式训练默认 CPU。
- 事务训练记录并固定设备，恢复时不自动切换；CUDA checkpoint 的模型和 optimizer 恢复通过。

## 2026-10-06

- 新增事务式 MaskablePPO Task/Commit 恢复：每 Task 4096 步，完整保存模型、optimizer、环境、RNG 与指标；同盘 rename 后才推进 committed 指针，重启后隔离并重做未提交 Task。
- 新增可恢复的 MaskablePPO smoke training：分块训练、原子 checkpoint、固定 seed 评测、CSV/TensorBoard 日志，以及 tmux 后台 start/status/pause/resume；第一阶段在 20480 步安全停下。
- 新增 Gymnasium `TetrisEnv`：237 维 Observation、1840 动作与 Action Mask；一步执行一个 Core placement，奖励与游戏分数分离，支持固定 seed、Game Over 与最大方块数截断。
- 新增 Python TetrisCore 与 JS/Python Parity Test；验证 Mulberry32、7-Bag、Hold、SRS、计分、Top Out、Legal Placement 和 1840 位 Action Mask 一致。Python mask 基准约 3.24 ms、309.0 masks/s。
- 简化 AI placement BFS：仅以 `(x,y,rotation)` 去重，移除搜索中的 50 ms 计时与 Lock Reset 维度，保留完整 SRS 路径可达性；人工 Lock Delay 不变。基准从 232.39 ms/mask 提升至 4.64 ms/mask。
- 新增 1840 固定动作空间、encode/decode、真实可达 placement 搜索及 Action Mask；支持 SRS、Hold、15 次 Lock Reset，并固定去重与标准路径。
- 固化 Next Preview = 3，新增只读公开观察接口；隐藏 Next 4+、Bag 和 RNG 内部状态，为统一 Agent 接口建立信息边界。
- 固化 Benchmark v1.0 计分：消行按 `100/300/500/800 × level`，Soft Drop 每格 +1，Hard Drop 每格 +2；暂不加入 T-Spin、Combo、Back-to-Back。
- Lock Delay 为 450 ms；每块最多 15 次计时重置，仅成功的接地移动或旋转消耗次数；Hard Drop 仍立即锁定。
- Gameplay RNG 改用可复现的 Mulberry32；7-Bag 支持固定 seed，并保存、恢复当前 RNG 状态。
- 视觉随机性与游戏随机性分离；传统 AI 使用独立随机源，不影响出块顺序。
- 明确 Benchmark v1.0 Top Out：保留出生碰撞 Block Out，加入整块在棋盘上方锁定时的 Lock Out。
- Partial Lock 暂时允许继续，仍正常合并可见部分并消行。
- 将自定义旋转规则替换为标准 SRS，明确维护 `0 → R → 2 → L` 旋转状态。
- JLSTZ 与 I 方块分别使用对应的 Wall Kick 规则；O 方块原地旋转。
- 保留 Hold、Next、计分等已有规则，并兼容旧版游戏存档。
- 完成贴墙、地面、碰撞、旋转失败和现有玩法的可重复测试。
