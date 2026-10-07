# 修改日志

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
