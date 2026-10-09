# Candidate-Scoring Maskable PPO 设计

## 目标与边界

在固定 1840 动作空间中，每个合法最终落点由**同一套评分网络**给出一个 logit；现有合法动作掩码把不可达动作排除后形成策略分布。游戏规则、7-Bag、SRS、Hold、Top Out、计分和 Action ID 不变。候选特征只能由公开 Observation（当前棋盘、活动方块、Hold 可用性、Next 3 等）和当前动作的确定性一步放置计算；不能运行下一块出生、读取完整 queue、bag、RNG、存档或 final-test 数据。

## 当前代码与修改点

现有 `TetrisEnv` 返回 237 维 `float32`；`MaskablePPO(MlpPolicy)` 把它送入 `[256,256]` actor/critic，actor 最后直接输出 1840 个位置相关 logit。合法掩码由 `TetrisEnv.action_masks()` 提供，`MaskablePPO.collect_rollouts()` 把观察和掩码写入 `MaskableRolloutBuffer`，更新时调用 policy 的 `evaluate_actions(obs, actions, action_masks)`。因此候选评分需要连贯修改以下接口：

1. **Observation**：新增版本化的 `Dict` 观察，例如 `state: float32[237]`、`candidate: uint8[1840,16]`。每个动作行保存该合法落点的公开、确定性特征；非法行置零，并由正式 Action Mask 判定是否合法。候选行与现有 Action ID 一一对应。`uint8` 须定义固定量化表与范围；若精度不足，改用 `float16/float32` 并重新测内存。初始化、每步、恢复后的 observation 和 mask 必须从同一 Core 状态生成并校验 digest。
2. **候选特征**：以当前公开棋盘为起点，使用 `get_legal_placements()` 的 occupied cells 按 Core 的可见区合并和消行规则计算落点后的棋盘；不触发下一块出生。16 维初版包括：落点类型与 Hold 标记、归一化的 x/y/rotation、消行数、后局面的总高度/最高列/洞数/covered holes/凹凸度/井深、相应高度与洞数变化、触顶风险标记。具体编码与量化范围在实现前冻结；特征计算需要对比真实 `TetrisCore.step()` 的落点后棋盘，尤其测试 Partial Lock、Lock Out 和消行。公开 Next 仅在空 Hold 决定当前将放置的块时使用，不借此模拟下一块。
3. **Policy**：新增 `MaskableActorCriticPolicy` 子类及可序列化的自定义 `MaskablePPO` 配置。`state` 经共享或独立 encoder 得到 64 维上下文；16 维候选特征经共享 MLP `[64,64]`，与上下文拼接后逐合法候选输出标量 logit，再散射到 `[batch,1840]`。沿用 `MaskableCategorical` 与正式掩码求概率、采样、log-prob 和 entropy。critic 只读取公开 `state`，输出一个状态价值。须覆盖 `forward`、`evaluate_actions`、`get_distribution` 和 `predict_values` 的一致路径，并测试全非法掩码的明确错误。
4. **Rollout**：优先复用 SB3-Contrib 的 `MaskableDictRolloutBuffer`，避免改 PPO 损失、GAE 或 clip。`collect_rollouts()` 已能获取 Dict 观察和掩码；需确认自定义 policy 在采样、训练、独立推理三处产生相同 logits。候选数据只在观察中存一次；mask 仍单独保存，worker 端将合法落点与特征生成结果缓存到下一次状态变化，避免重复 BFS。
5. **Checkpoint/恢复/评测**：run 配置写入 `observation_version`、候选特征版本、policy 类路径、网络和量化表；SB3 `model.zip` 必须能在独立 Python 进程加载自定义类。事务 Task 原子保存 optimizer、主/worker RNG、环境状态；恢复后同时核对 Dict 观察各字段和 mask digest。独立评测按 checkpoint 的版本创建对应环境，仍用相同 32 validation seeds、5,000 块上限及 Action Mask；`best`、milestone 与 final-test 代码不得把新旧观察混读。

## 性能预算与验证门槛

现有 V3 训练尾部约 430–440 steps/s，约 75–90 分钟完成 2M；候选特征的 CPU 构造、8 worker 进程通信、PPO 每 epoch 重算网络会降低速度，当前尚无实测值。若直接保存 `float32[1840,16]`，4096 样本仅候选矩阵就约 **460 MiB**；`uint8` 约 **115 MiB**，加现有 1840 位 mask 的 `float32` 约 **29 MiB**。实际 optimizer、激活和进程复制额外占用必须在 RTX 3050 4GB 上测量。网络只对 mask 为真的合法行进行 `gather → shared MLP → scatter`，避免训练 minibatch 对全部 1840 行建立激活；GPU 显存与每秒决策量应通过 8-worker、4096 样本 rollout 和一次 PPO 更新实测。若生成候选特征需要重复 BFS，应先共享 Core 的合法落点缓存。

## 公平对照与首轮训练计划

先写公开信息泄露测试、特征与真实 Core 一步落点的 parity 测试、全部 1840 action 对齐测试，以及采样/训练/保存恢复后 logits 一致测试。随后做 CPU/CUDA 8-worker 的 4096 样本吞吐与显存 smoke test，确认事务中断恢复与独立评测不会改变后续轨迹。再建立**独立 run，从 seed42 随机初始化训练 2,002,944 步**；优先用 Raw Reward，使与 Raw 2M 的区别集中在候选评分架构。PPO 超参数、8 worker、n_steps=512、Task=4096、validation seeds `100000..100031`、每局 5,000 块与 Raw 2M 一致。对照同时报告存活均值/中位/P90、逐 seed 配对 bootstrap 区间、消行、正式分数、洞、高度、中央尖塔率、吞吐和峰值内存。只在 validation 阶段选择方案；final-test seeds 不参与调参。这个计划尚未开始训练。
