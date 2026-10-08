# PPO-Shaped V1：2M 对照实验协议

本实验只改变 MaskablePPO 的 Reward。每次合法落点后，以 Python TetrisCore 的真实棋盘计算洞：同列中空格上方曾有占用格即为洞。先完成放置和消行，再计 `new_holes = max(0, holes_after - holes_before)`；训练 Reward 为原 Raw Reward 减去 `0.10 × new_holes`。洞减少不加分，游戏 Score、公开 Observation、1840 动作空间、Action Mask、SRS/Top Out 等规则及 PPO 超参数不变。`TetrisEnv()` 默认系数为 0，保留 Raw Reward。版本分别为 `placement-reward-v1` 和 `placement-reward-hole-v1`。

正式配置为 [`configs/ppo_hole_v1_2m_seed42.json`](../../configs/ppo_hole_v1_2m_seed42.json)：CUDA、8 workers、seed 42–49、每 Task 4096 步，目标 2,000,000 步，实际完整 Task 边界 2,002,944 步。新策略由相同随机 seed 从 Step-0 初始化；启动时逐参数对照 PPO-Raw 的 Step-0 模型。事务恢复同时校验 run、task 和 worker 环境中的 Reward 版本与系数。每 Task 只在提交后汇总逐步 Raw/Shaped Reward、新增洞及已结束 episode 的长度、消行和分数；未结束 episode 不计入完整 episode 均值。

检查运行状态：

```bash
.venv/bin/python -m training.train_vector_transaction --run-dir runs/ppo_hole_v1_2m_seed42_lambda010 --status
tmux capture-pane -pt tetris-ppo-hole-v1 | tail -30
```

训练完成后，使用固定 validation seeds `100000..100031`，每局上限 5000 块，运行独立配对评测：

```bash
.venv/bin/python -m training.evaluation.compare_hole_v1 --output runs/ppo_hole_v1_2m_seed42_lambda010/paired_validation_2m.json
```

该命令读取 Raw 和 Shaped 的 2,002,944 步模型，均以 `deterministic=True` 和合法 Action Mask 评测，逐 seed 记录存活、消行、游戏分数、造洞次数/总数及终局洞数。比较平均值、中位数、P90、上限截断和配对差异；跨 Reward 版本不比较训练 Reward 大小。特别检查造洞减少但存活下降、快速 Game Over 比例上升等潜在规避行为。正式 final-test seeds 不用于选择奖励系数，未经配对比较不扩展至 10M。
