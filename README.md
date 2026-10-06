# Tetris AI Benchmark

`tetris-game/` 中的 JavaScript 网页是 Benchmark v1.0 规则参考实现；`training/tetris_core/` 是通过 JS/Python Parity Test 的 Python 训练规则核心。Python Core 必须继续通过 Parity Test 才视为有效。

`training/env/tetris_env.py` 提供 Gymnasium `TetrisEnv`：Observation 为固定 237 维 `float32`，Action 为 `Discrete(1840)`，`action_masks()` 返回 1840 个布尔值。一步对应一个完整方块的合法最终 placement；默认 `max_pieces=10000`，达到上限且未 Game Over 时截断。环境直接调用 Core，RL reward 与正式游戏分数分开计算。

在项目根目录运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/test_tetris_env.py -q
.venv/bin/python tests/stress_tetris_env.py --steps 100000
.venv/bin/python -m unittest discover -s tests/parity -v
```

环境文件不包含 PPO 或其他训练程序。

## 首轮 MaskablePPO smoke training

安装依赖后，在项目根目录运行：

```bash
.venv/bin/python -m pip install -r requirements.txt
./scripts/train_control.sh start
./scripts/train_control.sh status
./scripts/train_control.sh pause
./scripts/train_control.sh resume
```

训练由 `tmux` 在后台运行，日志与 checkpoint 写入 `runs/ppo_smoke_seed42/`（不提交 Git）。使用 237 维 Observation、1840 动作的 MaskablePPO `MlpPolicy`，每个 5120 步 chunk 完成 PPO 更新后原子保存 `latest.zip`。第一阶段在 20480 步自动停为 `awaiting_resume_test`；此时可关机，之后回到 Ubuntu 执行 `resume`，累计训练到目标 50000 步。由于 rollout 固定为 1024 步，最终安全边界是 50176 步。`pause` 会等待当前 chunk 保存完成再退出；用 `status` 确认 `Safe to shutdown: YES` 后再自行关机。没有配置开机自动恢复。

## 事务式恢复实验

事务实验使用独立的 `runs/ppo_transaction_test_seed42/`，与上面的 smoke run 互不影响。每个 Task 是完整的 4096 步（4 个 PPO rollout/update），目标 32768 committed steps。只有模型、optimizer、环境、RNG、指标和 TensorBoard 文件全部写盘，并从 `working/` rename 到 `committed/` 后，`transaction_state.json` 才推进 committed 指针。未完成的 Task 在恢复时标记为 abandoned，从最后的 committed Task 重新执行。

```bash
./scripts/train_transaction.sh start
./scripts/train_transaction.sh status
# 用户自行正常关机并再次进入 Ubuntu 后：
./scripts/train_transaction.sh resume
```

事务训练使用独立的 `tetris-transaction` tmux session；`status` 同时显示 committed 与 working 进度。当前实验不配置开机自动恢复，也不要求先人工 pause。正式训练指标仅由 committed Task 重建，未提交 Task 的指标不会进入 `training_metrics.csv`。
