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

## Linux NVIDIA PyTorch

本项目的 `requirements.txt` 默认安装 CPU 版 PyTorch。RTX 3050 Laptop GPU（驱动 595.91.07）已验证官方 `torch 2.10.0+cu130`；先安装基础依赖，再在 Linux NVIDIA 机器上执行：

```bash
.venv/bin/python -m pip install -r requirements-cuda-linux.txt
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

CUDA wheel 使用 [PyTorch 官方索引](https://pytorch.org/get-started/previous-versions/)；无需系统级 CUDA Toolkit。再次安装 `requirements.txt` 会恢复 CPU 版，需要随后重新执行 CUDA 安装命令。

## 多进程环境短基准

`training/vector_env.py` 用 `SubprocVecEnv` 的 `forkserver` 模式创建独立的 `TetrisEnv` worker，种子为 `base_seed + worker_rank`；`action_masks()` 保留在各 worker 内。基准固定每轮 `n_envs × n_steps = 4096`、每组 16384 训练步，结果写在被 Git 忽略的 `runs/vector_env_benchmark/`，不保存模型。

```bash
.venv/bin/python -m scripts.benchmark_vector_env --device cpu --n-envs 4 --run-dir runs/vector_env_benchmark/cpu_env4
.venv/bin/python -m scripts.benchmark_vector_env --device cuda --n-envs 4 --run-dir runs/vector_env_benchmark/cuda_env4
```

该入口由 `if __name__ == "__main__"` 保护，可安全启动子进程。多进程环境目前用于 benchmark；事务式训练的完整多 worker 状态保存尚未接入。

[8 组实测结果](reports/performance/vector-env-benchmark-2026-10-07.md) 中最快的是 CUDA + 8 worker（566.83 steps/s）。后续可据此设计可恢复的多 worker 正式训练；现有事务式训练入口仍默认单环境 CPU。

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

## 单环境事务式恢复实验（历史结果）

事务实验使用独立的 `runs/ppo_transaction_test_seed42/`，与上面的 smoke run 互不影响。每个 Task 是完整的 4096 步（4 个 PPO rollout/update），目标 32768 committed steps。只有模型、optimizer、环境、RNG、指标和 TensorBoard 文件全部写盘，并从 `working/` rename 到 `committed/` 后，`transaction_state.json` 才推进 committed 指针。未完成的 Task 在恢复时标记为 abandoned，从最后的 committed Task 重新执行。

原单环境 run 保留在 `runs/ppo_transaction_test_seed42/`；可用 `TETRIS_TRANSACTION_RUN_DIR=$PWD/runs/ppo_transaction_test_seed42 ./scripts/train_transaction.sh status` 查看。它仍使用已记录的 CPU 配置，不会被下面的新 run 覆盖。

## 8 worker CUDA 事务式恢复实验

`./scripts/train_transaction.sh` 现在默认使用独立的 `runs/ppo_transaction_vec8_cuda_test_seed42/`，配置固定为 CUDA、8 个 `forkserver` worker、`n_steps=512`、Task 4096 步、目标 32768 步。worker seed 为 `42 + worker_index`。每个 Task 完整保存模型/optimizer、主进程 CPU/CUDA RNG、8 个环境逻辑状态与 observation/mask digest，文件校验通过并同盘 rename 后才推进 committed 指针。恢复时只从最近 committed Task 重建 worker 并逐个核对 digest；未提交 Task 被隔离到 `abandoned/`。TensorBoard 留在 per-Task 目录，正式 CSV 只含 committed Task。

```bash
./scripts/train_transaction.sh start
./scripts/train_transaction.sh status
# 用户自行重启机器并回到 Ubuntu 后：
./scripts/train_transaction.sh resume
```

训练由 `tetris-transaction` tmux session 在后台运行；没有 systemd 自动启动。`status` 同时显示 committed 与 working 进度，关机恢复时以 committed 步数为准。单环境与多环境的 CPU/CUDA 性能比较见 [设备报告](reports/cuda/2026-10-06.md)和[多 worker 报告](reports/performance/vector-env-benchmark-2026-10-07.md)。
