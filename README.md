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
