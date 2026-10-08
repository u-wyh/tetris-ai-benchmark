# 霓虹方块

无需安装依赖，直接用浏览器打开 `index.html` 即可游玩。

JavaScript 网页是 Benchmark v1.0 的参考实现；`training/tetris_core/` 是 Python 训练核心。Python Core 只有通过 `python3 -m unittest discover -s tests/parity -v` 的 JS/Python Parity Test 才视为有效。测试逐步比较公开状态、规则结果和 1840 位 Action Mask；Python 核心仅执行 AI Benchmark 的最终 placement，不模拟人工模式的实时 Lock Delay。

## 操作

- `←` / `→`：左右移动
- `↑` / `X`：旋转
- `↓`：加速下降
- `空格`：直接落下
- `C`：暂存或交换方块（每个方块锁定前只能使用一次）
- `P` / `Esc`：暂停

移动端可使用画面下方的 Hold 按钮。暂存槽为空时会从 Next 取出下一个方块；交换已有暂存方块时不会消耗 Next。换出的方块会从默认位置和方向重新出生。

游戏使用七袋随机算法，每消除 10 行提升一级，下落速度会随等级逐渐加快。最高分会保存在当前浏览器中。

## Benchmark v1.0 计分

单消、双消、三消、四消分别获得 `100/300/500/800 × 当前等级` 分；Soft Drop 每下降一格加 1 分，Hard Drop 每下降一格加 2 分。每累计消除 10 行升一级，跨过升级门槛的那次消行按消行前的等级计分。v1.0 不计算 T-Spin、Combo、Back-to-Back 或 Perfect Clear 奖励。

`score` 是正式游戏规则和 Benchmark 指标。未来强化学习的 reward function 应独立计算，不能修改或污染正式游戏分数。

## Benchmark v1.0 Observation Policy

公开观察接口 `getPublicObservation()` 返回只读副本：20×10 Board、Current Piece（类型、矩阵、旋转状态和位置）、Hold 与是否可用、Next 3、Score、Level、Lines、游戏阶段与 Game Over，以及锁定计时和重置次数。Hold 是否可用按游戏规则判断，与当前传统 AI 是否使用 Hold 无关。

Next 4+、7-Bag 剩余顺序、初始 seed、Gameplay/AI RNG state 和浏览器存档数据均不公开。内部队列仍保持至少 5 个方块。网页菜单中的 V1 Adapter 与 V2 均只接收公开观察；保留的原始网页 V1 代码仍按旧方式读取 `queue[0]`，仅用于历史行为回归。

## Benchmark v1.0 Legal Placements

动作空间固定为 1840：`Hold(2) × Rotation(4) × X(10) × Y(23)`。`rotation` 为 `0/R/2/L` 对应的 `0/1/2/3`；`x`、`y` 是最终占据格子的最左列和最上行，`y ∈ [-3,19]`。`encodeAction()` 与 `decodeAction()` 互逆；`getLegalPlacements()` 返回真实操作路径，`getActionMask()` 返回 1840 个布尔值。Agent 每块选择一次最终 placement，执行器以后可按返回路径执行。Mask 只表示机械可达性，致死 placement 仍可选。

### Human Gameplay Rules

人工游戏继续使用 SRS 实时操作、450 ms Lock Delay 和每块最多 15 次接地重置；Hard Drop 立即锁定。

### AI Benchmark Reachability

Agent 每块选择一个最终 placement。BFS 从当前状态出发，只以 `(x,y,rotation)` 去重，按 `Left → Right → RotateCW → Down` 扩展；可先下降再横移或旋转，包括 SRS Wall Kick 和 Tuck。只有当前状态合法且继续 Down 会碰撞时才形成候选终点。同一 Hold 分支中占据格相同则只保留最小 action ID。搜索不模拟 Lock Delay、Lock Reset 时间或手速，也不读取隐藏队列或 RNG，不修改当前对局。网页中的 V1 Adapter 和 V2 均从此模块取得合法落点与执行路径；原始网页 V1 代码仍可能建议不可达落点。

性能基准可在项目根目录运行 `node tests/placement-benchmark.cjs`。在 4 个固定 seed 的合法局面上各测 3 次，本次环境测得平均 4.64 ms、中位数 4.41 ms、约 215.6 masks/s，平均搜索 1194 个状态、47.8 个 placements；旧实现平均 232.39 ms、约 4.3 masks/s。

需要复现一局时，可在浏览器开发者控制台运行 `resetGame({ seed: 12345 })`。seed 为 0 到 4294967295 的整数；不指定时新游戏会自动生成 seed。当前 seed 和随机数状态随对局一起保存在浏览器中。

在侧栏选择 Human、V1 Adapter、V2 Hold-only 或 V2 Beam-8。默认选中 Hold-only；Beam-8 搜索两层，计算较慢。输入 0–4294967295 的 Seed，点击“重新开始”可复现同一开局；清空输入后重新开始会生成随机 Seed 并显示出来。可暂停、继续、调整演示速度。网页 AI 在独立 Worker 中搜索，主线程逐步执行 BFS 路径；AI 的 Down 不计人工 Soft Drop 分。Human 的实时操作和计分规则不变。

切换到其他标签页或最小化浏览器后，AI 会通过后台计时器继续运行；返回页面时，画面会显示后台运行后的最新棋盘和分数。

对局会自动保存在浏览器本地，包括棋盘、当前方块、后续队列、暂存方块及本轮使用状态、积分、等级、算法模式、方块放置数量、存活时间和暂停状态。刷新页面后会从最近的状态继续运行。

从项目根目录启动 Tailscale 网页服务：

```bash
tmux new-session -d -s tetris-original-web -c "$PWD" 'python3 -m http.server 8080 --bind "$(tailscale ip -4)" --directory tetris-game'
tmux ls | grep tetris-original-web              # status
tmux kill-session -t tetris-original-web         # stop
```

当前服务若已运行，无需重复 start。通过 `http://<Tailscale IPv4>:8080/` 访问。跨语言验证可运行 `.venv/bin/python -m unittest discover -s tests/parity -v`；网页测试运行 `node --test tests/*.test.cjs`。
