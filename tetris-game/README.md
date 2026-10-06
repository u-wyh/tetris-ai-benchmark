# 霓虹方块

无需安装依赖，直接用浏览器打开 `index.html` 即可游玩。

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

Next 4+、7-Bag 剩余顺序、初始 seed、Gameplay/AI RNG state 和浏览器存档数据均不公开。内部队列仍保持至少 5 个方块；当前传统 AI 尚未迁移到此接口，仍直接读取棋盘、活动方块及 `queue[0]`，只前瞻 1 个 Next。

## Benchmark v1.0 Legal Placements

动作空间固定为 1840：`Hold(2) × Rotation(4) × X(10) × Y(23)`。`rotation` 为 `0/R/2/L` 对应的 `0/1/2/3`；`x`、`y` 是最终占据格子的最左列和最上行，`y ∈ [-3,19]`。`encodeAction()` 与 `decodeAction()` 互逆；`getLegalPlacements()` 返回真实操作路径，`getActionMask()` 返回 1840 个布尔值。Agent 每块选择一次最终 placement，执行器以后可按返回路径执行。Mask 只表示机械可达性，致死 placement 仍可选。

搜索先检查 HardDrop 终点，再按 `Left → Right → RotateCW → Down` 扩展；最短路径优先，同长路径按此固定顺序决定。同一 Hold 分支中占据格相同则只保留最小 action ID。离散锁定模型以每次成功操作 50 ms 计时，首次接地后持续计时，最多 15 次接地重置；HardDrop 立即锁定。搜索不读取隐藏队列或 RNG，也不修改当前对局。当前传统 AI 尚未使用此模块，可能建议被障碍隔开的不可达落点。

性能基准可运行 `node tests/placement-benchmark.cjs`（项目根目录）。在 4 个固定 seed 的合法局面上各测 3 次，当前环境测得平均 232.39 ms、中位数 219.57 ms、平均 47.8 个 placements，约 4.3 次 mask/s；这只是当前实现的基线。

需要复现一局时，可在浏览器开发者控制台运行 `resetGame({ seed: 12345 })`。seed 为 0 到 4294967295 的整数；不指定时新游戏会自动生成 seed。当前 seed 和随机数状态随对局一起保存在浏览器中。

点击右上角“AI 选手”可开启自动玩家。AI 会分析当前棋盘及下一个方块，并在画面中逐步完成旋转、移动和下降；再次点击即可随时切回人工操作。

切换到其他标签页或最小化浏览器后，AI 会通过后台计时器继续运行；返回页面时，画面会显示后台运行后的最新棋盘和分数。

对局会自动保存在浏览器本地，包括棋盘、当前方块、后续队列、暂存方块及本轮使用状态、积分、等级、AI 开关和暂停状态。刷新页面后会从最近的状态继续运行。
