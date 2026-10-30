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

## PPO、BC 与 Hybrid B12 网页演示

网页现在还可选择 PPO 30M、BC Champion、Hybrid B12。Human、V1 Adapter、V2 Hold-only、V2 Beam-8 保持原有功能。三个神经网络模式需要独立的 Python CPU 推理服务；原 8080 静态网页服务保持原样。服务只接收网页公开状态和合法 Action ID 列表，校验冻结 checkpoint SHA256、官方掩码后返回动作，不推进游戏、不读取 Bag 或 RNG。网页按原有 BFS 路径落子。

从项目根目录启动推理服务：

```bash
web_ip=$(tailscale ip -4 | head -1)
mkdir -p runs/web_inference
tmux new-session -d -s tetris-web-inference -c "$PWD" \
  ".venv/bin/python -m training.web_inference --host $web_ip --port 8081 --web-origin http://$web_ip:8080 >> runs/web_inference/service.log 2>&1"
tmux has-session -t tetris-web-inference
```

游戏本访问 `http://<Tailscale IPv4>:8080/`。推理服务只绑定指定的本机或 Tailscale 地址，浏览器请求只允许对应的 8080 来源。服务状态可访问 `http://<Tailscale IPv4>:8081/api/health`；停止命令是 `tmux kill-session -t tetris-web-inference`。模型请求超时或服务不可用时，网页会停止当前 AI 决策，用户可点击“重试 AI”、暂停或切换模式。刷新后可继续本地保存的对局；B12 的 PPO/BC 采用计数也会保存。

侧栏历史结果来自独立 32-seed、每局最多 100K 的对照评测，与当前网页对局分开显示。100K 触顶是右截断。人工模式的时间为真实游玩时间，AI 模式显示真实演示时间；跨模型生存比较以放置方块数为准。

长局网页链路验证使用 `tests/integration/web-ai-longrun.cjs`，通过真实 8081 HTTP、网页 BFS 执行和逐步 Python Core 对照，每 500 块原子保存可恢复快照及校验和。例如从项目根目录运行：

```bash
node tests/integration/web-ai-longrun.cjs ppo 10000 12345 "$(tailscale ip -4 | head -1)" runs/web_longrun_validation/manual_ppo
```

该脚本以不渲染动画的大步进运行，测得的是验证吞吐量；网页正常播放的速度仍由演示滑块控制。正式长局测试结果见 `reports/experiments/web_longrun_validation.md`。


## Public Lookahead (Experimental)

独立可选混合控制器；默认选项保持V2 Hold-only，Human/V1/V2/PPO/BC/B12继续可用。网页执行官方1840 Action ID的BFS路径，8081加载冻结高分学生epoch1与BC；规划源码和配置固定于a3bfc02（4根、最多4落子、阈值12），不读取隐藏Bag/RNG/Next4+，不训练模型。

- 选择新模式后填Seed并重新开始。得分、消行、块数沿用正式规则。
- “AI决策耗时”含服务器排队、预处理和规划；HTTP耗时含网络和客户端响应解析，与动画倍率分开。动画倍率只改变播放速度，“块/秒”包含动画与等待，不代表纯推理速度。
- 保留/覆盖次数与第三类次数只在真实落子后累计。第三类是覆盖的子集：动作同时不同于学生与BC首选。比例分母为保留+覆盖；切换别的模式不计入，重新开始清零。刷新保留已落子计数，尚未执行的决策不计数。
- 暂停会取消在途请求；恢复后继续。请求失败或10秒超时会停止自动请求并显示“重试AI”，旧请求不能在切换模式/重新开始后落子。
- 网页新增独立历史结果区：32个全新确认seed，最多100K；规划31/32达到上限、BC32/32，规划共同窗口正式得分+10.536%。达到上限为右截断，实验候选仍可能死亡；这不是当前网页对局成绩，也不与旧历史评测混为同一组。

服务从集成分支目录启动（先按根README准备虚拟环境与三份冻结模型）：

```bash
web_ip=$(tailscale ip -4)
mkdir -p runs/web_inference
tmux new-session -d -s tetris-web -c "$PWD"   "python3 -m http.server 8080 --bind $web_ip --directory tetris-game >> runs/web_inference/static.log 2>&1"
tmux new-session -d -s tetris-web-inference -c "$PWD"   "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m training.web_inference --host $web_ip --port 8081 --web-origin http://$web_ip:8080 >> runs/web_inference/service.log 2>&1"
curl -fsS "http://$web_ip:8081/api/health"
```

已有服务时先检查tmux/端口，勿重复绑定。浏览器需能访问同一Tailscale设备的8080与8081，来源必须匹配配置。服务以单线程Torch执行并串行访问模型；最多4个在途推理请求，超出返回503，网页提供重试。不会按浏览器保存服务端游戏状态。

真实Chromium复现测试（Playwright只用于验证，不是网页运行依赖）：

```bash
npm install --prefix runs/model_based_planning_web/browser-runtime playwright@1.64.1
node runs/model_based_planning_web/browser-runtime/node_modules/playwright/cli.js install chromium
node tests/integration/lookahead-browser.cjs 300 3102841000 "http://$web_ip:8080" runs/model_based_planning_web/recheck300 0 '' frozen
node tests/integration/lookahead-controls.cjs "http://$web_ip:8080" runs/model_based_planning_web/recheck-controls.json
```

以上种子仅作工程回归，不作新的独立质量确认。首次300块验证通过后可用同一脚本以cap10000、seed3102841001验证稳定性；原子Core/浏览器断点、逐步公开轨迹与SHA256保留在指定目录。生产验证结果和CPU/响应分位数见[集成报告](../reports/experiments/model_based_planning_web_integration.md)。
