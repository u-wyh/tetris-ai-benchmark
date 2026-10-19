# 修改日志

## 2026-10-17

- 按用户要求安排 PPO 30M 三个新训练种子的固定预算跟进：每个种子配对普通起点和危险课程各 503808 样本；等待现有 300 万块对局完成后自动启动。新的 16 个相同开发种子将比较原版 PPO、BC、B12、六个新模型和不训练的传统 V2 Hold，按共同存活得分与长期生存分开报告。
- 为课程训练结束后的 300 万块长局建立 8 种子、6 策略、8 CPU worker 的独立 tmux 队列。评测快速路径复用正式合法落点、跳过与动作排序无关的重复计算；逐局结果原子保存，每 1 万块保存可恢复局内快照。已完成旧路径对局按 SHA256 清单复用，冻结模型与 TetrisCore 规则未变。独立对照显示 PPO 1000 块整局约提速 1.14 倍，传统 V2 约提速 1.7 倍；长局总耗时仍以实际运行记录为准。

## 2026-10-15

- 新增 PPO30M 冻结 Top-5 公开有限前瞻 S1/S2：官方合法 BFS、可见 Next/Hold 模拟、`unknown` 信息边界、状态去重、保守 PPO 概率约束与逐局可恢复的 tmux 评测队列。独立 990000/990100/990200 种子组及参数已冻结，先 smoke 再按性能和开发门槛决定是否继续。
- 20K 开发组完成 16 个配对种子：S1/S2 均 15/16 触顶，平均存活 18932，高于 PPO 的 17895；S2 触发率不足 1%，两者都出现 seed 990109 的 2908 块新增早死。按右截断和冻结质量筛选，继续同组 50K 检验，不提前宣称长期改进。
- 同组 50K 的 PPO 与 B12 各 16 局完成，分别平均存活 37529/46055；S1 9/16、S2 0/16 时因同快照反事实发现 S1 在 seed 990109 第 2849 块的单次替换导致 59 块后死亡，而原 PPO 动作分支至少续存 1000 块，提前停止搜索扩展及 100K 确认。明确记录部分结果不可用于均值胜负比较，不进行蒸馏。
- 新增 PPO30M 前瞻失败逐项评分审计和同 Core 快照消融，发现旧 S1 的额外四消奖励使已知危险替换越过切换门槛。实现独立 V3-A 去额外四消奖励、V3-B 可见 Hold 连续性检查，以及压缩全量替换事件、原子逐局结果和 tmux 可恢复配对评测；新旧模型及 S1/S2 权重与规则保持冻结。
- V3 新开发 20K 与同组 50K 配对完成。V3-A 的 seed 991104 同快照确认有害替换，20K 即淘汰；V3-B 虽没有 Hold 改写、四消比例接近 PPO，但 50K 平均存活 41545/P10 22682，低于 PPO 的 44471/29534，按预设 no-go 停止，未启动独立 100K 或神经网络训练。修复已淘汰候选缺席 50K 汇总时的恢复错误。
- 启动 PPO30M 长期死亡机制研究：审计 68 条历史 PPO 对照/轨迹记录并标记旧用途；新增独立 992000–992031 长局种子、上限 64 局的预注册扩充规则、逐步压缩轨迹、稀疏完整 Core/RNG 快照、SHA256 原子恢复与确定性重放校验。不修改冻结策略或正式游戏规则。
- 补齐 PPO30M 死亡研究自主后台队列：完成长局后逐 seed 重放、死亡窗口与配对存活对照、分层单动作反事实 1000/5000 块、右截断与共同得分分析，再生成报告并运行 Python/Parity/JS 回归；分支原子保存和哈希恢复，最多 120 状态、每状态 6 个候选，不训练新模型。
- 新增独立 PPO30M 危险状态课程强化学习路线：历史训练快照与新死亡轨迹按来源构建完整 Core/RNG 数据集；A 正常开局、B 70% 课程重置，均采用真实死亡 −20 与冻结 PPO30M 的 Masked KL。事务训练记录危险来源、死亡和窗口截断；开发 16-seed 的冻结门槛控制是否追加 B 至 2M 样本及启动独立 32-seed 长局确认。队列等待当前死亡分析完成后才占用 GPU/CPU。
- 新增独立 PPO30M 辅助长期风险 MLP 与整局隔离、右截断标签、可恢复训练及留出轨迹评测。复用旧 PPO 已执行动作轨迹训练 5 epoch；最佳模型的 1000 块死亡 PR-AUC 0.147，低于高度加洞数基线 0.274，按门槛停止，不启用 Top-5 重排序或在线控制。详见 `reports/experiments/ppo30m_risk_predictor_evaluation.md`。
- 新增 PPO 30M 夜间 KL 强化学习微调实验：Raw 与 0.003 高度风险奖励两组均从冻结 PPO30M Actor/Critic 初始化，并以原 PPO 为 KL 参考；新实验训练单局上限 100000，8-worker、4096 样本原子任务、两阶段 503808/2002944 样本上限与独立配对长局验证。新增显存、奖励分量、梯度及冻结正常状态策略保留监控，旧模型和正式计分不变。两方案均通过完整 Task 与恢复试运行，后台队列继续执行。详见 `reports/experiments/ppo30m_overnight_finetune_design.md`。

## 2026-10-16

- 完成 PPO30M 死亡机制研究：32 局中 19 局真实死亡、13 局 100K 右截断；完成 120 状态、541 个 1000 块及 349 个有条件 5000 块同快照分支，报告按有限窗口和右截断解释结果。
- 危险课程数据集冻结为 210 个完整快照，Normal-Start 与 Curriculum 各有 1 个 4096 样本事务 smoke task 并通过哈希复核。正式训练受 Ubuntu `6.8.0-147` 缺失对应 NVIDIA 595-open 模块阻断；不改驱动分支、不自动重启或降级 CPU。诊断及恢复边界见 `reports/experiments/ppo30m_cuda_recovery_diagnostic.md`。

## 2026-10-14
- 完成 PPO 30M 多候选反事实试点：复用 V2 训练组快照，冻结 46 个状态、368 个合法候选，从同快照 PPO 续控 256/1000 块并逐分支原子保存。按同一 A 类标准，多候选在 9 个状态找到 28 个动作，BC 单候选在 4 个状态找到 4 个；24 个 A 动作非 BC，20 个来自 PPO Top5。多候选每 CPU 小时 A 状态产出低于 BC 对照，按预设门槛停止第二批，不启动训练。详见 `reports/experiments/ppo_multiaction_counterfactual_results.md`。
- 局部纠错主动反事实 V2 完成 556 条：新增 A 22（训练 19、独立样本验证 3），通过率 3.96%，低于 V1 的 9.42%；合并后训练/验证 A 仅 37/6，停止下一轮正式模型训练。新增近死窗口与远距/右截断分层、534 条 B 类拒绝原因、V2 Hold 真实方块序列变化及不可变数据集哈希复核。详见 `reports/experiments/local_correction_counterfactual_v2.md` 和 `local_correction_dataset_v2.md`。
- V2 反事实完成后自动从各原子分支生成不可变的 `dataset_rows.jsonl.gz`，补齐 Episode ID、PPO 原动作概率、离线死亡距离标记、逐条哈希与数据分片 SHA；后台收尾程序生成报告、记录状态并推送 Git。
- 补充局部纠错 V1 的 Hold／Next 一步公开状态核查、四消深井粗代理、202 条 B 类首因与重叠拒绝原因统计，以及逐条专家动作概率/排名和过拟合曲线；新增 V2 反事实、诊断、数据版本三份报告生成器。
- 新增 PPO 30M 局部纠错失败诊断与主动反事实数据扩充 V2：逐条审计 21 条可信标签的专家动作概率/排名，隔离的小样本过拟合和梯度检查；从 V1 冻结轨迹中按 episode 隔离、时间间距及随机配额选取最多 556 个新分支，结果独立原子保存，可恢复且不覆盖 V1。正式模型训练和长局评测仍受数据质量门槛约束。
- PPO 30M 局部纠错 V1 三轮小规模训练及 32-seed、20K 开发评测完成。三轮均未通过预设继续门槛：前两轮 P10 生存不足，第三轮四消事件占比比纯 PPO 低 5.19 个百分点。停止 V2、100K 长局与独立确认；补充逐 seed 配对、共同存活区间得分和失败诊断报告。详见 `reports/experiments/ppo30m_local_correction_comparison.md`。
- 新增 PPO 30M 局部专家纠错独立流水线：冻结模型真实轨迹、优先真实死亡前状态的同状态 BC 反事实筛选、A/C 类 Actor 监督与 masked KL、V1/V2 阶段门槛、可恢复开发及独立确认评测；收尾程序自动发布成功或科学上合理的提前停止结果。旧 checkpoint、TetrisCore、Action ID、网页和历史结果未修改；大型产物只写入 `runs/ppo30m_local_correction/`。
- 对 Hybrid B12 六个失败局完成确定性重放、同 Core 状态的 PPO/BC 反事实分支和可见 Next 落点诊断；保留有限观察窗与归因限制。新增独立 C1 保守拦截和 C2 Hold／可见下一块空间保护规则，冻结新开发 16-seed 与确认 32-seed 协议；原 B12 和两个模型保持不变。详见 `reports/experiments/hybrid_b12_failure_analysis.md`、`hybrid_v2_design.md`。

## 2026-10-13

- 增加后台融合评测收尾程序：确认组完成后提交并推送最终报告与操作记录；失败时单独记录状态，不把未完成结果当作结论。
- 新增冻结 PPO 30M + BC 双策略融合控制器：状态切换 A 含滞回与危险动作拦截 B 分开评测；公开状态一步风险规则、每步策略连续区间、切换风险记录、双模型 SHA/版本核验。独立后台队列等待 DAgger 完成后使用隔离开发种子运行可恢复对照；不训练或改动基础模型。详见 `reports/experiments/candidate_ppo30m_bc_hybrid.md`。
- PPO 节点评测按独立种子并行运行，保留正式结果顺序；长局阶段先测本机不同进程数的整批吞吐量，再自动选择最快配置。串并行对照及定向测试通过。
- 新增项目级 `OPERATIONS.md` 操作流水账，按 Git 与运行文件补录模仿学习实验和评测加速的重要节点；后续重要操作、暂停、恢复与失败持续追加。评测环境复用当前状态合法落点，同机短局约快 39%，保留原训练环境和 Core。
- 冻结 20M PPO + BC epoch 3 Champion 并核验模型/数据/教师版本，排队运行全新 30-seed、100,000 块 BC 与 V2 Hold 对照；新增 BC+普通 PPO 退步的动作漂移、KL 和训练曲线分析。实现独立 KL-PPO 事务训练、真实 Core DAgger 采集与混合监督训练及可恢复后台队列，正式生存结果仍待完成。详见 `reports/experiments/candidate_imitation_next_stage.md`。
- 20M PPO + BC 在与传统 V1、V2 Hold 相同的 10 个 validation seeds、100,000 块上限下全部 10/10 到达上限；V2 Hold 同为 10/10，V1 为 1/10、平均存活 26,002.6 块。BC 与 V2 在此上限内持平，超出上限的真实寿命仍未知。详见 `reports/experiments/candidate_bc_vs_traditional_100k.md`。

## 2026-10-12

- 新增独立 V2 Hold 模仿学习路线：真实 Core 专家轨迹分片、公开状态与合法 Action ID 校验、SHA256 可恢复采集、从 20M PPO 复制的 Actor-only BC、独立 2M PPO 微调初始化和 30M 完成后自动推进的后台队列。256 条双 seed 烟雾样本上，BC 验证 Top-1 从 35.9% 提至 38.3%；完整生存对比仍待队列完成。详见 `reports/experiments/candidate_imitation_learning.md`。

## 2026-10-11

- 新增 Candidate PPO 20M 独立选模与测试队列：核验 7M/19M/20M checkpoint 和 20M 长局结果，按统一 32-seed、20,000 块验证冻结 Champion；随后使用未参加旧实验的 400000–400099 种子运行 100 局独立测试，再串行运行 V2 Hold-only 配对评测。逐 seed 原子保存，队列支持恢复，旧 Champion 与旧 V2 结果保留。

## 2026-10-10

- 新增 Candidate Champion 与传统 V2 Hold-only 的 100-seed、50000 块正式配对队列：复用已冻结 Candidate 结果，严格核对历史 V2 配置与 Core 代码 SHA，传统策略逐 seed 原子保存并可恢复；完成后自动汇总截断存活、得分、配对 bootstrap 区间和实际吞吐。未训练新模型。
- Candidate PPO Champion 从 7M/8M/9M/10M 统一 32-seed、20000 块验证中预先选出 7,000,064 steps；冻结后完成 100-seed、50000 块正式测试：平均存活 20247.0 块、10/100 局达到上限。详见 `reports/experiments/ppo_candidate_champion_selection.md` 和 `reports/experiments/ppo_candidate_final_100seed.md`。

- 新增 Candidate PPO Champion 自动队列：核验 7M–10M milestone SHA256 与训练配置，按固定 32-seed/20000 块逐局持久化长寿命验证，使用预先固定的存活优先规则选模并冻结；随后单次执行 100-seed/50000 块正式测试、Raw 已冻结报告对照和传统算法公平比较成本方案。独立 tmux 队列可从已完成局恢复，不重新训练或使用正式测试调参。
- Candidate-Scoring PPO 从原 2M checkpoint 完整续训至 10,002,432 committed steps，原 Task 489 与全部历史评测保留；CUDA 8-worker 训练、里程碑、长寿命验证及最终 checkpoint 均完成，未运行 final-test seeds。
- 同一批 32 个 validation seeds、5000 块上限下，Candidate 10M 平均存活 4543.12 块、平均分 33,924,706.25，27/32 局达到上限；Raw 10M 分别为 198.06 块、45,712.50 分、0/32 局达到上限。逐 seed 对照见 `reports/experiments/ppo_candidate_raw_10m_comparison.md`。
- 独立 16-seed、20000 块长寿命验证中，Candidate 10M 平均存活 12,677.7 块、5/16 局达到上限；7M 里程碑为 15,247.4 块、9/16 局达到上限。长期生存尚不稳定，不能据此宣称超过传统模型的 100000 块能力。

## 2026-10-09

- 完成 Candidate-Scoring PPO 2M 对照审计：32 个 validation seeds 平均存活 3242.75 块、中位数 3532.5 块；14/32 局在 5000 块上限截断，逐 seed 结束标志与块数一致。报告补充全部逐 seed 结果与截断说明。
- 新增独立 10M 续训准备：验证已完成 Task 的完整 manifest、CUDA RNG 和 8 worker 状态，复制原 run 到新目录，保留 2M 原件与固定 Task，严格校验除目标外的全部训练配置。CUDA 对照测试覆盖中断恢复，并比对最终权重、optimizer、RNG 和 worker 状态。
- 10M 续训保留原 5000 块 periodic 与 10000 块 milestone 协议，另在 2M 起点及后续 milestone 使用 16 个固定 validation seeds、20000 块上限的独立长寿命协议，记录截断率；训练完成后自动生成与 Raw 10M 的 32-seed 配对报告，不调用 final-test seeds。
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

## 2026-10-18

- 新增 PPO 30M 新候选的一次性 32-seed 独立确认评测队列：冻结模型与协议、精确种子审计、每局 10000 块可恢复快照、原子逐局结果、配对 Bootstrap 和预设通过门槛；评测只读既有模型。
- 新增 Hybrid B12 独立失败机制研究：13 个死亡 seed 与 3 个触顶对照，逐局确定性重放、每 5000 块可恢复 Core/RNG 快照、最多每局 4 个同状态双动作反事实；不训练或调整策略。

- PPO 30M 独立 32-seed、100K 确认评测完成：normal_994303 预冻结门槛未通过；完整配对结果见 reports/experiments/ppo30m_independent_confirmation.md。

- Hybrid B12 独立失败机制研究完成：13 个死亡、3 个触顶对照；有害拦截证据 3 个 seed，有益拦截证据 1 个 seed。详见 reports/experiments/hybrid_b12_death_study.md。

- Hybrid B12 机制研究重放结果增加逐文件 SHA256 校验记录；已有 16 份重放结果完成校验并补齐 seal。

## 2026-10-19

- 新增 Hybrid B12 V3-A 定向拦截规则及分阶段评测队列：保留 B12 触发，审查 BC 落点行转换与改变 Hold 后的安全下一步落点；冻结新开发/确认种子和门槛，失败自动停止，不训练模型。

- Hybrid B12 V3-A 分阶段评测完成：stopped_after_20k，独立确认未通过或未启动；未修改 PPO/BC 权重、Core 或网页默认 AI。

- 2026-10-19：新增传统 V2 Hold 对 Hybrid B12 危险状态的探索性反事实研究。复用已核验快照，按 Action ID 去重并在 tmux 后台执行，结果见 `reports/experiments/hybrid_b12_v2_expert_study.md`。

- 2026-10-19：新增冻结的 Hybrid B12 + V2 Hold 单步救援候选和分阶段、可恢复评测队列。仅在公开棋盘高度与凹凸度达标、V2 合法独立动作不立即 Lock Out 时接管当前一步；后续由原 B12 决策。
