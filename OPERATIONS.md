# 操作流水账

这里记录对训练、评测和项目代码产生实际影响的操作。时间为北京时间（UTC+08:00）。代码提交以 Git 为准，训练步数和阶段以 `runs/` 中的事务及队列文件为准。补录项只写可核实的事实；普通只读查询和每条 shell 命令不逐项记录。

## 2026-10-14

- 16:51｜Local-BC V1 已完成三轮训练，开始开发评测后在新模型对局前停止队列，将四消保留门槛从相对 85% 收紧为短局至多下降 5 个百分点、正式确认前至多下降 3 个百分点；当时只完成 9 局纯 PPO 开发对照，均原子保存复用。协议升为 v5，旧协议和日志归档 `pre_gate_v4/`，14 项定向测试通过。依据：`runs/ppo30m_local_correction/evaluations/development_20k/ppo/`、新 `protocol.json`、`tests/test_local_correction.py`。尚未恢复队列。
- 16:49｜Local-BC V1 三个短 epoch 监督训练完成，保存每轮模型、优化器和 RNG；训练集 A 类动作命中从 0/18 到 5/18，独立样本验证仍为 0/3，正常状态对原 PPO 一致率从 100% 降到约 92.1%（验证 masked KL 约 0.0476）。队列已开始 32-seed、20K 开发评测；当前不能判定长局效果。依据：`runs/ppo30m_local_correction/training/V1/history.json`、`status.json`。
- 16:47｜局部纠错 v4 20 局反事实筛选完成；训练 A/C 样本 18/11,448，独立样本验证 A/C 为 3/1,752。全部 20 局标签按源分片 SHA256 保存，已进入小规模 Actor 监督训练；未启动 PPO 续训。依据：`runs/ppo30m_local_correction/dataset_summary.json`、`labels/`、`status.json`。
- 16:22｜已推送 v4 纠错证据规则 Git `ef000c6` 并恢复 tmux 队列；20 局已校验轨迹及 v3 前 4 局反事实结果复用，未重复已合规的 Core 对局或分支。依据：`runs/ppo30m_local_correction/status.json`、`initial_trigger_v3/`、tmux 会话。
- 16:21｜前 4 个训练种子的 v3 分支共 48 个，发现 4 个满足“PPO 单步增高至少 3 格、BC 明显更安全、同状态分支多活至少 250 块且保留共同得分/四消”的样本，但原高位触发均未接纳；停止队列，将 4 份原分支证据与日志归档至 `initial_trigger_v3/`。在未查看样本验证种子和开发评测前冻结 v4 快速增高触发，复用旧分支结果重新分级，`730001` 从 0 变成 3 条 A 类；13 项定向测试通过。依据：v3 归档标签、v4 `labels/seed_730001.json`、`tests/test_local_correction.py`。尚未恢复队列。
- 16:09｜推送分支目标修订 Git `fad9057` 并恢复 `tetris-local-correction`；v3 仅复用校验通过的 20 局 PPO 轨迹，重放目标状态后重新做反事实筛选。旧 v2 标签不参与训练。依据：`runs/ppo30m_local_correction/status.json`、`invalid_branch_selection_v2/`、tmux 会话。
- 16:07｜分支筛选前 4 局 48 个目标仅有 4 个 PPO 分支在 1000 块观察窗内死亡、A 类 0；据此停止 v2 分支队列并归档 4 份标签与协议，保留 20 局已合规采集轨迹。将离线目标选择改为优先 PPO 真实死亡前约 500 块的风险状态，重放时校验公开状态哈希和 PPO 动作；协议升至 v3，新增选择与重放测试，6 项定向测试通过。未使用开发或确认种子调规则。依据：`runs/ppo30m_local_correction/invalid_branch_selection_v2/`、`tests/test_local_correction.py`、新 `protocol.json`。
- 15:44｜推送采样器修复 Git `a9e7390`，按 v2 协议重新启动 `tetris-local-correction`；归档的 3 局 v1 数据不参与新结果。依据：`runs/ppo30m_local_correction/status.json`、tmux 会话、`invalid_sampler_v1/`。
- 15:43｜发现采样池未满时错误替换样本，3 局已完成的采集分片不符合目标抽样协议；停止 `tetris-local-correction`，将旧协议、日志和 3 局分片原样移至 `runs/ppo30m_local_correction/invalid_sampler_v1/` 保存证据。修复后协议升至 v2，新增未满容量时全部保留的测试并通过 4 项定向测试；尚未恢复队列。依据：归档目录、`tests/test_local_correction.py`、新 `protocol.json`。
- 15:37｜已推送局部纠错流水线实现（Git `e238e05`），启动 `tetris-local-correction` tmux 可恢复队列，当前仅采集冻结 PPO 30M 自主轨迹；后续反事实样本不足或 V1 评测不达门槛时自动停止，不启动大规模 PPO。依据：`runs/ppo30m_local_correction/status.json`、`queue.log`、tmux 会话。
- 15:36｜实现独立 PPO 30M 局部纠错数据、反事实筛选、Actor 监督训练和可恢复分阶段评测；冻结模型 SHA、版本与独立种子组校验通过。定向 Python 10 项、Parity 21 项及 35 子测试、JS 语法检查通过；额外 Python 全量测试运行 7 分 16 秒已有 136 项通过，无失败，但因包含耗时训练事务测试而主动中断。CUDA 16 样本反向传播烟雾测试通过，峰值约 24.1 MiB。依据：`tests/test_local_correction.py`、`reports/experiments/ppo30m_local_correction_dataset.md`、本次测试输出。尚未启动正式数据采集或训练。
- 14:34｜Hybrid V2 独立 32-seed 确认评测完成，最终比较报告已生成并提交推送。依据：`runs/hybrid_v2/status.json`、`reports/experiments/hybrid_v2_comparison.md`。
- 10:09｜补充冻结 PPO/BC 模型经临时 Save/Load 后的确定性动作一致性测试，两模型均通过；新测试文件共 6 项通过，未修改原 checkpoint。依据：`tests/test_hybrid_v2.py`、本次 pytest 输出。
- 10:07｜提交并推送 B12 失败分析、冻结 C1/C2 设计及独立 V2 评测实现（Git `1ee5800`）；启动 `tetris-hybrid-v2` 可恢复后台队列和独立收尾推送程序。烟雾阶段已开始；不会启动任何 PPO、BC、DAgger 或蒸馏训练。依据：`runs/hybrid_v2/status.json`、tmux 会话、Git `1ee5800`。
- 10:06｜冻结 C1/C2 设计与独立种子协议，新增可恢复评测和最终报告／推送程序；Python 定向回归 30 项、Parity 6 项含 35 子测试、JS 语法检查通过。尚未启动新开发长局评测。依据：`reports/experiments/hybrid_v2_design.md`、`tests/test_hybrid_v2.py`、本次测试输出。
- 10:02｜六个 B12 失败局完成确定性重放与选定位置同状态双分支；生成诊断报告。六局 1131 次 BC 动作中，753 次 Hold 用法与 PPO 候选不同、77 次 BC 候选新增洞更多；关键单点结论及不确定性见报告。依据：`runs/hybrid_v2/failure_analysis/`、`reports/experiments/hybrid_b12_failure_analysis.md`。
- 09:52｜`700016` 的旧 B12 轨迹与终局重放一致。为缩短剩余五局离线反事实耗时，中断串行分析并改为最多 4 个 CPU 进程并行后恢复；已完成且字段齐全的局不重跑。依据：`runs/hybrid_v2/failure_analysis/seed_700016.json`、`failure.log`。
- 09:47｜启动独立 B12 失败局确定性重放与同 Core 状态反事实分析；`700026` 原局分数和存活重放一致。首次拦截位置改用 PPO 动作、之后同用 B12 时，5000 块观察窗内仍存活；原 BC 分支在第 1715 块结束。该单点结果只作诊断。依据：`runs/hybrid_v2/failure_analysis/seed_700026.json`。
- 09:48｜为补充每次拦截的消行／Hold 和末段棋盘指标，中断并重启尚在进行的失败分析后台进程；已保存结果会按字段校验后复用或重放，未动原始评测。依据：`runs/hybrid_v2/failure.log`、`training/evaluation/hybrid_failure.py`。
- 02:34｜Hybrid 调参及独立确认评测完成，最终报告已生成并准备提交推送。依据：`runs/candidate_ppo30m_bc_hybrid/status.json`、`reports/experiments/candidate_ppo30m_bc_hybrid.md`。
## 2026-10-13

- 23:54｜Hybrid 队列完成烟雾阶段并进入 16-seed 20K 调参；启动独立 `tetris-hybrid-publish` 收尾程序，待全部确认局完成后提交推送报告。依据：`runs/candidate_ppo30m_bc_hybrid/status.json`、`training/evaluation/hybrid_publish.py`。
- 23:46｜双策略队列入口检查意外进入等待 DAgger 的循环，立即中断；仅写入模型协议与等待状态，未加载模型或执行对局。依据：`runs/candidate_ppo30m_bc_hybrid/status.json`、本次终端中断记录。
- 23:51｜核验冻结 PPO 30M 与 BC checkpoint SHA256、观测／特征／动作版本；实现独立双策略 A/B 控制器与可恢复评测队列。4 项策略单测、环境／并行定向测试共 10 项通过，Parity 6 项和 35 个子测试通过；两模型 CPU 1000 块低负载试跑完成。启动 `tetris-hybrid` tmux 队列，先等待 DAgger 队列完成。依据：`tests/test_hybrid_policy.py`、`runs/candidate_ppo30m_bc_hybrid/`、本次测试输出。
- 23:52｜补齐逐局恢复校验、模型空间核验、配对报告和真实模型一致性测试后，停止并重启仍在等待 DAgger 的 `tetris-hybrid` 队列；停止前无 Hybrid 对局。更新后 11 项定向测试通过。依据：队列状态、`tests/test_hybrid_policy.py`、本次终端输出。
- 12:43–13:53｜建立并行的 BC／传统算法 10 万块配对测试，提交结果报告；10 个共同种子中 BC 与 V2 Hold 均 10/10 到达上限。依据：Git `db72647`–`87119b2`、`reports/experiments/candidate_bc_vs_traditional_100k.md`。
- 14:55–15:23｜提交全新种子泛化、KL-PPO、DAgger 和可恢复后台队列；补充恢复与数据版本测试，启动 `tetris-imitation-next`。依据：Git `f637691`–`1322a44`、队列状态。
- 17:27｜30 个全新种子、每局上限 10 万块的 BC／V2 Hold 对照完成：BC 30/30、V2 28/30 到达上限。依据：`runs/candidate_bc_next_stage/generalization/summary.json`。
- 17:28–18:03｜完成普通 PPO 退步分析、KL 烟雾测试和两组 beta 小试；按预定规则选择 beta=0.03。依据：`runs/candidate_bc_next_stage/stages/`、`runs/candidate_bc_next_stage/kl_tuning/selection.json`。
- 18:03｜启动独立的 KL-PPO 2M 正式微调。依据：`runs/candidate_bc_next_stage/queue_status.json`、`runs/candidate_bc_klppo_2m/events.log`。
- 20:23｜KL-PPO 保存第 367 个事务 checkpoint，累计 1,503,232 步，随后运行周期验证。依据：`runs/candidate_bc_klppo_2m/transaction_state.json`、`events.log`。
- 20:23 后｜应用户要求，短暂暂停队列、训练器和正在运行的验证进程；完成评测加速与一致性检查后恢复。暂停／恢复的精确时刻未自动写入运行日志，本项依据会话操作记录。
- 20:32｜提交并推送评测加速 `2fb5925`。同机同种子 1000 块交替测试：旧环境约 7.79 ms/块，新环境约 4.73 ms/块；19 个定向测试、4 个 Core 对照测试通过，另有两局各 2000 块逐步一致。依据：Git `2fb5925`、`reports/experiments/candidate_evaluation_speedup.md`。已启动的验证进程继续使用旧代码，之后新启动的评测使用加速环境。
- 20:36｜建立本操作流水账与 `AGENTS.md` 续写约定，补录可核实的重要节点。此时后台队列仍处于 `train_kl_ppo_2m` 阶段；未改变训练配置或进程。依据：Git 提交与 `runs/candidate_bc_next_stage/queue_status.json`。
- 20:57｜KL-PPO 正式微调提交第 489 个事务 checkpoint，累计 2,002,944 PPO 步；随后开始训练节点验证。依据：`runs/candidate_bc_klppo_2m/transaction_state.json`、`events.log`。
- 20:58｜把训练节点的独立种子验证改为最多 8 个 CPU 进程并行，保留输出种子顺序；为未来 10 万块长局加入一次性 4/6/8/10/12 进程吞吐量试跑并自动选最快值。串并行逐 seed 结果一致，相关 9 个定向测试通过。长局试跑将在长局阶段才执行，不与 PPO 训练争抢 CPU。依据：本次代码提交及 `tests/test_parallel_candidate_evaluation.py`。
- 20:59｜2M 节点的 16-seed、每局 5000 块周期验证以新并行实现完成，耗时 64.484 秒；此前同规格串行验证通常约 10 分钟。32-seed、每局 10000 块里程碑验证随后启动。依据：`runs/candidate_bc_klppo_2m/events.log`。这仅是同任务评测墙钟时间对比，不代表模型质量变化。
- 22:49｜后台队列进入 4 模型、共同 32-seed、每局上限 10 万块的长局验证。一次性吞吐试跑在 4/6/8/10/12 路中选出 12 路；BC 已完成 32 局，其中 31 局达上限、平均 99,320.5 块；纯 PPO 30M 已完成 32 局，其中 12 局达上限、平均 69,392.7 块。KL-PPO 2M 正在跑，DAgger 长局待跑；未据此宣称全部任务完成。依据：`runs/candidate_bc_next_stage/long_worker_tuning.json`、`long_validation_protocol.json` 及 `comparisons/long_100k_*/status.json`、`summary.json`。
- 22:57｜核查纯 PPO 30M 的较高得分：双方都跑满 10 万块的 12 个种子上，30M 平均约 158.5 亿，BC 约 83.4 亿；同种子 `100008` 前 3000 块追踪显示 30M 有 293 次四行消除，BC 为 0，双方总消行相近。依据：`reports/experiments/candidate_score_mechanism.md` 与原始逐局结果。

## 2026-10-12

- 10:30–10:31｜提交 V2 Hold 教师数据、BC 及独立 PPO 微调队列，补充专家数据采集速度记录。依据：Git `09ae0c1`、`54c0291`、`b01e347`，详细数据见 `reports/experiments/candidate_imitation_learning.md`。

## 记录约定

- 重要代码修改：记录做了什么、测试结果和 commit。
- 训练或评测操作：记录启动、暂停、恢复、失败、完成及对应的 run／checkpoint。
- 性能或模型结论：注明实测范围和结果文件；估算必须标成估算。
- 失败和中断也记录，不改写旧条目来掩盖历史。
