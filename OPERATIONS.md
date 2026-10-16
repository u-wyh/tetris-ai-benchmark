# 操作流水账

这里记录对训练、评测和项目代码产生实际影响的操作。时间为北京时间（UTC+08:00）。代码提交以 Git 为准，训练步数和阶段以 `runs/` 中的事务及队列文件为准。补录项只写可核实的事实；普通只读查询和每条 shell 命令不逐项记录。

## 2026-10-14

- 21:20–21:23｜V2 20 个种子的 556 条新反事实全部完成并通过不可变清单与压缩数据逐条哈希复核：A 22、B 534；训练/验证新增 A 19/3，合并 V1 后 37/6。V2 通过率 3.96%，低于 V1 的 9.42%；429 条新分支不在原 PPO 死亡前 1000 块内或原局触顶，均无 A。补齐 V2 B 类首因及 Hold 实际方块状态差异，判断数据不足，未启动新模型训练。依据：`active_v2/dataset_manifest.json`、`dataset_rows.json`、`readiness.json`、`reports/experiments/local_correction_counterfactual_v2.md`。
- 21:20｜局部纠错主动反事实 V2 队列完成：新分支 556，新增训练/验证 A 19/3，就绪判断 `insufficient_quality_data`；未自动训练。依据：`runs/ppo30m_local_correction/active_v2/status.json`、`dataset_manifest.json`、`readiness.json` 及三份 V2 报告。
- 20:59–21:01｜完成仅在内存中的固定批次 KL 系数敏感性诊断：同一初始 PPO、24 次更新、系数 0/1/3/10 时训练 A Top-1 都为 4/18，验证 A 均 0/3；正常状态对 PPO 一致率分别为 91.67%/91.89%/92.29%/93.55%。未保存模型或做游戏评测，结果仅说明更强 KL 可略减短程漂移。依据：`runs/ppo30m_local_correction/active_v2/loss_sensitivity.json`。
- 20:38–20:42｜新增 V2 独立收尾 tmux 程序；阶段性诊断报告 Git `643a043`、收尾程序 Git `3baff89`、梯度复核 Git `4ce15f4` 已推送。补充数据集物化步骤，待分支全部完成后生成带 Episode ID/逐条哈希的压缩分片；当前队列仍在运行，尚未把中间进度当作最终通过率。依据：`active_v2/status.json`、tmux 会话、`training/evaluation/local_correction_v2_materialize.py`。
- 20:40｜V1 epoch 3 的 16 个固定 8A+8C batch 梯度复核完成：16/16 梯度余弦为负，余弦中位数 -0.553，C/A 范数比中位数 14.5%；不作任何优化器更新，结果写入 `active_v2/gradient_sweep.json`。这是有限 batch 诊断，不能外推为全数据梯度。依据：`runs/ppo30m_local_correction/active_v2/gradient_sweep.json`。
- 20:33–20:38｜V2 概率字段修订 Git `189604e` 已推送并恢复独立 tmux 队列。完成 21 条 V1 A 类的一步公开 Hold/Next 审计：12 条 Hold 标志不同中仅 3 条使后续 Current/Hold/Next 方块状态实际不同；以邻列差作为深井粗代理时，BC 动作在 Hold 与非 Hold 两组均使井变浅，尚无长期因果结论。新增三份阶段性报告生成器及 3 个固定案例；Python 定向 12 项、Parity 6 项、JS 61 项通过。依据：`active_v2/hold_audit.json`、`reports/experiments/local_correction_*.md`、tmux `tetris-local-correction-v2`。
- 20:30–20:32｜启动 V2 独立 tmux 反事实队列后，在正式扩充前发现原子分支记录尚缺 PPO 对原动作的概率字段；停止队列并归档旧日志、状态及 5 条 smoke 分支到 `active_v2/pre_probability_field/`。为新分支增加合法掩码下的 PPO 动作概率与恢复校验，重新验证 `730015` 的 5 条分支。旧归档不计入 V2 新数据集，原 V1 保持不变。依据：`active_v2/pre_probability_field/`、`branches/seed_730015/`。
- 20:28–20:29｜完成 V1 训练机制诊断：18 条训练 A 类在仅用纠错损失的隔离实验中经 80 次更新达到 18/18 Top-1；正式 epoch 3 为 5/18，独立样本 0/3 但平均专家概率上升。梯度存在且 Actor 更新，参考 PPO 冻结。主动采样首版 20 块间距在早死 seed `730008` 不足 25 条，未执行该选择的反事实；选择文件归档 `active_v2/invalid_selection_spacing20/`。修订为每局最多 30 条、间距 10 块，20 局共冻结 556 条，其中随机配额 108 条。`730015` 的 5 条分支 smoke 及断点复用校验通过；定向测试 11 项通过。依据：`runs/ppo30m_local_correction/active_v2/diagnosis.json`、`selection/`、`branches/seed_730015/`、`tests/test_local_correction_active_v2.py`。
- 19:47–19:50｜补齐局部纠错失败报告的训练诊断、四消门槛解释、共同存活区间得分及 32 个开发种子逐局配对；重新生成报告。定向 Python 15 项、Parity 6 项和 JS 61 项回归通过。依据：`reports/experiments/ppo30m_local_correction_*.md`、`tests/test_local_correction.py`、`tests/parity/` 及 Node test 输出。
- 19:46｜V1 三个 epoch 的 32-seed、20K 开发评测全部完成；第 1、2 轮 P10 未过门槛，第 3 轮四消占比 87.65% 对 PPO 92.84%，降幅 5.19 个百分点、略超 5 点门槛。队列按协议停止为 `v1_failed_short_validation`，没有启动 V2、长局或确认种子；收尾程序提交并推送 Git `74e004a`。依据：`runs/ppo30m_local_correction/status.json`、`publish_status.json`、`evaluations/development_20k/*/summary.json`。
- 19:46｜PPO 30M 局部纠错队列结束，结果状态为 `v1_failed_short_validation`；数据、训练及比较报告已生成。依据：`runs/ppo30m_local_correction/status.json`、`reports/experiments/ppo30m_local_correction_comparison.md`。
- 19:32｜推送 P10 筛选门槛修订 Git `d3bfe0a`，恢复 `tetris-local-correction` 独立队列；启动后校验既有 20 局轨迹和反事实标签，进入已保存的 V1 三轮训练恢复流程。开发评测将复用合规已完成对局，当前尚无新的完整候选结果。依据：`runs/ppo30m_local_correction/status.json`、`queue.log`、tmux 会话。
- 17:25｜首个 Local-BC V1 epoch 在 32 个 20K 开发种子上平均存活 17,221、P10 4,977、平均分 5.26 亿，对照 PPO 分别为 18,406、18,119、5.73 亿；因原继续门槛漏掉严重低分位退步，在仅完成 V1 epoch 2 的 3 局时中断队列，增加 P10 至少保留 PPO 的 80% 的开发筛选，协议升 v6。已完成的合规开发对局保存复用；旧协议/日志归档 `pre_p10_gate_v5/`，15 项定向测试通过。未使用最终确认种子。依据：`evaluations/development_20k/`、`status.json`、`tests/test_local_correction.py`。尚未恢复队列。
- 16:56｜推送 RNG 恢复修复 Git `1c92592`，将失败时状态、日志与收尾状态归档 `failed_rng_resume_v5/`；重新启动 `tetris-local-correction` 与收尾发布程序，继续 v5 已保存的开发对照。依据：`runs/ppo30m_local_correction/status.json`、tmux 会话。
- 16:55｜v5 恢复时 PyTorch 将已保存的 CPU RNG 字节张量映射到 CUDA，`set_rng_state` 拒绝 GPU 张量，后台队列按预期失败并停止；已把恢复输入转回 CPU，直接从 epoch 3 checkpoint 恢复验证通过，15 项定向测试通过。已完成数据、模型和 9 局 PPO 开发对照均保留，尚未恢复评测。依据：`runs/ppo30m_local_correction/status.json`、`training/V1/history.json`、`tests/test_local_correction.py`。
- 16:52｜PPO 30M 局部纠错后台队列失败；保留已提交的局与 checkpoint，未把不完整结果作为结论。依据：`runs/ppo30m_local_correction/status.json`、`queue.log`。
- 16:52｜推送四消保留门槛修订 Git `b17d119` 并恢复后台队列；v5 直接复用 20 局合规轨迹、全部标签、三轮训练 checkpoint 与已完成的纯 PPO 开发对局。依据：`runs/ppo30m_local_correction/status.json`、`protocol.json`、tmux 会话。
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

- 2026-10-14 23:27–23:28 CST｜PPO30M 夜间微调 Raw/Risk 各完成首个 4096 样本事务试运行，原 PPO30M 作为初始化和冻结 KL 参考，单局上限 100000；两组约 500 样本/秒，GPU 峰值分配约 149 MiB。依据：`runs/ppo30m_overnight_finetune/{raw,risk}/committed/task_000001/metrics.json` 与各 `events.log`。
- 2026-10-15 10:08 CST｜Raw/Risk 均从 Task 1 检查点恢复并提交 Task 2，累计各 8192 样本；约 508 样本/秒，GPU 峰值分配约 151 MiB。依据：两组 `transaction_state.json`、`committed/task_000002/metrics.json` 与 `events.log`。
- 2026-10-15 10:23 CST｜队列启动命令误以普通进程运行，Raw 已提交 Task 3（12288 样本），Task 4 尚未提交；已停止该普通进程，保留 Task 3 提交指针，后续从此恢复至独立 tmux。依据：`runs/ppo30m_overnight_finetune/raw/transaction_state.json`、`events.log` 和 `runs/ppo30m_overnight_finetune/queue_status.json`。
- 2026-10-15 10:24 CST｜独立 tmux 会话 `ppo30m-overnight-finetune` 已启动；恢复时自动废弃未提交的 Task 4，再次完成并原子提交到 Raw 16384 样本，Risk 保持 8192 样本待顺序执行。全量 Python 169 测试及 35 子测试、Parity 6、JS 61 通过。依据：`tmux list-sessions`、Raw `events.log`、`transaction_state.json` 和 `runs/ppo30m_overnight_finetune/queue.log`。

- 2026-10-14 22:48–22:50 CST｜冻结 PPO30M 多候选反事实试点协议与 46 个训练组状态，校验原模型和 V2 快照；启动独立 `tmux` 队列 `ppo-multiaction-pilot`，单进程逐分支原子保存。首 5 个状态已完成；Python 相关测试 16 通过，Parity 6 通过，JS 61 通过。依据：`runs/ppo30m_multiaction_counterfactual/protocol.json`、`selection.json`、`queue.log`，`tests/test_ppo_multiaction_counterfactual.py`。
- 2026-10-14 23:15–23:16 CST｜多候选反事实队列完成 46/46 状态、368/368 原子分支，无失败；同标准 A 类动作 28 个，覆盖 9 状态，BC 对照覆盖 4 状态。生成报告及全部结果 SHA256 清单；未达到预设第二批计算效率门槛，停止扩大且不训练。Python 全量 162 测试和 35 子测试、Parity 6、JS 61 通过，新增定向测试 4 通过。依据：`runs/ppo30m_multiaction_counterfactual/completed.json`、`summary.json`、`dataset_manifest.json`、`reports/experiments/ppo_multiaction_counterfactual_results.md`。
- 2026-10-14 23:17 CST｜本次研究代码、测试和三份报告已提交并推送 `origin/main`，代码提交 `8f33b00`；后台试点队列已正常结束。依据：`git push origin main` 成功输出、`git log -1` 和 `tmux list-sessions`。

- 重要代码修改：记录做了什么、测试结果和 commit。
- 训练或评测操作：记录启动、暂停、恢复、失败、完成及对应的 run／checkpoint。
- 性能或模型结论：注明实测范围和结果文件；估算必须标成估算。
- 失败和中断也记录，不改写旧条目来掩盖历史。

- 2026-10-15 13:08–13:11 CST｜启动 PPO30M 辅助风险模型训练；首次启动因缺少 scikit-learn 失败，改用本地指标实现后成功运行 CUDA 训练 5 epoch，最佳 epoch 1。复用 16 训练局、4 验证局的冻结 PPO 已执行动作轨迹；最佳模型 SHA256 `287ab77df6977a752cf5bca15e80d025a0715159cb2bceda805f5f8f7823c793`。留出局 1000 块风险 PR-AUC 0.147，低于高度加洞数 0.274；按预设门槛标记 `completed_no_go`，停止 Top-5 重排序及在线评测。依据：`runs/ppo30m_risk_predictor/{protocol.json,data_manifest.json,history.json,heldout_evaluation.json,status.json}` 与三份实验报告。
- 2026-10-15 13:08–13:18 CST｜风险模型定向测试 5 通过；Python 全量回归 172 通过、35 subtests 通过；Parity 6 通过；JS 61 通过。依据：本次 pytest、unittest、node --test 命令输出。
- 2026-10-15 13:26 CST｜冻结 PPO30M Top-K 前瞻 v1 协议，模型哈希核验成功、990000/990100/990200 三组种子无已发现冲突；定向公开模拟与隐藏信息不变性测试 3 通过。启动独立 tmux `tetris-ppo-lookahead`，`tmux has-session` 成功，pane PID `1108068` 是 `.venv/bin/python -m training.evaluation.ppo_lookahead_queue`。先 smoke，之后按冻结门槛自动推进。依据：`runs/ppo30m_lookahead/protocol.json`、`queue.log`、`status.json`、`tests/test_ppo_lookahead.py`。
- 2026-10-15 13:28 CST｜smoke 2×3 局已原子完成，S2 触发率 1.5%；开发首局未完成时主动中断队列，准备将独立 seed 并行化以缩短后台耗时，未修改搜索规则或冻结参数。中断记录为 `KeyboardInterrupt`，已完成 smoke 结果保留且恢复将校验跳过。依据：`runs/ppo30m_lookahead/queue.log`、`status.json` 与 smoke 各局 JSON。
- 2026-10-15 13:29 CST｜将独立开发/确认 seed 改为最多 4 个 spawn worker 并行，每 worker 各自加载冻结模型；搜索版本与评分参数未变。恢复 tmux `tetris-ppo-lookahead`，`has-session` 成功，pane PID `1110771`，核实 4 个子进程；smoke 合规结果未重跑，状态更新为开发组已恢复。依据：`runs/ppo30m_lookahead/status.json`、`queue.log`、进程列表。
- 2026-10-15 13:31–13:32 CST｜完成 4 局纯 PPO 开发对照、候选开发对局尚为 0 时暂停并终止 tmux 进程组，补入原任务要求的共同存活分数 ≥95% 门槛及 PPO 在 20K 至少 12/16 触顶时的同组 50K 延长验证；搜索与候选评分未改。完成的纯 PPO 对局仍按原协议校验复用。冻结补充协议见 `runs/ppo30m_lookahead/protocol_amendment.json`；暂停状态和原子对局文件见同目录。
- 2026-10-15 13:32 CST｜恢复 tmux `tetris-ppo-lookahead`，`has-session` 成功，pane PID `1112576`，核实 4 个 spawn 子进程；状态已恢复为 `development_20k`。依据：`runs/ppo30m_lookahead/status.json`、tmux pane 和进程列表。
- 2026-10-15 13:37 CST｜前瞻搜索阶段代码与报告框架已提交并推送 `origin/main`，提交 `2e9a353`；回归测试 Python 177 通过、35 subtests 通过，Parity 6 通过，JS 61 通过。后台开发队列仍运行。依据：`runs/ppo30m_lookahead/{python_tests.log,parity_tests.log,js_tests.log,queue.log,status.json}` 和 Git push 成功输出。
- 2026-10-15 13:39 CST｜纯 PPO 16 局 20K 对照完成、S1/S2 开发局尚为 0 时暂停队列，冻结 50K 延长前的质量筛选，防止两个明显失败方案仅因 PPO 触顶多而耗费更长评测。先前 16 局纯 PPO 结果保留，搜索规则未变。依据：`runs/ppo30m_lookahead/development_20k/ppo/summary.json`、`protocol_amendment_2.json`、`status.json`。
- 2026-10-15 13:39 CST｜质量筛选定向测试 4 通过；恢复 tmux `tetris-ppo-lookahead`，`has-session` 成功，pane PID `1116475`，核实 4 个 spawn 子进程；纯 PPO 16 局合规文件跳过。依据：`runs/ppo30m_lookahead/status.json`、进程列表及 `tests/test_ppo_lookahead.py`。
- 2026-10-15 13:59–14:00 CST｜开发 20K 的 PPO、B12、S1、S2 各 16 局完成，S1/S2 均 15/16 触顶、平均存活 18931.75；共同存活分数比 PPO 约 0.999，四消约 92.9%/92.8%。两者在 990109 均新增 2908 块早死，S2 未救回。纯 PPO 13/16 触顶且 S1 符合冻结 50K 质量筛选，队列自动进入同组 50K。依据：`runs/ppo30m_lookahead/development_20k/*/summary.json`、逐 seed JSON、`queue.log`。
- 2026-10-15 14:32–14:39 CST｜同组 50K 的 PPO/B12 各 16 局完成，S1 9 局、S2 0 局时诊断 seed 990109 新增早死。S1 轨迹 24 次替换、22 次改变 Hold、24 次当步四消；第 2849 块同 Core 快照和 RNG 的双分支，仅改首动作后继续 S1：搜索动作 59 块 Game Over，原 PPO 动作至少存活 1000 块（右截断）。据此主动终止 tmux 进程组，`has-session` 不存在，原子保存已完成局，状态 `completed_no_go_counterfactual`；未启动 100K 确认。依据：`runs/ppo30m_lookahead/diagnostics/{S1_seed_990109.json,S1_branch_seed_990109_step_2849.json}`、`status.json`、逐局结果和最终报告。
- 2026-10-15 14:41 CST｜生成最终比较报告并补充可复现失败诊断脚本；定向 lookahead 测试 5 通过，诊断与报告模块编译通过，确认 tmux 队列已结束。依据：`reports/experiments/ppo30m_lookahead_comparison.md`、`training/evaluation/ppo_lookahead_failure.py`、本次终端测试输出。
- 2026-10-15 14:49 CST｜启动独立 tmux `tetris-lookahead-v3`，pane PID 1132001，执行冻结 S1 的 seed 990109 全部替换事件逐项评分审计；原 S1/S2 和模型未修改。依据：`training/evaluation/ppo_lookahead_v3_audit.py`、`runs/ppo30m_lookahead_v3_audit.log`、tmux pane 查询。
- 2026-10-15 14:50–14:53 CST｜tmux `tetris-lookahead-v3`，pane PID 1132316，完成 seed 990109 第 2849 块同快照消融：去四消奖励、Hold 门槛和提高 margin 均选择原 PPO 动作；原 S1 仍选择 #249 且 59 块后死亡。依据：`runs/ppo30m_lookahead_v3/diagnostics/snapshot_ablations_seed_990109_step_2849.json`。之后误将 V3 队列 `--help` 作为前台执行，smoke 各 2 局完成，开发局尚无完成；14:53 中断前台以改为 tmux，状态记录 `KeyboardInterrupt`，会在 tmux 中恢复。依据：`runs/ppo30m_lookahead_v3/status.json` 和 smoke 文件。
- 2026-10-15 14:54 CST｜把误执行的前台 smoke 结果移至 `diagnostics/foreground_smoke_discarded/`，不会用于开发比较；在独立 tmux `tetris-lookahead-v3` 启动先 Python/Parity/JS 回归再重新 smoke 与可恢复评测的队列，pane PID 1133465、pytest 子进程 PID 1133472，确认后台存在。依据：`training/evaluation/run_ppo_lookahead_v3.sh`、`runs/ppo30m_lookahead_v3/*_tests.log`、tmux/ps 查询。
- 2026-10-15 15:03–15:04 CST｜后台 Python 182 项及 35 子测试、Parity 6 项、JS 61 项全部通过；新增快照恢复及已知异常早死动作回归的定向 V3 测试 5 项通过。tmux 重跑的四方案 smoke 各 2 局完成，队列主进程 PID 1138774，准备进入开发组。依据：`runs/ppo30m_lookahead_v3/{python_tests.log,parity_tests.log,js_tests.log,queue.log,status.json}`、`tests/test_ppo_lookahead_v3.py`。
- 2026-10-15 15:27 CST｜开发组 V3-A 已发现 seed 991104 于 2097 块 Game Over、纯 PPO 同 seed 达 20000；该局仅两次替换，均改变 Hold 且当步四消。另 seed 991105 于 11357 块结束。保持冻结规则与配对队列不变，启动独立 tmux `tetris-lookahead-v3-diagnose`（pane PID 1143839）对 991104 第 1592 块做同 Core 快照、固定 PPO/V3-A 续控分支；证据将保存至 `runs/ppo30m_lookahead_v3/diagnostics/new_failure_V3A_seed_991104_step_1592.json`。
- 2026-10-15 15:28 CST｜新失败分支完成：seed 991104 第 1592 块首选 V3-A #249，两种固定续控均 505 块后死亡；首选 PPO #1052，两种续控均至少存活 1000 块（右截断）。诊断 tmux 正常结束，满足 V3-A 预设严重回归停止门槛。依据：`runs/ppo30m_lookahead_v3/diagnostics/new_failure_V3A_seed_991104_step_1592.json`、`new_failure_991104.log`。
- 2026-10-15 15:37 CST｜新开发 20K 完整配对：PPO/B12/S1/V3A/V3B 各 16 局，平均存活分别 18666/20000/18825/18341/18904；V3-A 严重早死门槛失败，V3-B 安全门槛通过且 PPO 14/16 触顶，队列依冻结规则进入同组 50K，仅扩展 PPO、B12、V3-B。依据：`runs/ppo30m_lookahead_v3/development_20k/*/summary.json`、`development_gate.json`、`status.json` 和对照报告。
- 2026-10-15 16:22 CST｜同组 50K 的 PPO、B12、V3-B 各 16 局已原子保存；汇总阶段因仍索引已淘汰的 V3-A 而失败，队列记录 `KeyError('V3A')` 并停止，未启动 100K。依据：`runs/ppo30m_lookahead_v3/development_50k/*/summary.json`、`queue.log`、`status.json`。
- 2026-10-15 22:09–22:11 CST｜修复 50K 质量汇总只检查参评候选，定向测试 6 通过；在 tmux `tetris-lookahead-v3`（pane PID 1215107）恢复队列，全部 50K 合规对局跳过，成功写入 `completed_no_go_extension`，无 100K 对局。V3-B 的 50K 平均存活/P10 为 41545/22682，低于 PPO 的 44471/29534；死亡前最后 20 次替换摘要已从原子结果提取（共 20 个 Game Over 文件），完整报告重新生成。依据：`runs/ppo30m_lookahead_v3/{extension_gate.json,status.json,diagnostics/deaths/}`、`reports/experiments/ppo30m_lookahead_v3_comparison.md`、定向测试输出。
- 2026-10-15 22:34–22:38 CST｜PPO30M 死亡机制研究先审计历史数据：20 条局部纠错、32 条 Hybrid PPO 对照、16 条 V3 PPO 对照，共 68 条历史记录（37 条死亡）；其动作/快照粒度不足以系统重建全部死亡前动态，均标记旧训练、验证或开发用途。新增独立采集、Core/RNG 快照重放校验与研究设计；24 块临时目录烟测 1 通过，确认原子轨迹、哈希恢复和截断区分。尚未启动正式长局。依据：`runs/ppo30m_death_analysis/historical_inventory.json`、`reports/experiments/ppo30m_death_mechanism_design.md`、`tests/test_ppo_death_analysis.py`。
- 2026-10-15 22:38 CST｜已提交并推送采集设计 `b382a44`，冻结 `runs/ppo30m_death_analysis/protocol.json`（首批 992000–992031、100K、4 worker）。独立 tmux `tetris-ppo-death-analysis` 启动正式 PPO 长局，pane PID 1227892、主进程 1227893、worker PID 1227920/1227923/1227924/1227925，`has-session` 成功，未启动训练。依据：`runs/ppo30m_death_analysis/{protocol.json,status.json,capture.log}` 与 tmux/ps 查询。
- 2026-10-15 22:47 CST｜实现死亡窗口、匹配对照、分层合法候选、同 Core/RNG 单动作分支、有限窗口报告及捕获后自动续跑队列；修复捕获协议恢复时 Git commit 漂移，定向测试 4 通过，代码编译通过。采集仍在独立 tmux 中运行，未启动新训练。依据：`training/evaluation/ppo_death_{analysis,counterfactual,report,pipeline}.py`、`tests/test_ppo_death_analysis.py` 和本次终端测试输出。
- 2026-10-15 22:47 CST｜提交并推送自主队列代码 `9ea6df0`；独立 tmux `tetris-ppo-death-analysis-pipeline` 启动成功（pane PID 1234063、主进程 1234068），正在等待采集完成后自动执行重放、分析、反事实、报告、回归和推送。采集状态为首批 5/32 局，原 tmux 未中断。依据：`runs/ppo30m_death_analysis/{pipeline_status.json,pipeline.log,status.json}` 和 tmux pane 查询。
- 2026-10-15 23:04 CST｜实现独立危险状态课程 PPO 环境、冻结快照数据构建、A/B 事务训练与顺序评测队列；课程环境 Core/RNG、合法 Mask、相对窗口截断和 VecEnv TimeLimit 测试通过。死亡分析采集仍在运行（首批 19/32），夜间队列计划在其完成后启动，暂未开始新训练。依据：`training/env/failure_curriculum_env.py`、`training/failure_curriculum_queue.py`、`tests/test_failure_curriculum.py`、`runs/ppo30m_death_analysis/status.json`。
- 2026-10-15 23:05 CST｜夜间课程队列代码提交并推送 `9525981`；独立 tmux `tetris-failure-curriculum` 启动成功（pane PID 1243178、主进程 1243179），状态 `waiting_for_death_analysis`，没有与现行 4-worker 采集抢占训练资源。依据：`runs/ppo30m_failure_curriculum/{status.json,queue.log}`、tmux 和进程查询。

- 2026-10-15T23:25:10+08:00｜死亡分析 32 局完成逐 seed 确定性重放校验。依据：runs/ppo30m_death_analysis/replay_validation/。

- 2026-10-15T23:26:11+08:00｜死亡分析后台队列停止：ModuleNotFoundError("No module named 'matplotlib'")。依据：runs/ppo30m_death_analysis/pipeline_status.json。

- 2026-10-15T23:26:13+08:00｜危险课程后台队列失败并停止依赖阶段：RuntimeError('Death-analysis queue failed; curriculum source is unvalidated')。依据：runs/ppo30m_failure_curriculum/status.json。
- 2026-10-16 09:31 CST｜修复死亡分析图表依赖缺失：改用独立 SVG 图并从已验证轨迹生成代表性图，未更动模型或已保存的长局；首批 32 局已完成，19 局真正死亡、13 局 100K 右截断。准备恢复死亡分析和依赖队列。依据：`runs/ppo30m_death_analysis/{capture_completed.json,death_analysis.json,figures/death_seed_992012.svg}`、`training/evaluation/ppo_death_analysis.py`。

- 2026-10-16T09:32:27+08:00｜死亡分析 32 局完成逐 seed 确定性重放校验。依据：runs/ppo30m_death_analysis/replay_validation/。
- 2026-10-16 09:32 CST｜死亡分析 tmux 队列从已验证 32 局恢复（pane PID 1344065、主进程 1344067），危险课程 tmux 同步恢复等待依赖（pane PID 1344130、主进程 1344133）；同时修复课程自定义死亡奖励版本在事务恢复预设核验中的不一致，避免 smoke 后续任务无法恢复。依据：`runs/ppo30m_death_analysis/pipeline_status.json`、`runs/ppo30m_failure_curriculum/status.json`、`training/train_vector_transaction.py`。
