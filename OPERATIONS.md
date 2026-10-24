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

- 2026-10-16T09:33:30+08:00｜死亡窗口、对照和探索性分类完成：真实死亡 19 局，100K 右截断 13 局。依据：runs/ppo30m_death_analysis/death_analysis.json。

- 2026-10-16T09:37:21+08:00｜预先确定反事实状态 120 个、合法动作分支 541 个。依据：runs/ppo30m_death_analysis/counterfactual_prepared.json。

- 2026-10-16T10:08:41+08:00｜1000 块与有条件 5000 块反事实原子分支完成。依据：runs/ppo30m_death_analysis/counterfactual_{1000,5000}_completed.json。

- 2026-10-16T10:18:08+08:00｜死亡机制报告与 Python、Parity、JS 回归测试完成。依据：reports/experiments/ppo30m_death_*；runs/ppo30m_death_analysis/*_tests.log。

- 2026-10-16T10:24:28+08:00｜冻结危险课程数据：210 状态、21 独立死亡 seed。依据：runs/ppo30m_failure_curriculum/curriculum_manifest.json。

- 2026-10-16T10:34:07+08:00｜normal 完成 4096 个环境交互样本、1 个事务任务。依据：runs/ppo30m_failure_curriculum/normal/state.json。

- 2026-10-16T10:34:16+08:00｜curriculum 完成 4096 个环境交互样本、1 个事务任务。依据：runs/ppo30m_failure_curriculum/curriculum/state.json。

- 2026-10-16T10:34:16+08:00｜危险课程后台队列失败并停止依赖阶段：RuntimeError('Resume reward config differs from recorded run')。依据：runs/ppo30m_failure_curriculum/status.json。

- 2026-10-16T13:10:44+08:00｜危险课程后台队列失败并停止依赖阶段：CalledProcessError(9, ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])。依据：runs/ppo30m_failure_curriculum/status.json。
- 2026-10-16 13:10–13:13 CST｜死亡机制队列已完成并推送 `bb9ad0a`（19 死亡、120 反事实状态）。课程 A/B 各完成并校验 1 个 4096 样本 smoke task；后续恢复曾因旧进程加载修复前的奖励核验代码而停止。重启队列时发现当前内核 `6.8.0-147-generic` 没有 NVIDIA 模块（已安装的 595-open 模块仅覆盖 142/146），`nvidia-smi` 与 CUDA 不可用；正式训练未启动。修改资源检查为等待驱动恢复，不使用 CPU 退化训练。依据：`runs/ppo30m_death_analysis/pipeline_status.json`、`runs/ppo30m_failure_curriculum/{status.json,normal/committed/task_000001,curriculum/committed/task_000001}`、`modinfo nvidia`、`nvidia-smi`、本次终端检查。
- 2026-10-16 13:11 CST｜危险课程 tmux 队列重启（pane PID 18614、主进程 18615）并进入 `waiting_for_gpu_driver`；当前内核 6.8.0-147 没有 NVIDIA 内核模块，`nvidia-smi`/CUDA 不可用。已完成的正常组与课程组各 1 个事务 smoke task 均保留并校验，正式 500K 未开始；队列会等待驱动可用，不降级成 CPU 训练。依据：`runs/ppo30m_failure_curriculum/{status.json,queue.log,normal/transaction_state.json,curriculum/transaction_state.json}`、tmux pane 查询。
- 2026-10-16 13:25 CST｜完成 CUDA 故障审计：147 当前内核缺 NVIDIA 模块，APT 595.91.07 预编译模块只覆盖 142/146；147 包候选依赖 595.99.02，安装模拟涉及同分支多包升级；Secure Boot 已开但无模块签名拒绝证据。无免密 sudo，未安装或重启。课程数据集及 A/B 各 1 个已提交事务通过 SHA/manifest 校验，CUDA 仍不可用，后台继续等待。依据：`reports/experiments/ppo30m_cuda_recovery_diagnostic.md`、`runs/ppo30m_failure_curriculum/nvidia_cuda_diagnostic_2026-10-16.json`、本次 `nvidia-smi`/PyTorch/`apt-get -s` 输出。

- 2026-10-17T18:26:27+08:00｜冻结危险课程数据：210 状态、21 独立死亡 seed。依据：runs/ppo30m_failure_curriculum/curriculum_manifest.json。
- 2026-10-17 18:26 CST｜机器现运行旧内核 6.8.0-146-generic，NVIDIA 595.91.07 的 RTX 3050 在 `nvidia-smi` 正常、PyTorch CUDA 可用且 GPU 张量加法通过。复核 210 快照数据集、冻结 PPO SHA256、Normal/Curriculum 各 1 个 4096 样本事务 manifest 完整；确认无其他训练进程后，独立 tmux `tetris-failure-curriculum` 从已提交事务恢复（pane PID 268176、主进程 268181），先运行既定回归测试，未清空或重训已提交任务。依据：`runs/ppo30m_failure_curriculum/{status.json,queue.log,curriculum_manifest.json,normal/transaction_state.json,curriculum/transaction_state.json}`、本次 CUDA/事务校验输出。

- 2026-10-17T18:35:55+08:00｜normal 完成 4096 个环境交互样本、1 个事务任务。依据：runs/ppo30m_failure_curriculum/normal/state.json。

- 2026-10-17T18:35:55+08:00｜curriculum 完成 4096 个环境交互样本、1 个事务任务。依据：runs/ppo30m_failure_curriculum/curriculum/state.json。

- 2026-10-17T18:51:45+08:00｜normal 完成 503808 个环境交互样本、123 个事务任务。依据：runs/ppo30m_failure_curriculum/normal/state.json。

- 2026-10-17T19:07:46+08:00｜curriculum 完成 503808 个环境交互样本、123 个事务任务。依据：runs/ppo30m_failure_curriculum/curriculum/state.json。
- 2026-10-17 19:34:42 CST｜启动独立 `tetris-post-curriculum-3m` 队列；首次启动因日志目录尚不存在而退出，创建目录后重启并确认处于 `waiting_for_curriculum`，尚未进行高负载对局。定向测试 8 passed。依据：`runs/post_curriculum_3m/{status.json,protocol.json,queue.log}`、`training/evaluation/post_curriculum_3m.py`。
- 2026-10-17 19:35:54 CST｜按用户澄清停止尚未开局的旧 30×10 万队列，冻结为 4 个相同新种子、每局 300 万块上限，加入传统 V2 Hold 并重启独立 tmux 队列；当前仍等待课程实验完成。依据：`runs/post_curriculum_3m/{superseded_protocol.json,protocol.json,status.json}`；传统 V2 5 块烟测通过。
- 2026-10-17 19:39:45 CST｜按用户要求将尚未开局的 300 万块/局评测从每模型 4 个种子扩至 8 个，停止旧等待会话并重启同名 tmux，会话和新协议已验证，仍未进行高负载对局。依据：`runs/post_curriculum_3m/{protocol.json,status.json,superseded_protocol_4.json}`。
- 2026-10-17 19:41:24 CST｜修正长局评测排队方式：传统 V2 Hold 从 8 局串行改为 4 个独立种子进程并发，逐局原子保存；重启并验证独立 tmux 仍等待训练完成，定向测试 8 passed。依据：`training/evaluation/post_curriculum_3m.py`、`runs/post_curriculum_3m/status.json`。
- 2026-10-17 19:44:58 CST｜根据 8 个物理 CPU 核和现有 worker 约 760 MiB RSS，将训练结束后的独立长局评测配置为 8 个 CPU worker（课程实验当前 4-worker 评测保持原状）；重启并验证等待队列，定向测试 8 passed。依据：`runs/post_curriculum_3m/{protocol.json,status.json}`、`training/evaluation/failure_curriculum_eval.py`。
- 2026-10-17 20:00:12 CST｜优化训练结束后的 300 万块长局评测：纯神经网络动作使用已核对的确定性候选 argmax；传统 V2 Hold 跳过不参与排序的模拟落下得分、无需 Beam 时跳过状态去重，并在 Core.step 复用同一步正式合法落点。28 个传统状态动作一致、4 个模型各 500 块轨迹一致；传统同 seed 1000 块正式分数与结局一致，烟测约 8.48s→5.05s；定向回归 27 passed。依据：`training/evaluation/{post_curriculum_3m.py,failure_curriculum_eval.py}`、`training/traditional/v2.py`。

- 2026-10-17T20:01:29+08:00｜危险课程 A/B 训练、冻结门槛和有条件完整对局评测完成。依据：runs/ppo30m_failure_curriculum/{training_signal.json,development_gate.json,status.json}。

- 2026-10-17T20:05:31+08:00｜课程训练后长局比较失败并停止。依据：runs/post_curriculum_3m/status.json。
- 2026-10-17 20:08:59 CST｜课程训练完成后启动 300 万块/局评测；为启用经 856 个公开状态逐项对照的快速合法落点搜索，停止旧评测进程，保留并按 SHA256 清单导入 3 个已完成 PPO 局，后续又保留 1 个已完成 BC 局，未完成局重新排队；新队列 8 worker 正在运行，内存充足。神经网络 4 模型各 500 块、B12 1000 块、传统 V2 1000 块轨迹/得分对照一致；定向测试 30 passed、JS 61 passed。依据：`runs/post_curriculum_3m/{status.json,imported_canonical_games.json}`、`runs/ppo30m_failure_curriculum/post_curriculum_3m_fastbfs/`。

- 2026-10-17T20:11:06+08:00｜课程训练后长局比较失败并停止。依据：runs/post_curriculum_3m/status.json。
- 2026-10-17 20:13:00 CST｜修复长局恢复队列中本地变量遮蔽 checkpoint() 导致的启动失败，保留已完成局并恢复 8-worker 评测；每 1 万块原子保存局内 Core/RNG 与策略状态，PPO、B12、传统 V2 的同局恢复对照通过，现场检查确认快照正在写入。Parity 6 tests/29 scenarios/273 snapshots 与 JS 61 tests 通过。依据：`runs/post_curriculum_3m/status.json`、`runs/ppo30m_failure_curriculum/post_curriculum_3m_fastbfs/`。

- 2026-10-17T22:01:39+08:00｜为课程后 300 万块长局评测启用每状态碰撞缓存：46 个公开状态的合法落点耗时 3.118→2.669 ms/状态、结果逐项一致；快速环境回归 7 passed。按已有每 1 万块 Core/RNG 检查点停止并重启 8-worker tmux 队列，已完成对局保持不变，新 PID 328315，BC 7 局及 B12 1 局从检查点继续。依据：training/evaluation/fast_legal.py、runs/post_curriculum_3m/status.json、runs/ppo30m_failure_curriculum/post_curriculum_3m_fastbfs/。

- 2026-10-17T22:05:50+08:00｜优化长局评测专用候选特征计算：以公开棋盘位掩码计算与正式特征逐项相同的 1840×16 uint8 矩阵和合法掩码，不改训练特征或 TetrisCore。97 状态特征基准约 1.04→0.62 ms/状态；PPO 同种子 500 块 A/B 两轮约 3.49→3.13 秒，正式得分和消行一致。Python 定向测试 20 passed、Parity 6 passed/29 scenarios/273 snapshots、JS 61 passed。保留已完成局，从每 1 万块 Core/RNG 快照重启 8-worker tmux 队列，PID 330976。依据：training/evaluation/fast_candidate_features.py、tests/test_fast_candidate_env.py、runs/post_curriculum_3m/status.json。

- 2026-10-17T22:09:24+08:00｜在长局评测专用合法落点 BFS 中按原顺序展开 Left/Right/RotateCW/Down 邻居扩展，动作、路径和规则保持一致。84 状态搜索耗时约 2.996→2.760 ms/状态；PPO 同种子 500 块整局 A/B 两轮约快 4%，得分和消行一致。定向环境测试 10 passed；保留已完成局，从原子 Core/RNG 快照重启 8-worker tmux 队列，PID 332916。依据：training/evaluation/fast_legal.py、tests/test_fast_candidate_env.py、runs/post_curriculum_3m/status.json。

- 2026-10-17T23:54:07+08:00｜按用户授权建立 PPO 30M 三训练种子固定预算跟进：每种子普通与危险课程各 503808 样本；当前 300 万块传统 V2 评测结束后才训练，并在 16 个新评测种子上加入传统 V2 Hold 对照。冻结 PPO、BC、课程数据与 V2 代码哈希，评测种子冲突审计为零；传统 V2 50 块 smoke、相关定向测试 18 passed、新队列测试 3 passed。依据：runs/ppo30m_multiseed_followup/protocol.json、training/evaluation/ppo30m_multiseed_followup.py、reports/experiments/ppo30m_multiseed_followup_design.md。

- 2026-10-17T23:54:28+08:00｜已启动独立 tmux `tetris-ppo30m-multiseed-followup`，PID 355611；现场验证队列处于 `waiting_for_3m_evaluation`，等待传统 300 万块评测完成后自动训练，当前没有争用高负载资源。依据：runs/ppo30m_multiseed_followup/status.json、runs/ppo30m_multiseed_followup/queue.log。

- 2026-10-18T02:35:22+08:00｜完成课程训练后六策略、每局上限 300 万块的 8 种子长局比较。依据：runs/post_curriculum_3m/summary.json。

- 2026-10-18T02:51:19+08:00｜多种子训练 normal_994101 完成 503808 样本，事务 checkpoint 校验通过。依据：runs/ppo30m_multiseed_followup/training/normal_994101/committed/task_000123/model.zip。

- 2026-10-18T03:07:19+08:00｜多种子训练 curriculum_994101 完成 503808 样本，事务 checkpoint 校验通过。依据：runs/ppo30m_multiseed_followup/training/curriculum_994101/committed/task_000123/model.zip。

- 2026-10-18T03:23:11+08:00｜多种子训练 normal_994202 完成 503808 样本，事务 checkpoint 校验通过。依据：runs/ppo30m_multiseed_followup/training/normal_994202/committed/task_000123/model.zip。

- 2026-10-18T03:39:14+08:00｜多种子训练 curriculum_994202 完成 503808 样本，事务 checkpoint 校验通过。依据：runs/ppo30m_multiseed_followup/training/curriculum_994202/committed/task_000123/model.zip。

- 2026-10-18T03:55:05+08:00｜多种子训练 normal_994303 完成 503808 样本，事务 checkpoint 校验通过。依据：runs/ppo30m_multiseed_followup/training/normal_994303/committed/task_000123/model.zip。

- 2026-10-18T04:11:07+08:00｜多种子训练 curriculum_994303 完成 503808 样本，事务 checkpoint 校验通过。依据：runs/ppo30m_multiseed_followup/training/curriculum_994303/committed/task_000123/model.zip。

- 2026-10-18T05:45:29+08:00｜PPO 30M 三训练种子、普通与危险课程、传统 V2 同种子对照完成。依据：runs/ppo30m_multiseed_followup/summary.json。

- 2026-10-18｜冻结 PPO 30M 新候选独立确认协议：32 个种子 `1421000..1421031` 精确用途审计无冲突；原版 PPO、BC、normal_994303 与 normal_994202 模型 SHA256 和训练事务校验通过；100K 上限、预设门槛与 8 worker 配置已保存。依据：runs/ppo30m_independent_confirmation/seed_audit.json、runs/ppo30m_independent_confirmation/protocol.json。

- 2026-10-18T19:49:19+08:00｜独立 tmux `tetris-ppo30m-confirmation` 已启动并现场核验 PID 37034 处于 PPO 评测；Python 定向测试 9 passed、Parity 6 passed（35 subtests）、JS 61 passed。依据：runs/ppo30m_independent_confirmation/status.json、runs/ppo30m_independent_confirmation/queue.log。

- 2026-10-18T19:53:00+08:00｜独立确认队列暂停：审计发现 normal 评测依赖同时加载原 PPO 参考模型以计算排序诊断；保留已原子提交的 9 局 PPO 结果及在途快照，修正 worker 模型路径后恢复。依据：training/evaluation/ppo30m_independent_confirmation.py、runs/ppo30m_independent_confirmation/evaluation/ppo/。

- 2026-10-18T19:53:09+08:00｜修正 normal 模型 worker 加载原 PPO 参考模型后，独立评测从已提交的 9 局 PPO 结果恢复；50 块 normal_994303、normal_994202、B12 烟测均正常触顶。独立 tmux 收尾会话负责完成后的测试、报告提交与 push。依据：runs/ppo30m_independent_confirmation/status.json、runs/ppo30m_independent_confirmation/finalize.sh。

- 2026-10-18｜建立 Hybrid B12 事后机制研究协议：冻结 PPO/BC 与 13 死亡、3 触顶原始记录 SHA256；50 块策略烟测及 1421014 全局重放严格匹配原评测。1421014:183 双分支烟测通过；带烟测标签的临时分支已清除，正式队列将按冻结标签运行。依据：runs/hybrid_b12_death_study/protocol.json、training/evaluation/hybrid_b12_death_study.py。

- 2026-10-18T21:11:28+08:00｜独立 32-seed PPO 候选 100K 评测完成；预冻结门槛未通过。依据：runs/ppo30m_independent_confirmation/summary.json。

- 2026-10-18T23:26:19+08:00｜Hybrid B12 机制研究独立 tmux `tetris-b12-death-study` 已启动，PID 79364，现场确认进入 16 局重放阶段；13 项定向测试通过。依据：runs/hybrid_b12_death_study/status.json、runs/hybrid_b12_death_study/queue.log。

- 2026-10-18T23:40:49+08:00｜Hybrid B12 独立失败机制研究完成：16 局重放、46 个状态、92 条反事实分支。依据：runs/hybrid_b12_death_study/summary.json。

- 2026-10-18T23:43:51+08:00｜Hybrid B12 机制研究已完成并推送；16 局重放、46 个同状态快照和 92 条分支完整性复查通过。为 16 份重放结果补齐逐文件 SHA256 seal，复核模型和原始记录哈希，相关定向测试 13 passed。依据：runs/hybrid_b12_death_study/replays/、runs/hybrid_b12_death_study/summary.json、runs/hybrid_b12_death_study/final_commit.txt。

- 2026-10-19T00:01:07+08:00｜冻结 Hybrid B12 V3-A 规则、PPO/BC/Core 哈希、新开发与独立确认种子及预设门槛；全仓库种子用途审计无冲突。14 项相关 Python 测试通过，四策略 100 块 Smoke 全部合法触顶。旧反事实位置只用于规则假设，不用作确认结果。依据：runs/hybrid_b12_v3/protocol.json、runs/hybrid_b12_v3/seed_audit.json、runs/hybrid_b12_v3/smoke/。

- 2026-10-19T00:01:40+08:00｜独立 tmux `tetris-b12-v3` 已启动并现场核验 PID 90482 进入 16-seed 开发 20K 阶段；收尾会话 `tetris-b12-v3-finalize` 已启动，达不到冻结门槛将不访问确认组。依据：runs/hybrid_b12_v3/status.json、runs/hybrid_b12_v3/queue.log、runs/hybrid_b12_v3/finalize.sh。

- 2026-10-19T00:17:55+08:00｜Hybrid B12 V3-A 队列完成：stopped_after_20k，独立确认未通过或未启动；逐局 SHA256 与 Python/Parity/JS 回归测试通过。依据：runs/hybrid_b12_v3/status.json、reports/experiments/hybrid_b12_v3.md。

- 2026-10-19T00:18:18+08:00｜Hybrid B12 V3-A 20K 初筛失败并自动停止：P10 与早死数量未过冻结门槛；100K 开发及 32-seed 独立确认未运行。完善报告中按块归一的推理耗时；逐局数据 SHA256、Python/Parity 33 passed（35 subtests）、JS 61 passed。依据：runs/hybrid_b12_v3/development_20k_gate.json、runs/hybrid_b12_v3/finalize.log、reports/experiments/hybrid_b12_v3.md。

- 2026-10-19 10:34 +08:00：冻结 V2 Hold 诊断协议，复用 B12 46 个死亡状态并选 3 个触顶对照；单状态合法性和恢复烟测通过，9 项定向 Python 测试通过。证据：`runs/hybrid_b12_v2_expert_study/protocol.json`、`runs/hybrid_b12_v2_expert_study/candidates/seed_1421001_step_606.json`。
- 2026-10-19 10:39–10:41 +08:00：`tetris-b12-v2-expert` 队列完成 49 个状态、112 条唯一动作反事实；49 个候选与 112 条分支 SHA256 均校验通过。3 个死亡 seed 有 V2 独立短窗救援，1 个 seed 有明显有害动作；完成报告，队列已退出。定向 Python 9 项、Parity 6 项及 35 子测试、JS 61 项通过。队列中旧的 `npm test` 调用因仓库根目录无 `package.json` 失败，随后使用正确的 `node --test tests/*.test.cjs` 命令通过。证据：`runs/hybrid_b12_v2_expert_study/{status.json,summary.json,queue.log}`、`reports/experiments/hybrid_b12_v2_expert_study.md`。
- 2026-10-19 15:38 +08:00：完成 B12 + V2 单步救援规则、模型和新种子审计，`1440000..1440031` 及 `1441000..1441031` 无历史用途冲突。旧 3 个救援状态均触发，旧有害例与有益 BC 拦截不触发；四方案 100 块 smoke 均合法触顶，7 项定向测试通过。冻结协议见 `runs/hybrid_b12_v2_rescue/protocol.json`，smoke 见同目录 `smoke/`。
- 2026-10-19 15:40–15:55 +08:00：`tetris-b12-v2-rescue` tmux 会话已启动并验证存在，4 CPU worker；16 种子 20K 初筛完成且通过预冻结得分、四消和早死门槛，自动进入 32 种子 100K 开发评测。20K 数据：B12/救援平均存活 18066/18282，平均得分 5.34/5.46 亿，早死均 1 局；V2 调用 68 次、改动作 27 次。依据：`runs/hybrid_b12_v2_rescue/development_20k_gate.json`、`runs/hybrid_b12_v2_rescue/status.json`。

- 2026-10-19 17:45 +0800｜B12 + V2 单步救援队列完成；评测结论见 `runs/hybrid_b12_v2_rescue/status.json`、报告 `reports/experiments/hybrid_b12_v2_rescue.md`，Python/Parity/JS 测试日志位于实验目录。
- 2026-10-19 18:35 +08:00：冻结 B12＋V2 救援归因协议，核对五个开发种子的原始结果 SHA256；`1440024` Rescue 重放烟测与原存活、得分和两次 V2 替换严格一致。首次 V2 替换前完整 Core/RNG 快照可恢复，B12 首步短窗分支合法执行。证据：`runs/hybrid_b12_rescue_attribution/{protocol.json,replays/Rescue/seed_1440024.json,states/seed_1440024_step_17450.pkl,branches_5000/seed_1440024_step_17450_b12_then_b12.json}`。

- 2026-10-19 22:57 +0800｜B12＋V2 正负差异归因后台队列完成。五局重放、短窗及 100K 反事实和 SHA256 校验见 `runs/hybrid_b12_rescue_attribution/`；报告见 `reports/experiments/hybrid_b12_rescue_attribution.md`，Python/Parity/JS 结果见相应测试日志。
- 2026-10-19 22:58 +08:00：B12＋V2 救援归因报告补充首次替换、后续多次干预与公开状态反例。10 局重放、15 个完整状态快照、15 个候选记录、45 条短窗及 18 条长窗分支均逐文件通过 SHA256 校验；Python 13 项、Parity 6 项及 35 子测试、JS 61 项通过，tmux 队列已退出。证据：`runs/hybrid_b12_rescue_attribution/`、`reports/experiments/hybrid_b12_rescue_attribution.md`。
- 2026-10-20 00:08 +08:00：冻结三专家数据集协议和源码 SHA256。审计新采集 `1450000..1450031`、探索验证 `1451000..1451015`、预留确认 `1452000..1452031` 无历史冲突，未来确认组不运行。首局 `1450000` B12 死亡于 72665 块，事前选 2 个状态；同状态 B12/V2 分支的 2/5000+ 块结果与状态恢复、合法掩码一致。定向 Python 7 项通过。证据：`runs/hybrid_b12_expert_dataset/{protocol.json,seed_audit.json,episodes/train/seed_1450000.json,branches/}`。
- 2026-10-20 00:09 +08:00：独立 tmux 会话 `tetris-b12-expert-dataset` 已启动并核验存在，固定 4 个 CPU worker；队列顺序为 32 局训练候选组、16 局探索验证组、入选状态双分支、统计与测试。已完成首局及分支烟测，未来确认组不在任务列表中。证据：`runs/hybrid_b12_expert_dataset/{queue.sh,status.json,protocol.json}`。

- 2026-10-20 01:11 +0800｜三专家危险决策数据集固定 48 局及 5000 块反事实队列完成；状态、校验和及统计见 `runs/hybrid_b12_expert_dataset/`，报告见 `reports/experiments/hybrid_b12_expert_dataset.md`，Python/Parity/JS 回归见实验测试日志。
- 2026-10-20 01:12 +08:00：复核三专家数据集 48 局、133 个入选状态快照、202 条分支逐文件 SHA256，失败 0；预留 `1452000..1452031` 无输出文件。补充报告中的验证组零明确救援、落点特征反例与数据不足结论；冻结采集器源码未改。证据：`runs/hybrid_b12_expert_dataset/{status.json,analysis.json,episodes/,states/,branches/}`、`reports/experiments/hybrid_b12_expert_dataset.md`。
- 2026-10-20 09:33 +08:00：冻结扩充批次协议和源码 SHA256；审计新采集 `1460000..1460031`、新探索验证 `1461000..1461015` 无历史冲突，原正式确认 `1452000..1452031` 仍预留。单局 `1460000` B12 于 48940 块真正死亡，入选 2 状态，首个双分支烟测完成；扩充批次 6 项定向测试通过。独立 tmux `tetris-b12-expert-dataset-extension` 启动并核验 PID 434628、4 CPU worker，正在采集组。证据：`runs/hybrid_b12_expert_dataset_extension/{protocol.json,seed_audit.json,status.json,episodes/train/seed_1460000.json,branches/,queue.sh}`。
- 2026-10-20 11:37 +08:00：网页神经 AI 推理服务独立 tmux `tetris-web-inference` 启动，PID 461573，绑定 Tailscale `100.91.227.47:8081`；原 8080 静态服务 PID 1860 持续可用。启动校验 PPO/BC 模型 SHA256，服务只接受对应 8080 Origin；三模式实际 HTTP 返回合法动作，异源 Origin 返回 403。证据：`runs/web_inference/service.log`、8081 `/api/health`。
- 2026-10-20 11:42 +08:00：PPO、BC、B12 各完成 Seed 12345 的 300 块真实 HTTP＋网页 BFS 执行测试；逐步棋盘、Current/Next/Hold、分数、等级、消行、结束状态及合法动作与 Python Core 一致。300 次实测服务端平均决策分别 4.76/5.68/5.51 ms，B12 采用 PPO/BC 297/3。相关 Python 36 项、Parity 6 项及 35 子测试、JS 65 项通过。证据：`runs/web_inference/{live_ppo_300.json,live_bc_300.json,live_b12_300.json,latency_300.json,python_tests.log,parity_tests.log,js_tests.log}`、`reports/experiments/web_ai_integration.md`。
- 2026-10-20 11:52 +08:00：长局网页链路烟测：Seed 12345 的 PPO 100 块逐步校验并从 checkpoint 续跑至 700 块；发现并修复刷新恢复时神经 AI 在合法落点脚本加载前发起决策的时序问题。服务重启后从 650 块续跑到 700 块，8080 持续正常；两个并发网页会话 PPO/BC 各 200 块逐步校验通过。原子 SHA256 及恢复检查见 `runs/web_longrun_validation/{smoke_ppo,concurrent_ppo,concurrent_bc}/`；神经模式刷新、暂停、切换、错误重试及请求超时定向 JS 7 项通过。

- 2026-10-20 10:33 +0800｜三专家数据扩充队列完成：48 局、148 个入选状态、254 条分支逐文件 SHA256 校验通过；Python/Parity/JS 日志通过，预留确认种子无输出。证据：`runs/hybrid_b12_expert_dataset_extension/`、`reports/experiments/hybrid_b12_expert_dataset_extension.md`。

- 2026-10-20 12:57 +0800｜网页长局端到端队列完成：PPO/BC/B12 各最多 10K 与 BC 最多 100K；逐 500 块 SHA256 原子快照和完整状态对照通过，Python/Parity/JS 回归日志通过，结论见 `reports/experiments/web_longrun_validation.md`，证据位于 `runs/web_longrun_validation/`。
- 2026-10-20 14:06 +0800：执行 v1.0.0 发布检查：本地 `main` 与 `origin/main` 均为 `ab21d69`，本地和远端无 `v1.0.0` 标签；两份冻结模型 SHA256 与 `training/evaluation/hybrid_queue.py` 一致，`verify_models()` 通过；8080 静态页和 Tailscale 8081 `/api/health` 返回正常。Python 定向 6 项、Parity 6 项、JS 68 项通过。其他任务未提交文件未触碰；发布说明记录了独立备份未核验。证据：`reports/releases/v1.0.0.md`、本次 Git 提交。
- 2026-10-20 14:09 +0800：扩大 Python 回归发现历史 V2 Hold 配置代码哈希与当前实现不符，`tests/test_candidate_vs_v2_hold.py` 两项失败；全量测试中止时 64 通过、1 跳过。发布相关定向 Python 28 项另行通过。未修改历史配置、规则或权重；暂缓创建 `v1.0.0` 标签。证据：`reports/releases/v1.0.0.md`。
- 2026-10-20 14:50 +0800：建立独立分支 `research/autonomous-rl`，以隔离 worktree 对比 V2 历史身份，定位唯一源码差异为 `training/traditional/v2.py` 的评分和去重加速。三个历史 seed 各前 200 步动作及正式得分/消行重放一致；建立源码双身份与语义回归。CUDA 当前不可用，未修改驱动；预设独立 CPU 4 worker 的有限预算事务实验，训练和评测仅写入 `runs/autonomous_rl_research/`。`/data` 与 `/home` 同一 NVMe，未将其视为独立备份。依据：`reports/experiments/autonomous_rl_design.md`、`training/evaluation/fixtures/v2_hold_historical_prefix.json`。
- 2026-10-20 14:52 +0800：独立 tmux `tetris-autonomous-rl` 已核验运行；CPU 4-worker 第一事务 4096 步完整提交，墙钟 22.63 秒。两局 2000 块功能评测均触顶，PPO/候选四消占比分别 94.19%/92.53%，只作烟测不判长期改进。队列进入目标 123 事务的第一阶段；网页 8080、8081 保持健康。证据：`runs/autonomous_rl_research/{status.json,pilot_timing.json,evaluation/smoke/summary.json,training/transaction_state.json}`。
- 2026-10-20 14:54 +0800：额外重放传统 V2 历史 100016–100025 十个 seed 的各前 1000 个动作；当前 V2 动作与历史记录逐步一致，正式得分与消行由未变化 Core 重新计算。此为有限前缀等价证据，不宣称十局 100K 全程一致。依据：`reports/experiments/autonomous_rl_design.md` 与原始 `reports/traditional_longlife/validation10_100000_20261008/hold/seed_*.json`。
- 2026-10-20 14:59 +0800：独立收尾 tmux `tetris-autonomous-rl-finalize` 已核验存在，PID 2645675；主队列终止后自动运行 Python、Parity、JS 回归，生成阶段总结并仅提交研究报告、CHANGELOG 与 OPERATIONS 至 `research/autonomous-rl`。主训练队列此时已提交 23 个事务（94,208 步），尚未到开发评测门槛。依据：`runs/autonomous_rl_research/{finalize.sh,training/transaction_state.json}`。
- 2026-10-20 22:17 +0800：自主训练队列实际于 15:48 按开发 20K 四消门槛停止：503,808 步、16 个新 seed，候选四消 88.26% 对 PPO 92.85%；无 100K 或独立确认，模型事务 SHA256 已核验。收尾 tmux 最初因会话名前缀匹配自身而等待，22:11 改为精确会话名后完成并推送 `a32c40b`，Python 21、Parity 6、JS 68 项通过。停止后新增可选合法落点缓存训练环境，200 步状态/动作/奖励/快照对照通过；3 seed×300 步×3 次配对基准 144.6→255.4 步/秒，RSS 峰值约 495 MiB，未修改原实验配置或恢复其训练。证据：`runs/autonomous_rl_research/{status.json,evaluation/dev_20k/summary.json,performance.json,inference_benchmark.json,finalize.log}`、`reports/experiments/autonomous_rl_results.md`。
- 2026-10-20 22:21 +0800：按用户继续开发授权，冻结完整事务训练缓存对照协议：新种子 1493000–1493003，普通/缓存环境各两个 4096 步 CPU 事务，第二事务从 checkpoint 恢复，比较参数、非计时指标、吞吐和内存。若不一致立即停止，不将单环境微基准当训练加速。CUDA 仍不可用，网页服务保持运行。依据：`reports/experiments/autonomous_rl_followup_design.md`、`training/evaluation/full_train_cache_benchmark.py`。

- 2026-10-20 22:12 +0800：自主研究队列结束，状态 stopped_dev_20k；Python/Parity/JS 回归通过，详细结果和限制见 `runs/autonomous_rl_research/status.json`、`reports/experiments/autonomous_rl_results.md`。
- 2026-10-20 22:25 +0800：`tetris-cache-train-benchmark` 完成普通/缓存环境各两个 4096 步事务，第二事务从 checkpoint 恢复；各对应事务策略参数与非计时指标精确一致，墙钟 20.312/20.315 秒对 17.562/17.400 秒，合计吞吐 1.162×，峰值 RSS 654588 KiB。历史 PPO 死亡反事实仅 19 个独立死亡 seed，明显负例不足，本轮停止排序训练设想，未使用确认种子。证据：`runs/autonomous_rl_research/full_train_cache_benchmark/summary.json`、`reports/experiments/autonomous_rl_followup_results.md`。
- 2026-10-21 22:48 +08:00：冻结 PPO30M 新种子反事实协议，审计 1517000..1517007 无文件冲突；首局 1517000 采集 3000 块、5 个事前状态。首次快照烟测因候选概率输入未转 Tensor 失败，修正后同状态 8 块分支两次完全一致；定向测试 2 项通过。独立 tmux `tetris-ppo-fresh-cf` 已启动，8 seed、每 seed 最多 5 状态、每状态最多 5 动作、2048 块上限、90 分钟墙钟上限。证据：`runs/ppo30m_fresh_counterfactual/{protocol.json,seed_audit.json,episodes/,states/,queue.log}`。
- 2026-10-21 22:49 +08:00：首次队列启动因冻结协议 `fixed_steps` 的 tuple/JSON list 比较失败，尚未生成反事实分支；统一为 list 后重新启动同一 tmux 队列，冻结协议内容未改。证据：`runs/ppo30m_fresh_counterfactual/{protocol.json,queue.log}`。
- 2026-10-21 22:54 +0800：在隔离检出中为 Python 合法落点 BFS 添加单次搜索内的碰撞结果缓存，保持正式规则和输出不变；尚未合入正在执行反事实采集的研究分支。242 个冻结公开状态的完整合法落点 JSON SHA256 在原版和缓存版同为 `33686768d78f206dfda40f088638c1e8ba072cb8dfdbf82f424fb7bf111a6ccd`；5 次遍历中位耗时 0.699→0.589 秒，约 1.19×。Parity 6 项、相关 Python 12 项通过（2 跳过）、JS 68 项通过。状态样本仅用于性能和语义验证，不作模型评测；原始样本临时文件 `/tmp/tetris_geometry_states.json`。
- 2026-10-21 23:03 +08:00：PPO30M 新种子反事实固定队列完成，8 局均至 3000 块上限，33 状态、91 分支、58 替代比较；逐文件快照/候选/分支身份及哈希检查通过。58 对均在 2048 块右截断，严格生存及四消负例均为 0；按预算停止扩充和排序训练。恢复烟测初次因候选安全特征 tuple/JSON list 表示不一致失败，统一序列化表示后 33 状态无新增分支复核通过；协议和数值未变。定向测试 2 项通过。证据：`runs/ppo30m_fresh_counterfactual/{analysis.json,status.json,queue.log,resume.log}`、`reports/experiments/ppo30m_fresh_counterfactual_results.md`。
- 2026-10-21 22:57 +0800：在隔离分支修复 PPO 死亡反事实任务的历史结果复用校验：候选记录核对完整状态身份，已封存分支核对 seed、step、首动作、观察窗口、choice、模型身份与结果数组长度。未改动历史文件或正式规则。定向测试 4 项通过，另从旧数据直接恢复 `992000:26404` 的 1000 块分支通过；新增重封存但更改窗口或候选动作的拒绝测试。正在运行的全新 PPO 数据任务使用独立代码和目录，未受影响。
- 2026-10-21 23:02 +0800：强化候选复用校验，重算确定性 PPO/BC/公开安全候选并与已存候选逐字段比较；新增保持状态身份不变但篡改候选来源时拒绝复用的测试。4 项定向测试通过，历史 `992000:26404` 的候选重新计算与存档完全一致。证据：`tests/test_ppo_death_analysis.py`、`training/evaluation/ppo_death_counterfactual.py`。
- 2026-10-21 23:07 +0800：在隔离检出合并合法落点碰撞缓存与 PPO 反事实恢复身份校验；由于合法落点源码改变传统 V2 的组合代码哈希，登记新身份 `11ec3adb30b0b17aa09a80489090864676d16d62f329326f0e2fef1e7f4129ea`，仍需通过既有 10 seed×1000 步冻结动作/得分/清行前缀重放，已通过 3 项 V2 对照测试。未更改历史 V2 哈希或评测结果。证据：`training/evaluation/v2_semantic_identity.py`、`tests/test_candidate_vs_v2_hold.py`。
- 2026-10-21 23:13 +0800：隔离检出完成合法落点碰撞缓存的完整 PPO 事务对照；原版与优化版各运行普通/缓存环境两轮 4096 步事务并恢复 checkpoint，四对模型参数逐张量一致，非计时指标一致。原版普通/缓存耗时 40.123/34.764 秒，优化版 37.565/33.277 秒；短基准受运行时负载影响。优化版首次 tmux 启动因日志父目录未建立而失败，建目录后重新启动成功，没有产生已提交事务。证据：`reports/experiments/autonomous_rl_collision_cache_results.md`、两个隔离检出各自的 `runs/autonomous_rl_research/full_train_cache_benchmark/summary.json`。
- 2026-10-22 12:28 +0800：在隔离研究检出为新建 PPO 向量事务加入关键训练源码逐文件 SHA256，恢复时校验；历史元数据缺少清单时保持原恢复路径。未运行新训练、未改动既有 checkpoint。初次 Parity 测试命令引用不存在的 `tests/test_parity.py`，改用正确路径后训练事务/V2/Parity 共 10 通过、2 项 CUDA 依赖测试跳过、Parity 内 29 个子测试通过，JS 68 通过。证据：`training/train_vector_transaction.py`、`tests/test_train_vector_transaction.py`、`reports/experiments/autonomous_training_source_integrity.md`。
- 2026-10-22 14:49 +0800：在隔离研究检出实现可选 `survival_balanced_v1` 训练奖励并冻结单候选 65,536 步 CPU 协议。PPO30M 冻结权重 SHA256 校验通过；开发/预留确认种子按 seed 字段和文件名扫描现有仓库无命中。NVIDIA 驱动当前不可通信，未修改驱动。奖励/环境/Parity 测试 31 通过、1 跳过、Parity 内 29 子测试通过；JS 68 通过。尚未开始正式训练。证据：`reports/experiments/ppo30m_survival_balanced_design.md`、`tests/test_balanced_reward.py`、`runs/ppo30m_survival_balanced/preset.json`。
- 2026-10-22 14:53 +0800：生存优先 PPO CPU 训练先后完成两个 4096 步事务，第二个从首个已提交事务恢复；两次 `verify_task`、当前源码身份校验通过，已提交 8192/65536 步。训练只使用 `1520000..1520003`，冻结模型及网页未改动。补充 16-seed 开发评测原子队列，四消下降本身不作为成功判据。证据：`runs/ppo30m_survival_balanced/{protocol.json,smoke.log,resume_smoke.log,training/transaction_state.json}`、`training/evaluation/balanced_reward_queue.py`。
- 2026-10-22 15:05 +0800：tmux `tetris-balanced-ppo` 完成全部 16 个训练事务（65,536 步；训练事务墙钟合计 263.92 秒），随后完成 PPO/候选相同 16 种子的 32 局 20K 开发评测。候选四消 88.13% 对 PPO 92.58%，但 P10 14,172 对 19,058，未通过预设门槛；状态 `stopped_20k_gate`，未进入 100K 或使用预留确认种子。保留事务 14–16 清单、32 局 seed/上限/模型身份和结果文件哈希校验通过，队列正常退出。证据：`runs/ppo30m_survival_balanced/{training/transaction_state.json,evaluation_status.json,integrity.json}`、`reports/experiments/ppo30m_survival_balanced_results.md`。
- 2026-10-22 15:13 +0800：事后探索性复盘队列初次运行时，4 个 worker 的 PyTorch 推理各启用多线程，CPU 占用过高；在无任何整局结果提交时终止本任务进程组，保留原 `failure_diagnostic/protocol.json` 和 `failure_diagnostic.log`，不影响训练、已完成评测或网页。修正为每 worker 单线程，结果写入新的 `failure_diagnostic_v2/`；仅重放既有开发种子，不用于独立确认。证据：`training/evaluation/balanced_reward_failure.py`、`runs/ppo30m_survival_balanced/failure_diagnostic.log`。
- 2026-10-22 15:18 +0800：`tetris-balanced-failure-v2` 完成 7 个历史开发种子的 PPO/候选成对重放；14 条轨迹的存活、正式得分、消行分布及结束状态与原评测完全一致，7 个原子文件 SHA256 复核通过。新增死亡候选最后 100 块高度/洞数急升且四消减少；高位与长四消空窗存在存活反例，只作为机制线索。tmux 正常退出，未训练新模型或动用独立确认种子。证据：`runs/ppo30m_survival_balanced/failure_diagnostic_v2/{status.json,integrity.json}`、`reports/experiments/ppo30m_survival_balanced_failure_diagnostic.md`。
- 2026-10-22 15:24 +0800：在隔离研究检出冻结 PPO 快速堆高/高位对照的前瞻采样协议：8 个新种子、50K 每局上限、每局最多 5 个事前选定公开状态；预留 1531xxx 不运行。主检出训练/测试/报告/runs 的精确 seed 字段及文件名扫描无命中，冻结 PPO SHA256 校验通过。新增采样逻辑定向测试 2 项通过；尚未开始长局采集。证据：`reports/experiments/ppo30m_transition_counterfactual_design.md`、`training/evaluation/ppo_transition_capture.py`、`tests/test_ppo_transition_capture.py`。
- 2026-10-22 15:30 +0800：tmux `tetris-ppo-transition-capture` 完成 8 个新 PPO 长局：3 局真实死亡、5 局 50K 右截断，24 个前瞻入选状态，快速堆高覆盖 8/8 seed、高位对照按冻结定义为 0。全部入选状态离真实死亡均超过本轮 3000 块分支观察窗，因此不能产生预设“原 PPO 窗口内死亡后救活”证据；仍按预算做替代动作短窗伤害检查。恢复快照与正式合法动作候选烟测通过（1530000:5000，3 个去重动作）；未改采样阈值。证据：`runs/ppo30m_transition_counterfactual/{capture_status.json,capture_summary.json,episodes/,states/}`、`training/evaluation/ppo_transition_branches.py`。
- 2026-10-22 15:33 +0800：独立采集审计发现 1530004:5000 的固定时点样本与快速堆高条件重叠，被原始记录误标为第三个快速堆高样本；首次审计失败并停止分支执行。原始记录及快照未改，生成带来源哈希的派生纠正清单，改标为固定时点样本后 8 局、24 状态通过快照/公开状态/入选概率/截断核验（快速堆高 16，固定时点 8，高位对照 0）。定向测试 4 项通过。证据：`runs/ppo30m_transition_counterfactual/capture_integrity.json`、`episodes_corrected/`。
- 2026-10-22 15:34 +0800：审计与有限反事实执行器提交 `2bdfbc7` 并推送研究分支；启动 tmux `tetris-ppo-transition-branches`，pane PID 3159523。队列仅使用已审计的派生清单，最多 200 分支、90 分钟；当前尚未产生完成结果。证据：`runs/ppo30m_transition_counterfactual/branch.log`、`branch_status.json`。
- 2026-10-22 15:42 +0800：有限反事实队列完成 24 状态、57 去重分支、33 替代动作比较；137/137 文件 SHA256 复核一致，tmux 正常退出。原 PPO 和替代分支全部达到 1000/3000 块观察上限，均为右截断，无可判定短窗救援或伤害；不能据此调低四消或宣称改善稳定性。结果与局限见 `runs/ppo30m_transition_counterfactual/{branch_status.json,analysis.json,integrity.json}`、`reports/experiments/ppo30m_transition_counterfactual_results.md`。
- 2026-10-22 15:44 +0800：完成结果报告；相关 Python 6 通过、Parity 6 通过（29 场景/273 快照）、JS 68 通过。网页及冻结模型未变；证据：`reports/experiments/ppo30m_transition_counterfactual_results.md` 与本次测试输出。
- 2026-10-22 15:49 +0800：`tetris-ppo-terminal-exploration` 完成三个已知死亡 seed 的精确重放和 9 个临终状态、23 条去重分支，tmux 正常退出；3 局死亡块数及正式得分均精确重现，53/53 个文件 SHA256 通过。12/14 个替代动作满足短窗延命判据，但替代分支全部在 2000 块右截断，且状态是事后选取；仅作为机制线索，不训练/调参。证据：`runs/ppo30m_transition_counterfactual/terminal_exploratory/{status.json,integrity.json,comparisons/}`。
- 2026-10-22 20:14 +0800：完成临终反事实结果报告，指出 PPO Top-2 在三个死亡 seed 中均出现短窗延命，BC 本组无独立救援动作；正式 Core、冻结模型与网页服务未变。证据：`reports/experiments/ppo30m_terminal_exploration_results.md`。
- 2026-10-22 20:17 +0800：`tetris-ppo-top2-extension` 按运行前冻结协议延长三条死亡前 500 块的 PPO Top-2 分支；三条均走满 10K 观察窗、未真实死亡，前 2000 块得分与四消序列和已提交短分支逐项一致，3/3 文件 SHA256 复核通过，tmux 正常退出。仅为事后探索，不视作独立验证或无限生存。8080 网页及 8081 `/api/health` HTTP 均为 200。证据：`runs/ppo30m_transition_counterfactual/terminal_exploratory/extended_top2/{status.json,integrity.json,seed_*.json}`、结果报告。
- 2026-10-22 20:30 +0800：冻结 24 个新 PPO seed、50K 上限、最多 4 worker 和稀疏 Core/RNG 检查点协议，提交并推送 `d386f43`。首次 tmux 启动因新结果目录未建立、shell 日志重定向失败，尚无对局；创建目录后重启 `tetris-ppo-top2-generalization`，pane PID 3214057，90 分钟墙钟上限。候选组死亡不足 3 局则不启动复现组，正式确认预留种子不使用。证据：`reports/experiments/ppo30m_top2_generalization_design.md`、`runs/ppo30m_top2_generalization/{capture.log,capture_protocol.json,seed_audit.json}`。
- 2026-10-22 20:31 +0800：新采集队列已完成前 3 个候选 seed，三局均真实死亡（12,452、14,498、27,377 块），精确 seed 审计为零冲突；队列继续运行。新增独立 Top-1/Top-2 配对重放程序，尚未启动分支评测。证据：`runs/ppo30m_top2_generalization/{capture_status.json,episodes/,capture.log}`、`training/evaluation/ppo_top2_generalization_branches.py`。
- 2026-10-22 20:33 +0800：配对重放程序已提交推送 `6176ff1`，启动独立 tmux `tetris-ppo-top2-branches`（pane PID 3215506）等待采集会话终止后自动检查合规完成状态；若采集失败或候选死亡不足 3 局，分支程序停止，不会静默继续。最多 48 分支、90 分钟墙钟、每分支 10K。证据：`runs/ppo30m_top2_generalization/branches.log`、`branch_status.json`（进入分支后）。
- 2026-10-22 20:42 +0800：24/24 全新 PPO 对局采集完成；候选组 6 死亡/10 触顶，探索复现组 3 死亡/5 触顶。独立审计核验 96 个稀疏 Core/RNG 检查点、216/216 个结果与清单文件 SHA256 全部一致，预留确认种子未用；采集 tmux 正常退出，配对 tmux 开始运行。证据：`runs/ppo30m_top2_generalization/{capture_status.json,capture_summary.json,capture_integrity.json,episodes/,checkpoints/}`。
- 2026-10-22 20:43 +0800：Top-1/Top-2 配对已完成前 2 个 seed，1 个满足冻结短窗延命判据、1 个不满足；继续按固定 24 seed 队列运行。相关 Python 6 项、Parity 6 项（29 场景/273 快照）、JS 68 项回归均通过。证据：`runs/ppo30m_top2_generalization/{branch_status.json,branches.log,comparisons/}` 及本次测试输出。
- 2026-10-22 20:47 +0800：提交并推送按独立 seed 组汇总的可复现分析程序 `7ac2e30`；启动 tmux `tetris-ppo-top2-analysis`（pane PID 3220076），等配对任务正常结束后核验所有分支文件 SHA256 并自动生成 `analysis.json`。若配对未完成，分析程序拒绝输出。证据：`training/evaluation/ppo_top2_generalization_analyze.py`、`runs/ppo30m_top2_generalization/analysis.log`。
- 2026-10-22 20:52 +0800：在探索复现组分支尚未运行前，冻结只使用落子前公开观察的一步特征表，提交并推送 `9889592`；tmux `tetris-ppo-top2-features`（pane PID 3222261）等待完整分组分析后自动生成带来源哈希的 `public_feature_table.json`。该表包含结局用于离线诊断，但未来在线决策只能使用记录的公开特征，不读取结果标签。证据：`reports/experiments/ppo30m_top2_generalization_design.md`、`training/evaluation/ppo_top2_generalization_features.py`。
- 2026-10-22 21:16 +0800：新 seed Top-1/Top-2 配对 24/24 状态、48/48 原子分支完成，三层 tmux 队列正常退出；候选组死亡局 4/6 短窗延命，但 10 个存活对照有 2 个 Top-2 真正提前死亡；独立探索复现组死亡局 2/3 延命、5 个存活对照 0 提前死亡。采集 216/216、分支 144/144、公开特征源 48/48 SHA256 均核对通过，源码哈希与冻结协议一致。证据：`runs/ppo30m_top2_generalization/{capture_integrity.json,branch_integrity.json,analysis.json,public_feature_table.json}`、`reports/experiments/ppo30m_top2_generalization_results.md`。
- 2026-10-22 21:18 +0800：完成 24 seed 逐局报告，明确 Top-2 有救援与害局双向案例、无可靠公开局部特征可作为在线切换规则；不启动选择器训练或修改四消。所有已提交结果在再次执行分支程序时经身份校验复用，未重复计算；相关 Python 6、Parity 6、JS 68 项通过；8080/8081 健康检查均 200。证据：同上报告、本次测试与 HTTP 输出。
- 2026-10-22 22:46 +0800：在独立研究 worktree 实现冻结 PPO 反事实快速分支与配对吞吐基准，未运行正式基准；快速分支等价性和环境测试 12 项通过。为用户离开期间准备有 2026-10-28 截止时间、互斥锁、单次 3 小时上限的 Codex CLI 定时任务脚本与提示；cron 尚待安装。证据：`training/evaluation/fast_counterfactual.py`、`training/evaluation/benchmark_fast_counterfactual.py`、`tests/test_fast_counterfactual.py`、`ops/unattended_research.sh`。
- 2026-10-22 22:46 +0800：研究分支 `codex/ppo-branch-throughput` 提交并推送 `4ff3380`；安装现有用户 crontab 每日 02:00/14:00 的五天有界研发唤醒，cron 服务为 active、原网页开机任务保留；启动 `tetris-ppo-branch-bench` tmux（启动时 pane PID 3245534）运行三状态配对吞吐实验，结果待完成。定时脚本以文件锁防重叠，单次 3 小时，2026-10-28 00:00 +0800 后不再启动。证据：`crontab -l`、`runs/ppo_branch_throughput_benchmark/benchmark.log`、`ops/unattended_research.sh`。
- 2026-10-22 22:47 +0800：定时命令预检发现 Codex CLI 不允许同时传 `--approve-for-me` 与 `-s workspace-write`；已修复脚本，仅保留前者（CLI 显示自动选择 workspace-write）。同一 worktree 的无人值守 CLI 启动与 shell `pwd` 实测成功；无训练或模型更改。证据：`ops/unattended_research.sh`、Codex CLI 自检输出。
- 2026-10-22 22:49 +0800：`tetris-ppo-branch-bench` 三状态配对基准正常退出，9 对各 1,500 块分支结果逐字段一致；旧版合计 47.094 秒、快速版 32.868 秒，提速 1.433×，进程峰值 RSS 663,348 KiB。Python 定向 12、Parity 6（29 场景/273 快照）、JS 68 项通过；正式规则/模型/网页未改。证据：`runs/ppo_branch_throughput_benchmark/{protocol.json,summary.json,benchmark.log}`、`reports/experiments/ppo_branch_throughput_results.md`。
- 2026-10-22 23:04 +0800：依用户要求将无人值守唤醒从每日两次改为每 5 小时一次：保留现有开机网页任务，将 cron 改为每小时调用轻量包装器，包装器用上次启动时间强制至少 18,000 秒间隔、文件锁防重叠，并在日志出现明确额度耗尽错误时写停止标记；原五天截止时间 2026-10-28 00:00 +0800 保留。`bash -n` 通过、cron 为 active，下一轮等待整点触发；未实际消耗新训练预算。证据：`ops/unattended_research.sh`、`crontab -l`。
- 2026-10-22 23:05 +0800：按用户最新“直到 token 用完”的明确要求，移除五天截止时间；每 5 小时启动间隔、互斥锁、单次 3 小时上限及明确额度耗尽停止标记保持。上一条关于 2026-10-28 停止的记录已被此条取代。证据：`ops/unattended_research.sh`、`ops/unattended_research_prompt.md`。
- 2026-10-22 23:06 +0800：按用户进一步澄清，将每 5 小时研发唤醒改为先发起一次小型 Codex 真实调用检查额度；仅当返回成功且最后消息严格为 READY 才运行研究任务。额度耗尽写停止标记，网络/认证等其他探测失败只跳过当前周期并留日志；登录状态不被当作额度证明。真实探测自检成功，`bash -n` 通过，crontab 仍每小时触发包装器并由 18,000 秒门控。证据：`ops/unattended_research.sh`、`runs/unattended_research/probe_smoke_last.txt`、`crontab -l`。
- 2026-10-23 11:14 +0800：审计无人值守计划发现 cron 于 00:00/05:00/10:00 触发，但三个 Codex 探测均因 cron 未继承交互 shell 的本地代理而 120 秒超时，研发任务未启动；不是额度耗尽。为定时脚本显式配置服务器现有 127.0.0.1:10809 代理，并增加仅供人工恢复错过周期的 `--force` 参数；使用近似 cron 的空环境进行真实 Codex 探测，成功返回 READY。证据：`runs/unattended_research/20261023T{000001,050001,100001}+0800.*`、`runs/unattended_research/cron_proxy_probe_last.txt`、`ops/unattended_research.sh`。
- 2026-10-23 11:15 +0800：从近似 cron 的空环境经 tmux `tetris-unattended-research` 强制补跑一次漏掉的五小时任务（包装器 PID 3367976），额度探测真实返回 READY，随后 Codex 研发进程 PID 3368261 启动并仍运行；8080/8081 健康检查均 200。定时入口调整为每 5 分钟检查一次本地时间门控，以使实际研发启动间隔接近 5 小时，而不是整点门控造成最多 6 小时间隔；文件锁仍防重复。证据：`runs/unattended_research/20261023T111402+0800.{probe.log,probe.reply,log}`、`crontab -l`、`tmux list-sessions`。
- 2026-10-23 11:17 +0800：实时复核发现 11:14 补跑的 Codex 任务虽完成额度探测并进入项目，但于 11:15 误将自身 tmux/进程视为其他正在运行的研发作业，未执行新里程碑便退出（exit 0）；已修正无人值守提示，明确排除当前包装器与当前 Codex 自身，且标明 1.433× 基准已完成。此条纠正上一条“仍运行”的时点描述；下一步需要重新补跑验证。证据：`runs/unattended_research/20261023T111402+0800.log`、`ops/unattended_research_prompt.md`。
2026-10-23 11:19 +0800：冻结 2000 步历史 seed 的 PPO 评测性能剖析协议；模型哈希已核验，正式 Core/模型未改。证据：runs/ppo_eval_profile_20261023/protocol.json。
2026-10-23 11:19 +0800：性能剖析首次启动失败，研究 worktree 无本地 .venv，尚无评测步；改用主检出既有虚拟环境。证据：本次命令退出码 127。
2026-10-23 11:20 +0800：针对剖析瓶颈调整评测专用合法落点 BFS，在已访问状态上跳过重复碰撞查询；开始原版/改版配对验证。证据：training/evaluation/fast_legal.py、runs/ppo_eval_profile_20261023/profile.txt。
2026-10-23 11:21 +0800：定向环境测试 13 项通过；Parity 因当前非交互 PATH 缺少 node 而无法启动，未发现算法断言失败；改用已安装 Node 的绝对路径重跑。
2026-10-23 11:21 +0800：完成已访问状态碰撞检查优化。两历史 seed 的八条 1000 步评测动作与最终状态逐组完全一致；旧/新总耗时 9.670/9.145 秒（1.057×）。定向与 Parity 16 项及 29 子场景通过。证据：runs/ppo_eval_profile_20261023/{paired.json,tests.log}、reports/experiments/ppo_eval_visited_state_results.md。
- 2026-10-23 12:27 +0800：按用户再次澄清，把无人值守策略改为“额度可用时连续执行有界研发里程碑，只有额度耗尽后等待 18,000 秒再探测”；其他连接/任务错误等待 900 秒；无安全有价值工作时等待五小时避免空转。cron 每五分钟只充当 tmux 看门狗，worker 在服务器持续运行并通过文件锁防重叠；仍保留单轮最多 3 小时。现有工作树已清洁，下一步启动新循环验证。证据：`ops/unattended_research.sh`、`ops/unattended_research_prompt.md`、`crontab -l`。
2026-10-23 12:27 +0800：冻结 3 个历史快照、5 次交替、每次 1000 次特征计算的配对协议；修改评测专用候选特征满行检查，仅检查落子触及行。正式 Core/模型未改。证据：runs/ppo_feature_rows_20261023/benchmark.py、training/evaluation/fast_candidate_features.py。
2026-10-23 12:27 +0800：评测特征满行检查优化在三份历史快照上输出逐项一致，但 5 次交替配对合计旧版 6.732 秒、新版 6.761 秒（0.996×）；无可信收益，已撤回代码，仅保留负结果报告。证据：runs/ppo_feature_rows_20261023/{protocol.json,result.json}、reports/experiments/ppo_feature_touched_rows_result.md。
2026-10-23 12:29 +0800：冻结三份历史快照的合法落点搜索配对微基准；仅试验评测专用 BFS 的冗余访问集检查，正式 Core/模型不变。证据：runs/ppo_legal_visited_20261023/protocol.json。
2026-10-23 12:30 +0800：三份历史快照配对微基准合法落点输出逐项一致，旧/新 CPU 加速 1.035×、1.013×、1.007×，后两份低于预定 1.03× 门槛；撤回评测专用代码改动，仅保留负结果。首次导入路径失败后修正，不影响结果。证据：runs/ppo_legal_visited_20261023/{protocol.json,result.json}、reports/experiments/ppo_legal_redundant_visited_result.md。
2026-10-23 12:32 +0800：完成已归档 PPO 分支吞吐基准的只读完整性审计：3 案例、18 次计时及 1.432843933929417× 汇总均一致；临时副本篡改被检出，无重跑或模型/服务更改。证据：training/evaluation/audit_branch_benchmark.py、reports/experiments/ppo_branch_throughput_integrity.md、runs/ppo_branch_throughput_benchmark/{protocol.json,summary.json}。
2026-10-23 12:34 +0800：冻结三个历史快照的评测专用碰撞缓存移除配对协议，保存原版；仅修改评测辅助模块，开始验证，正式 Core/模型不变。证据：runs/ppo_collision_cache_20261023/{protocol.json,fast_legal_before.py,benchmark.py}。
2026-10-23 12:34 +0800：评测 BFS 碰撞缓存移除实验完成。三个历史快照合法落点完整输出一致，但新版速度仅为原版的 0.869×/0.914×/0.891×；未达 1.03× 门槛，已撤回代码。首次导入路径失败后设置 PYTHONPATH 重跑，不计入计时。证据：runs/ppo_collision_cache_20261023/{protocol.json,result.json}、reports/experiments/ppo_collision_cache_result.md。
2026-10-23 12:36 +0800：冻结三快照五次交替配对协议；开始评测专用井深计算循环试验，正式 Core/模型不变。证据：runs/ppo_feature_wells_20261023/protocol.json、training/evaluation/fast_candidate_features.py。
2026-10-23 12:37 +0800：评测专用井深循环优化完成；三快照候选矩阵/mask 完全一致，10 项定向测试通过；孤立特征旧/新 6.971/6.095 秒（1.144×），两条冻结 PPO 1000 步完整评测结果一致，旧/新 4.974/4.862 秒（1.023×）。保留改动；正式 Core、模型、网页未改。证据：runs/ppo_feature_wells_20261023/{protocol.json,result.json,tests.log,full_eval_protocol.json,full_eval_result.json}、reports/experiments/ppo_feature_wells_result.md。
2026-10-23 12:39 +0800：冻结三历史快照、五次交替各 1000 调用的评测特征方块编号查找配对协议；仅试改评测辅助模块，正式 Core/模型不变。证据：runs/ppo_feature_piece_index_20261023/protocol.json、training/evaluation/fast_candidate_features.py。
2026-10-23 12:40 +0800：三快照方块编号预计算配对完成，候选矩阵/mask 完全一致，旧/新 5.948/5.872 秒（1.013×），低于预定 1.03× 门槛；撤回评测代码。首次导入路径失败后设置 PYTHONPATH 重跑，未计入计时。证据：runs/ppo_feature_piece_index_20261023/{protocol.json,result.json}、reports/experiments/ppo_feature_piece_index_result.md。
2026-10-23 12:41 +0800：冻结三历史快照、五次交替各 1000 调用的评测特征行列位图单扫描协议；正式 Core/模型/网页不变。证据：runs/ppo_feature_board_scan_20261023/protocol.json。
2026-10-23 12:42 +0800：评测特征行列位图单扫描完成，三快照矩阵/mask 逐项一致；旧/新合计 5.920/5.704 秒（1.038×），各快照未退化，12 项定向测试通过。保留评测辅助改动；正式 Core/模型/网页不变。证据：runs/ppo_feature_board_scan_20261023/{protocol.json,result.json}、reports/experiments/ppo_feature_board_scan_result.md。
2026-10-23 12:44 +0800：冻结评测特征方块行列单扫描的端到端配对协议；两历史 seed、各 1000 步、三次交替，先核验冻结模型/源码哈希，再比较完整结果。正式 Core/模型/网页不变。证据：runs/ppo_feature_board_scan_20261023/full_eval_protocol.json。
2026-10-23 12:45 +0800：特征扫描端到端配对完成，六对结果逐字段一致，旧/新 14.552/14.477 秒（1.005×）；收益不足，撤回该处评测辅助代码。定向测试 12 项通过；8080 根路径和 8081 `/api/health` 均 200。无训练、正式 Core/模型/网页改动。证据：runs/ppo_feature_board_scan_20261023/{full_eval_protocol.json,full_eval_result.json}、reports/experiments/ppo_feature_board_scan_full_eval.md。
2026-10-23 12:49 +0800：冻结三历史快照 PPO 快速反事实分支的下一观察复用配对协议，试改评测专用代码；九对完整输出一致，旧/新总耗时 19.888/19.722 秒（1.008×），低于预设 1.03× 门槛，已撤回代码。相关测试 6 项通过，另 2 项因本 worktree 缺少历史轨迹文件未运行成功；8080 根路径和 8081 健康端点均 200。无新增训练步、正式 Core/模型/网页改动。证据：runs/ppo_branch_observation_reuse_20261023/{protocol.json,result.json,benchmark.py}、reports/experiments/ppo_branch_observation_reuse_result.md。
2026-10-23 12:51 +0800：冻结评测专用候选特征双、三、四行消除差分验证协议；仅增加确定性正确性测试，正式 Core、模型、网页不变。证据：runs/ppo_feature_multiline_parity_20261023/protocol.json、tests/test_fast_candidate_env.py。
2026-10-23 12:52 +0800：评测候选特征双、三、四行消除差分验证完成；完整矩阵/mask 一致，定向测试 13/13 通过，8080/8081 均 HTTP 200。无训练步或性能收益主张。证据：runs/ppo_feature_multiline_parity_20261023/tests.log、reports/experiments/ppo_feature_multiline_parity_result.md。
2026-10-23 12:54 +0800：冻结评测专用合法落点搜索的边界状态差分协议：7 种方块、4 种固定棋盘、3 种 hold 状态，共 84 个公开状态；逐项比较正式 Core 的完整落点字典，随后运行现有定向测试。正式 Core/模型/网页不变。证据：`reports/experiments/ppo_fast_legal_edge_parity_design.md`。
2026-10-23 12:54 +0800：评测合法落点边界差分完成：7 方块 × 4 固定棋盘 × 3 hold 状态共 84 例与正式 Core 的有序完整落点字典一致；现有定向测试合计 97/97 通过（2.47 秒），8080/8081 均 HTTP 200。只新增回归覆盖，无训练步或性能收益主张。证据：`runs/ppo_fast_legal_edge_parity_20261023/tests.log`、`reports/experiments/ppo_fast_legal_edge_parity_result.md`。
2026-10-23 12:56 +0800：冻结评测候选特征边界与历史快照差分协议；计划比较 84 个固定边界状态和 3 个已归档开发快照的完整矩阵与掩码，正式 Core/模型/网页不变。证据：`reports/experiments/ppo_fast_feature_edge_parity_design.md`。
2026-10-23 12:57 +0800：评测候选特征差分完成：84 个固定边界状态及 3 个已归档开发快照的 1840 动作完整矩阵/掩码均与正式参考实现一致；固定状态中空落点 0 例，定向测试 97/97 通过（2.68 秒），8080/8081 均 HTTP 200。无新增训练步、模型或正式 Core 改动。证据：`runs/ppo_fast_feature_edge_parity_20261023_{tests.log,snapshots.json}`、`reports/experiments/ppo_fast_feature_edge_parity_result.md`。
2026-10-23 13:00 +0800：冻结两份既有真实终局 PPO 分支快照与归档输出的完整一致性协议，准备比较旧版与评测专用快速重放；无新训练、正式 Core/模型/网页修改。证据：`runs/ppo_fast_branch_terminal_20261023/protocol.json`。
2026-10-23 13:01 +0800：终局差分首次启动在归档 horizon 身份断言处停止，未运行分支；原协议误把实际死亡长度 500 当作分支上限，归档上限实为 10000。保留 v1 并在重放前冻结修订 v2。证据：`runs/ppo_fast_branch_terminal_20261023/{protocol_v1.json,protocol.json}`。
2026-10-23 13:01 +0800：修订协议下两份既有真实终局分支的旧版、快速版及归档完整输出逐字段相同，均在 500 块死亡；单次耗时分别为 1.691/1.083 秒、1.696/1.076 秒，仅作为诊断，不更新正式吞吐估计。证据：`runs/ppo_fast_branch_terminal_20261023/result.json`、`reports/experiments/ppo_fast_branch_terminal_parity.md`。
2026-10-23 13:03 +0800：冻结并完成 21 个阻塞出生棋盘的评测专用空合法落点差分：7 方块 × 3 hold 状态的正式 Core 落点均为空，快速落点、候选矩阵与掩码逐项一致；包含既有边界状态的定向测试 105/105 通过，8080/8081 均 HTTP 200。只增加回归测试，无训练步、正式 Core/模型/网页改动。证据：`runs/ppo_empty_legal_20261023/{protocol.json,tests.log}`、`reports/experiments/ppo_empty_legal_parity.md`。
2026-10-23 13:05 +0800：冻结并开始评测合法落点旋转与边墙差分验证：7 方块 × 4 朝向 × 3 横坐标 × 2 棋盘，共 168 个状态；只新增正确性回归，正式 Core/模型/网页不变。证据：`reports/experiments/ppo_fast_legal_rotation_parity_design.md`、`tests/test_fast_candidate_env.py`。
2026-10-23 13:06 +0800：旋转与边墙差分完成：168 个状态的评测快速合法落点与正式 Core 完整字典一致；新增及相邻既有边界测试共 161/161 通过（1.92 秒），8080/8081 均 HTTP 200。仅增加回归测试，无训练步、正式 Core/模型/网页改动。证据：`tests/test_fast_candidate_env.py`、`reports/experiments/ppo_fast_legal_rotation_parity_result.md`、本次 pytest 输出。
2026-10-23 13:08 +0800：冻结 12 个开发 seed、每局最多 40 步的正式 Core 可达状态差分协议；新增评测快速合法落点与候选特征逐状态一致性测试，尚待运行。证据：`reports/experiments/ppo_reachable_state_parity_design.md`、`tests/test_fast_candidate_env.py`。
2026-10-23 13:09 +0800：12 条合法轨迹产生 272 个正式 Core 可达状态、12,186 个合法动作；快速搜索的完整落点及候选矩阵/掩码逐状态一致，相关测试 175/175 通过（4.55 秒）。8080/8081 均 HTTP 200。无训练步、模型或正式 Core 改动。证据：`runs/ppo_reachable_state_parity_20261023_tests.log`、`reports/experiments/ppo_reachable_state_parity_result.md`。
- 2026-10-23 14:00 +0800：无人值守循环至 13:10 已连续完成若干性能与差分里程碑后，因提示缺少下一项具体研究而输出 `NO_SAFE_WORK`，进入五小时等待；未耗尽额度、当前无 Codex 研发进程。依据既有生存优先失败诊断和后续报告的明确建议，新增有预算上限的“事前采样清行节奏失稳反事实数据”任务：最多 8 个新 seed×20K、每 seed 4 状态、每状态两条 3K 分支，采样与停止门槛事前冻结；不训练、不使用预留确认种子。证据：`runs/unattended_research/status.log`、`ops/unattended_research_prompt.md`、`reports/experiments/ppo30m_survival_balanced_failure_diagnostic.md`。
- 2026-10-23 14:02 +0800：核查 worktree 与主数据根目录的 1550000..1550007 文本用途，仅任务提示命中；确认无独立冲突作业，8080/8081 监听。事前冻结清行节奏失稳与已恢复对照的抽样、候选动作、预算和停止门槛，尚未运行新 seed 或分支。证据：`reports/experiments/ppo_clear_cadence_counterfactual_design.md`；本次 `rg`、`ps`、`tmux`、`ss` 输出。
2026-10-23 14:04 +0800：开始按已冻结协议实现并运行 seed 1550000 的事前 PPO 采样；模型单线程、20K 上限，快照与整局结果原子保存。证据：training/evaluation/ppo_clear_cadence_capture.py、runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:05 +0800：首轮采样命令因研究 worktree 无本地 .venv 而未启动；改用主项目虚拟环境重试，未产生新 seed 结果。证据：/tmp/ppo_clear_cadence_1550000.log。
2026-10-23 14:06 +0800：完成首个开发 seed 1550000 的事前采样：20,000 块触顶，失稳 1 状态、危险后恢复对照 2 状态；三个快照 SHA256 复核通过，重复调用只校验不重算。未运行反事实分支、无训练步。证据：runs/ppo_clear_cadence_counterfactual/episodes/seed_1550000.json、states/、reports/experiments/ppo_clear_cadence_capture_seed1550000.md。
2026-10-23 14:08 +0800：开始对已事前采集的 seed 1550000 三个状态运行冻结 PPO 双分支反事实（每支至多 3000 块，单线程）；候选、分支和比较各自原子保存，恢复时校验身份和哈希。证据：training/evaluation/ppo_clear_cadence_branches.py、runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:09 +0800：首次提交推送组合命令被自动审批拒绝，原因是尚未核实 origin 归属；已只读核查 origin 为既有 `git@github.com:u-wyh/tetris-ai-benchmark.git`，当前分支跟踪同名远端。未发生远端写入。证据：git remote -v、git branch -vv。
2026-10-23 14:11 +0800：开始按冻结协议采集开发 seed 1550001，冻结 PPO、单线程、20K 上限；整局和快照原子保存。证据：runs/ppo_clear_cadence_counterfactual/episodes/seed_1550001.json。
2026-10-23 14:12 +0800：开发 seed 1550001 采样完成，20,000 块触顶；失稳 0 状态、已恢复对照 2 状态。快照哈希与恢复调用核验通过；无反事实分支或训练步。证据：runs/ppo_clear_cadence_counterfactual/episodes/seed_1550001.json、reports/experiments/ppo_clear_cadence_capture_seed1550001.md。
2026-10-23 14:14 +0800：按已冻结协议启动 seed 1550001 两个已采样状态的 PPO 原动作/固定合法替代动作配对分支，每支最多 3000 块，单线程且支持原子恢复。证据：runs/ppo_clear_cadence_counterfactual/、/tmp/ppo_clear_cadence_1550001_branches.log。
2026-10-23 14:14 +0800：seed 1550001 两状态四分支完成，均达到 3000 块；共同窗口一处替代动作少 287,500 分及 8 次四消，另一处相同，寿命均未定。候选、分支及比较文件已保存；重复调用核验并直接返回相同结果。无训练步。证据：`reports/experiments/ppo_clear_cadence_seed1550001_branches.md`、`runs/ppo_clear_cadence_counterfactual/{candidates,branches,comparisons}/seed_1550001*`。
2026-10-23 14:17 +0800：按冻结清行节奏协议启动开发 seed 1550002 的 20K 上限采样，随后仅对入选状态运行每支最多 3K 的双分支；单线程、原子记录。证据：runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:18 +0800：seed 1550002 采样和两状态四分支完成；原子记录经重复调用校验，两个对照均双支触顶，得分和四消方向相反；8080/8081 均 HTTP 200。累计三个独立 episode，失稳状态仅见于 seed 1550000；无训练或确认种子使用。证据：runs/ppo_clear_cadence_counterfactual/{episodes,states,candidates,branches,comparisons}/seed_1550002*、reports/experiments/ppo_clear_cadence_seed1550002_results.md。
2026-10-23 14:20 +0800：按冻结协议启动开发 seed 1550003，采集上限 20K，入选状态双分支各上限 3K；单线程，原子记录和恢复。证据：runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:22 +0800：开发 seed 1550003 采样及两状态四分支完成；20K 触顶，失稳 0、恢复对照 2；双支均 3K 触顶，替代动作共同窗口得分与四消均较低，寿命未定。重复调用校验并返回已有记录；8080 首页及 8081 /api/health 均 HTTP 200。无训练步或确认种子使用。证据：reports/experiments/ppo_clear_cadence_seed1550003_results.md、runs/ppo_clear_cadence_counterfactual/{episodes,states,candidates,branches,comparisons}/seed_1550003*。
2026-10-23 14:24 +0800：按冻结协议启动开发 seed 1550004 的 20K 上限采样及入选状态双分支，每支至多 3K；使用既有原子恢复程序，单线程。证据：runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:26 +0800：开发 seed 1550004 采样与两状态四分支完成；20K 触顶，失稳 0、恢复对照 2，双支均 3K 触顶。共同窗口替代动作得分差 +56,700/0，四消差 +2/0，寿命未定；重复调用核对已有原子记录，8080/8081 均 HTTP 200。无训练步或确认种子使用。证据：reports/experiments/ppo_clear_cadence_seed1550004_results.md、runs/ppo_clear_cadence_counterfactual/{episodes,states,candidates,branches,comparisons}/seed_1550004*。
2026-10-23 14:27 +0800：按冻结协议启动开发 seed 1550005 的 20K 上限采样及入选状态双分支（每支最多 3K）；单线程，原子保存并支持恢复。证据：runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:30 +0800：开发 seed 1550005 采样和三状态六分支完成；20K 触顶，失稳 1、恢复对照 2，双支均 3K 触顶，寿命未定。重复调用逐字节一致，8080/8081 均 HTTP 200；无训练步或确认种子使用。证据：reports/experiments/ppo_clear_cadence_seed1550005_results.md、runs/ppo_clear_cadence_counterfactual/{episodes,states,candidates,branches,comparisons}/seed_1550005*。
2026-10-23 14:31 +0800：按已冻结协议启动第七个开发 seed 1550006 的采样及入选状态双分支；采集至多 20K、每支至多 3K，单线程、原子保存并支持恢复。证据：runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:33 +0800：开发 seed 1550006 采样及四状态八分支完成；20K 采样及各分支 3K 均触顶，失稳 2、已恢复对照 2，寿命未定。恢复调用输出逐字节一致，8080/8081 均 HTTP 200。无训练步或确认种子使用。证据：reports/experiments/ppo_clear_cadence_seed1550006_results.md、runs/ppo_clear_cadence_counterfactual/{episodes,states,candidates,branches,comparisons}/seed_1550006*。
2026-10-23 14:35 +0800：按冻结协议启动最后一个开发 seed 1550007 的 20K 上限采样及入选状态双分支（每支最多 3K）；单线程、原子保存并支持恢复。证据：runs/ppo_clear_cadence_counterfactual/。
2026-10-23 14:38 +0800：完成 seed 1550007 四状态八分支，并汇总全部 8 个开发 episode：共 157,795 采样块、22 状态、44 分支；独立 seed 中明确救援 0/8、明确误伤 1/8，其余寿命均未定，得分与四消方向混杂，按冻结门槛判为证据不足并停止扩样。恢复调用逐字节一致；8080/8081 均 HTTP 200。无训练步、确认种子或正式规则/模型改动。证据：reports/experiments/ppo_clear_cadence_counterfactual_results.md、runs/ppo_clear_cadence_counterfactual/、/tmp/ppo_clear_cadence_1550007_{capture,branches}*.log。
2026-10-23 14:41 +0800：对已保存清行节奏反事实记录运行独立只读完整性审计；8 局、22 状态、44 分支的哈希/身份/序列/比较均通过，重新汇总救援 0、误伤 1。未重跑 PPO、训练或改动历史数据。证据：training/evaluation/audit_clear_cadence_records.py、reports/experiments/ppo_clear_cadence_record_audit.md。
2026-10-23 19:44 +0800：核查已完成清行节奏反事实数据的代码来源：采样与分支脚本自各自首次提交后无修改，所查主要依赖自首个脚本提交后无后续提交；运行前源码哈希未保存，不能据此证明历史执行字节或提升证据等级。未重跑评测或训练。证据：reports/experiments/ppo_clear_cadence_code_provenance.md、git diff/log、sha256sum。
2026-10-24 00:48 +0800 | 为清行节奏后续新采样/分支记录增加执行源码与依赖版本清单，写盘前和恢复时核对；旧数据不补造来源、不重跑。定向编译和篡改哈希拒绝检查通过；旧记录只读审计通过（8 局、22 状态、44 分支）。证据：`reports/experiments/ppo_clear_cadence_future_provenance.md`、`training/evaluation/source_provenance.py`。
2026-10-24 00:50 +0800 | 独立清行节奏记录审计增加新记录的执行来源清单校验；既有 8 局、22 状态、44 分支审计通过，内存篡改 Core 哈希被拒绝，8080/8081 均 HTTP 200。未重跑评测或训练，旧记录不补造来源。证据：`training/evaluation/audit_clear_cadence_records.py`、`reports/experiments/ppo_clear_cadence_audit_provenance.md`。
2026-10-24 00:52 +0800 | 清行节奏历史记录只读审计增加快照恢复后的正式 Core 合法落点、环境掩码与候选集合三方核验；22/22 状态通过，44 分支汇总仍为 0 救援、1 误伤；8080/8081 均 HTTP 200。未重跑 PPO 或训练。证据：`reports/experiments/ppo_clear_cadence_candidate_legality_audit.md`、`training/evaluation/audit_clear_cadence_records.py`。
2026-10-24 05:55 +0800 | 修复清行节奏采样快照在中断后因 gzip 时间戳导致的误冲突：新快照使用固定时间戳，已有快照按解压后字节校验并保留原哈希。定向测试 2 项、历史记录只读审计（8 局、22 状态、44 分支）通过，8080/8081 均 HTTP 200；未运行新评测或训练。证据：`tests/test_ppo_clear_cadence_snapshot.py`、`reports/experiments/ppo_clear_cadence_snapshot_resume_fix.md`。
2026-10-24 15:59 +0800 | 开始对既有清行节奏快照的冻结 PPO 快速反事实续控做单线程 1000 块剖析；历史分支只读，证据：runs/ppo_branch_profile_20261024/。
2026-10-24 16:00 +0800 | 完成既有快照的单线程 1000 块快速反事实剖析：逐块得分与四消匹配归档前缀；合法落点 2.95 秒、候选特征 1.26 秒（cProfile 总计 4.66 秒）。8080/8081 均 HTTP 200；无新种子、训练或正式代码改动。证据：runs/ppo_branch_profile_20261024/、reports/experiments/ppo_fast_branch_profile_20261024.md。
2026-10-24 16:01 +0800 | 开始跨 3 个已归档独立 episode 的冻结 PPO 快速分支单线程剖析，每支限 1000 块；只读取原数据并对照逐块归档前缀，剖析结果写 runs/ppo_branch_profile_replication_20261024/。
2026-10-24 16:02 +0800 | 跨 3 个独立 episode 的冻结 PPO 1000 块剖析完成；三条逐块得分及四消均与历史分支前缀一致，合法落点累计 2.99–3.26 秒、候选特征 1.25–1.27 秒，8080/8081 均 HTTP 200。首次命令缺 PYTHONPATH 未运行，补齐后成功；无新种子或训练。证据：`runs/ppo_branch_profile_replication_20261024/`、`reports/experiments/ppo_fast_branch_profile_replication_20261024.md`。
2026-10-24 16:04 +0800 | 评测合法落点预位移掩码实验：175 项差分测试与 22 个归档快照完整落点比较通过；6 轮配对函数计时中位仅 1.011×，撤回实现并记录负结果。首次恢复快照因环境上限不匹配被拒绝，修正基准环境后完成；8080/8081 均 HTTP 200。无新 seed、训练或正式 Core/模型改动。证据：`reports/experiments/ppo_fast_legal_shifted_masks_20261024.md`。
2026-10-24 16:06 +0800 | 对既有三个独立 episode 的 PPO 反事实分支 cProfile 原件做只读热点细分；`_branch` 自身约占合法落点累计耗时 55%，碰撞和旋转各约 10%。未重跑评测、训练或修改代码；证据：`reports/experiments/ppo_fast_branch_hotspot_audit_20261024.md`、`runs/ppo_branch_profile_replication_20261024/*.prof`。
2026-10-24 16:09 +0800 | 评估离线快速合法落点搜索的重复 visited 查询消除：175 项差分测试、22 个归档快照完整落点一致；8 轮交替计时中位仅 1.001×，撤回代码改动。无新 seed、训练或正式规则变动。证据：`reports/experiments/ppo_fast_legal_duplicate_lookup_20261024.md`。
2026-10-24 16:11 +0800 | 在临时副本评估离线合法落点碰撞缓存改用字典：22 个归档状态完整落点一致，8 轮交替计时原实现/字典版中位为 0.927×；因减速未采纳。8080/8081 均 HTTP 200；无新 seed、训练或正式代码变更。证据：`reports/experiments/ppo_fast_legal_collision_cache_20261024.md`、`/tmp/bench_collision_dict.py`。
2026-10-24 16:13 +0800 | 开始有界评测实验：在临时副本试验快速合法落点搜索的整数 visited 状态，使用既有 22 个清行节奏快照做完整落点差分和交替计时；不运行 PPO 新分支或训练。证据计划：runs/ppo_fast_legal_integer_visited_20261024/。
2026-10-24 16:13 +0800 | 完成整数 visited 评测实验：22 个归档状态完整落点一致，8 轮交替计时中位原版 0.7177 秒、整数版 0.8149 秒，因减速未采纳；8080/8081 均 HTTP 200。无新种子、训练或正式实现改动。证据：`reports/experiments/ppo_fast_legal_integer_visited_20261024.md`、`runs/ppo_fast_legal_integer_visited_20261024/`。
2026-10-24 16:16 +0800 | 在临时源码副本消融快速合法落点的碰撞缓存：22 个归档状态的完整落点一致；8 轮交替计时原版/无缓存版中位 0.7294/0.8019 秒，移除缓存减速，未采纳。8080/8081 均 HTTP 200；无新种子、训练或正式实现改动。证据：`reports/experiments/ppo_fast_legal_no_collision_cache_20261024.md`、`runs/ppo_fast_legal_no_collision_cache_20261024/`。
2026-10-24 16:18 +0800 | 补全新清行节奏记录执行来源清单，纳入实际使用的快速合法落点与候选特征源码；两项篡改拒绝测试、历史记录只读审计通过，8080/8081 均 HTTP 200。无新评测或训练，历史来源不补造。证据：`reports/experiments/ppo_clear_cadence_fast_source_provenance_20261024.md`、`tests/test_clear_cadence_source_provenance.py`。
2026-10-24 16:21 +0800 | 补全新清行节奏记录的执行来源清单，纳入实际继承并调用的 `candidate_env.py` 和 `fast_candidate_env.py`；定向篡改拒绝测试 4/4 通过，历史记录只读审计为 8 局、22 状态、44 分支、救援 0、误伤 1，8080/8081 均 HTTP 200。无新评测、训练或确认种子使用。证据：`reports/experiments/ppo_clear_cadence_env_source_provenance_20261024.md`、`tests/test_clear_cadence_source_provenance.py`。
2026-10-24 16:24 +0800 | 只读审计三个既有独立 episode 的候选特征 profile：每 1000 块累计 1.246–1.275 秒，占 profile 总时间 25.2%–27.1%；未重跑评测、训练或改正式代码。证据：reports/experiments/ppo_fast_feature_profile_audit_20261024.md、runs/ppo_branch_profile_replication_20261024/*.prof。
2026-10-24 16:27 +0800 | 优化评测专用候选特征值域校验：22 个归档状态完整矩阵/掩码一致，8 轮交替计时中位 0.2306→0.2036 秒（1.132×），定向测试 177 项通过；无训练步或新种子。证据：`reports/experiments/ppo_fast_feature_limits_20261024.md`、`runs/ppo_fast_feature_limits_20261024/result.json`。
2026-10-24 16:29 +0800 | 候选特征完整分支基准首次启动失败：当前 worktree 无 `.venv/bin/python`，未执行评测；改用主项目虚拟环境。证据：命令退出码 127。
2026-10-24 16:30 +0800 | 启动有界成对基准：三个既有开发 episode 归档状态，旧/新特征值域实现在相同冻结 PPO 快速分支各运行 2 次、每支最多 500 块，单线程；协议先原子保存。结果待写入 `runs/ppo_fast_feature_limits_branch_20261024/`，不使用新种子或训练步。
2026-10-24 16:31 +0800 | 完成候选特征完整分支复核：三个独立归档 episode 共六对，分支结果逐项一致；旧版 6.631 秒、新版 6.473 秒，吞吐 1.024×。语法与 diff 检查通过，8080/8081 均 HTTP 200；无训练步、新种子或正式规则变更。证据：`runs/ppo_fast_feature_limits_branch_20261024/summary.json`、`reports/experiments/ppo_fast_feature_limits_branch_20261024.md`。
2026-10-24 16:32 +0800 | 完成候选特征完整分支基准只读完整性审计：3 个归档 episode、6 组成对计时的模型/源码/输入/结果哈希及汇总值一致；无新评测、训练或 seed。证据：`training/evaluation/audit_feature_limits_branch.py`、`reports/experiments/ppo_feature_limits_branch_integrity_audit_20261024.md`。
2026-10-24 16:34 +0800 | 启动评测候选特征棋盘扫描合并实验：临时实现一次遍历同时构造行列位掩码；用 22 个归档状态做完整矩阵差分及 8 轮交替计时，按实测结果决定保留或撤回。无新种子、分支续控或训练。证据：`/tmp/bench_feature_single_pass.py`、`/tmp/fast_candidate_features_single_pass_baseline.py`。
2026-10-24 16:35 +0800 | 完成评测候选特征棋盘扫描合并：22 个归档状态完整矩阵/掩码相同、定向测试 175/175；函数交替计时约 1.026×，三个独立归档 episode 的六对完整分支输出相同，累计吞吐约 1.009×。无新种子、训练或正式规则改动。证据：`reports/experiments/ppo_fast_feature_single_pass_20261024.md`、`runs/ppo_fast_feature_single_pass_branch_20261024/`。
2026-10-24 16:36 +0800 | 补全未来清行节奏评测记录的来源清单，纳入导入链中的 hybrid_policy、ppo_lookahead_v3_audit 和 tetris_core/__init__.py；新增篡改拒绝测试，定向测试 7/7、diff 检查通过。旧记录未修改，无新评测或训练。证据：`reports/experiments/ppo_clear_cadence_provenance_import_gap_20261024.md`、`tests/test_clear_cadence_source_provenance.py`。
2026-10-24 16:39 +0800 | 启动评测专用合法落点搜索冗余 visited 查询的有界成对分支实验：三个已归档独立 episode，各 500 块，旧/新各两次交替，单 CPU 线程；先做 Core 对照测试 175/175 通过，协议及分支结果保存于 `runs/ppo_fast_legal_visited_20261024/`。无新种子或训练步。
2026-10-24 16:40 +0800 | 完成该实验：六对冻结 PPO 完整分支输出相同，旧版 6.455 秒、候选 6.459 秒，速度比 0.9993×；无可见收益，已撤回评测代码改动。候选源码、原子记录和可恢复脚本保留；8080/8081 均 HTTP 200。证据：`reports/experiments/ppo_fast_legal_visited_20261024.md`、`runs/ppo_fast_legal_visited_20261024/`。无新种子或训练步。
2026-10-24 16:42 +0800 | 启动评测专用合法落点终局占格排序消除实验：按方块预排序相对坐标，在三个既有开发 episode 的归档状态上执行 Core 差分及有界成对分支基准；无新种子或训练。证据：runs/ppo_fast_legal_landing_sort_20261024/。
2026-10-24 16:42 +0800 | 完成评测专用终局占格预排序实验：三个独立归档 episode 六对 500 块分支输出一致，旧/新耗时 6.6204/6.3563 秒（1.0415×）；采用改动后 175 项 Core 差分通过，8080/8081 均 HTTP 200。无新种子、训练或正式规则变动。证据：reports/experiments/ppo_fast_legal_landing_sort_20261024.md、runs/ppo_fast_legal_landing_sort_20261024/。
2026-10-24 16:45 +0800 | 对已归档落点预排序分支基准做独立只读完整性审计：3 局、6 组成对计时的模型/源码/输入身份及汇总算术一致，吞吐比 1.0415×；8080/8081 均 HTTP 200。未重跑 PPO、训练或使用新种子。证据：`training/evaluation/audit_fast_legal_landing_sort.py`、`reports/experiments/ppo_fast_legal_landing_sort_integrity_audit_20261024.md`。
2026-10-24 16:46 +0800 | 启动评测专用向下碰撞检查复用实验：三个既有独立 episode，各 500 块，旧/新各两次交替；先执行 Core 差分测试，协议和结果写入 runs/ppo_fast_legal_down_collision_20261024/，不使用新种子或训练步。
2026-10-24 16:47 +0800 | 完成评测专用向下碰撞检查复用实验：175 项 Core 差分测试通过，三个独立 episode 六对完整 PPO 分支结果相同，旧/新累计 6.2693/6.0175 秒，吞吐 1.0418×；8080/8081 均 HTTP 200。无新种子、训练步或正式规则变动。证据：reports/experiments/ppo_fast_legal_down_collision_20261024.md、runs/ppo_fast_legal_down_collision_20261024/。
2026-10-24 16:49 +0800 | 新增向下碰撞检查基准归档完整性审计，并修正旧落点排序审计对已更新当前源码的过期断言；两项归档审计通过，速度比分别为 1.041848× 与 1.041547×，8080/8081 均 HTTP 200。未重跑 PPO、训练或使用种子。证据：`reports/experiments/ppo_fast_legal_down_collision_integrity_audit_20261024.md`、`training/evaluation/audit_fast_legal_down_collision.py`。
2026-10-24 16:51 +0800 | 启动评测专用清行检查实验：仅检查本次落块触及的行是否满行，先做 Core 差分和旧/新归档状态计时；未使用新种子或训练步。证据待写入 `reports/experiments/ppo_fast_feature_touched_rows_20261024.md`。
2026-10-24 16:52 +0800 | 完成评测清行检查有界实验：175 项差分通过，3 个独立 episode 的 6 对完整分支输出一致；旧/新累计 6.010125/6.009811 秒（1.00005×），无可见收益，已撤回正式评测代码改动。8080/8081 均 HTTP 200；无新种子、训练步或规则改动。证据：`reports/experiments/ppo_fast_feature_touched_rows_20261024.md`、`runs/ppo_fast_feature_touched_rows_branch_20261024/`。
2026-10-24 16:54 +0800 | 只读复核落块触及行基准归档：3 个 episode、6 组成对计时的模型、源码、快照、候选哈希与汇总算术一致；速度比 1.000052×，结论仍为无可见收益。未重跑分支或训练。证据：`reports/experiments/ppo_fast_feature_touched_rows_integrity_audit_20261024.md`、`training/evaluation/audit_feature_touched_rows_branch.py`。
2026-10-24 16:55 +0800 | 清行节奏归档审计增加磁盘冻结 PPO 文件哈希及逐局模型身份核验；既有 8 局、22 状态、44 分支审计通过，救援 0、误伤 1。仅核查归档，无分支重算或训练。证据：`reports/experiments/ppo_clear_cadence_model_identity_audit_20261024.md`、`training/evaluation/audit_clear_cadence_records.py`。
2026-10-24 16:57 +0800 | 启动当前评测代码三局归档分支剖析：每局原 PPO 动作最多 500 块、单线程；结果写入 runs/ppo_current_branch_profile_20261024/，不使用新种子或训练步。
2026-10-24 16:59 +0800 | 完成当前评测代码三局归档 PPO 分支剖析：各 500 块得分/四消与历史前缀完全一致；合法落点占 profile 总时 62.0%–64.9%，候选特征占 25.3%–27.6%；8080/8081 均 HTTP 200。无新种子、训练步或规则改动。证据：reports/experiments/ppo_current_branch_profile_20261024.md、runs/ppo_current_branch_profile_20261024/。
