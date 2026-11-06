# 独立worker上的critic-only拟合：未通过保留能力门槛

2026-11-06 14:27:15诊断passed，父控制器/保护器14:27:17 completed、MainPID0/Resultsuccess；**eligible_for_ppo_path_study=false**。完成计算不等于方案合格，未采用诊断权重，也未启动后续PPO质量训练。

## 固定协议

复用独立数据4段共16384条，前4worker的8192条用于拟合，后4worker的8192条仅作最终验证，两组实际episode_seed无重叠。只在原策略的独立policy副本中更新value_net：保留原critic Adam状态/lr3e-4、.5×MSE、critic梯度上限.5、batch256、5epoch，共160次更新，固定排列seed2800000。没有筛选/删除失败样本，没选epoch、改学习率或补采样。

源模型policy/Adam/PPO计数未变；副本actor权重及其已有Adam历史逐项不变。**160次更新发生在critic副本**，不能笼统称“0优化器更新”。没有环境推进、采样或普通model.zip；只保存明确诊断用critic权重及其Adam状态，回读精确。

5项受保护测试验证：actor原有非空Adam动量与遗留梯度不会造成误更新，critic实际发生更新；全部可行性门必须同时满足，无终止样本不能通过。72个冻结输入完成后逐项未改。

## 最终结果

误差相对于固定原策略GAE return，单位为RMSE。

| 组 | 源critic | 拟合副本 | 比值 | 预声明上限 |
| --- | ---: | ---: | ---: | ---: |
| 拟合全部8192条 | 7.149210 | 1.784745 | 0.249642 | 不用于选方案 |
| 验证全部8192条 | 7.841563 | 7.230733 | 0.922104 | 0.95，通过 |
| 验证终止前32步，共64条 | 86.324379 | 78.430990 | 0.908561 | 1.00，通过 |
| 验证其余8128条 | 1.816018 | 2.063688 | **1.136381** | **1.05，失败** |

拟合集明显改善，独立验证整体与终止窗口也改善，但其余步骤误差上升13.64%，超过允许的5%。因此未通过全部门槛；不因整体平均变好而忽略其余状态退化，也不调整门槛或继续试LR/epoch/loss。

目标仍包含原critic bootstrap，四个512步GAE窗口不是完整Monte Carlo真值；仅4个验证worker/2次终止，step相关。上述只是拟合/保留能力检查，没有证明策略回报、长局生存或PPO训练质量。单次验证也不能确定一个普遍因果机制。

## 保护与后续

CPU峰57℃、GPU38℃、组内存537.8MiB，全250%无降载，6GiB/swap0/原温控、180秒命令/600秒队列预算不变；实际保护器4.121秒。原30M、main27训练来源、推理20、row masks及训练模板保持冻结。

停止本次critic预拟合路线，不推广critic_diagnostic.pt，不启动与它相关的PPO路径/长局/扩训。不重复已完成采样与梯度诊断。后续开发先核对既有性能工作，选择有尚未验证证据支持的改进；IPC等待包含worker计算，不能据此直接认定通信是瓶颈，fused特征4.91%未过门槛也不能靠重复计时翻案。

证据：runs/autonomous_critic_fit_20261106/design.json、epochs.jsonl、feasibility.json、resource_audit.json、summary.json、unit_guard/和guard/。原生goal轮首工具14:23:51真实执行见native_goal_critic_fit_execution_20261106.json，未冒充新的小时投递。
