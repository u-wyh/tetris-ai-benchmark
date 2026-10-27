# BC Champion 得分强化学习

状态：**真实训练进行中**。截至 2026-10-27T21:09:59+08:00，已完整提交 65,536 步，正在执行 task 17。此处是启动阶段快照，最新进度以 `runs/bc_score_rl/training/transaction_state.json` 为准。

完整路线、历史区别和事前门槛见 [冻结设计](bc_score_rl_design.md)。第 1、2 个事务分别提交/恢复验证成功；前四个事务 Actor 参数与梯度均为零，后续 Actor/Critic 均有更新。五模型真实合法动作烟测通过；未得出新模型生存或得分改善结论。

冻结 BC SHA256：`5f5613cb5eb4e80a02406b70902ad3d5c571eb8fe38bec4b40f73b34289ca663`。
历史 KL-PPO 2M SHA256：`70794b10776104fb3c7631a7eddcdbac157418eeb4e7e473187b75b8c509d1e1`。

阶段 A 为 131,072 步。之后自动比较 BC、历史 KL-PPO、PPO30M、B12、新候选的 16-seed×20K；通过才进入 100K 和后续最多 1M/3M 训练。失败正常收尾，不追加路线。只有最终冻结模型通过全部开发门槛才运行全新独立确认。

后台 tmux：`tetris-bc-score-rl`。恢复命令：`bash ops/run_bc_score_rl.sh`（会校验冻结源码/模型/事务，原子完成结果不重算）。

目前测试：相关 Python/Parity 26 passed、3 skipped、35 subtests passed；JS 68 passed。新增评测恢复/门槛/共同得分测试 5 passed；原始 BC/PPO 与网页服务保持冻结。完整测试输出在 `runs/bc_score_rl/preflight_*.log`。

本报告由自动队列在实验结束时替换为最终结果，包括各阶段对照、逐 seed 配对、停止原因、checkpoint SHA256 及回归结果。
