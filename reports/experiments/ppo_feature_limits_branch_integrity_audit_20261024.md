# 候选特征完整分支基准：归档完整性审计

2026-10-24 16:32 +0800。只读检查既有三条独立 episode 的性能归档：冻结模型、旧版与新版特征源码、分支源码均与冻结协议哈希一致；三个输入快照与候选记录、三个结果文件与汇总哈希一致；计时顺序、次数和汇总数值重新计算一致。共六组成对计时，总吞吐比仍为 1.0244106396449613×。命令：`PYTHONPATH=. /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python -m training.evaluation.audit_feature_limits_branch`。

此审计核对保存文件的身份和内部一致性，不重新执行 PPO 分支，也不能独立证明归档中的 `outputs_identical` 断言；该断言仍来自原基准执行时的逐次结果比较。未使用新 seed、确认种子或训练步。8080/8081 服务未受本轮只读审计影响。
