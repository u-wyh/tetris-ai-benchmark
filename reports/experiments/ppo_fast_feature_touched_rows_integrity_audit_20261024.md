# 落块触及行特征基准归档完整性复核

2026-10-24 16:54 +0800。只读复核 `runs/ppo_fast_feature_touched_rows_branch_20261024/` 的冻结协议、三个独立 episode 的输入快照与候选文件哈希、冻结模型和旧/新特征及分支源码哈希、六组成对计时顺序，以及汇总文件对逐例记录的哈希和算术。全部通过；归档总耗时旧版 6.010125 秒、新版 6.009811 秒，速度比 1.000052×，仍无可见收益。

复核命令：`PYTHONPATH=. /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python -m training.evaluation.audit_feature_touched_rows_branch`。脚本位于 `training/evaluation/audit_feature_touched_rows_branch.py`。此审计核对归档内部一致性，不能仅凭摘要中的 `outputs_identical` 字段独立证明当时每次分支逐步输出相同；没有重跑 PPO、使用新种子或训练。
