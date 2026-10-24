# 评测候选特征清行检查：只看落块触及行

2026-10-24 16:51 +0800。候选实现只在落块触及的行检查新满行，再在有清行时重建棋盘列。正式 Core/JS 和冻结 PPO 未改；未用新 seed、训练步或预留确认组。

`tests/test_fast_candidate_env.py` 的 175 项差分测试全部通过。三个既有独立 episode 的归档快照（seed 1550002/1550005/1550007），原版和候选版每例交替各跑两次、每支最多 500 块，六对完整分支结果逐项相同。原版累计 6.010125 秒，候选 6.009811 秒，速度比 **1.00005×**。该差异小于成对波动，不能证明吞吐收益；因此撤回正式评测路径改动。

原子协议、逐例结果和汇总位于 `runs/ppo_fast_feature_touched_rows_branch_20261024/`。候选源码保留在 `training/evaluation/fast_candidate_features_touched_rows.py`，复核脚本为 `training/evaluation/benchmark_feature_touched_rows_branch.py`。8080 和 8081 均返回 HTTP 200。
