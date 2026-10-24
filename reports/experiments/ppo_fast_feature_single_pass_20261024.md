# 评测候选特征棋盘扫描合并

2026-10-24 16:35 +0800。既有三个独立 episode 的 profile 显示候选特征累计约占冻结 PPO 快速分支四分之一。评测专用特征生成器原先分别遍历 20×10 棋盘以构造行、列位掩码；本次改为一次遍历同时构造两者。不改正式 Core/JS、训练特征、冻结 PPO 或网页模型。

22 个既有开发快照的完整特征矩阵及合法掩码与改动前逐项一致；定向测试 `tests/test_fast_candidate_env.py` 为 175/175 通过。相同快照的函数交替计时 8 轮，中位数 0.2006→0.1956 秒，约 1.026×。完整冻结 PPO 分支以三个归档 episode 各运行旧/新版本两次，每支 500 块，所有输出一致；旧版累计 6.4970 秒，新版 6.4420 秒，约 **1.009×**。差异很小，不能推断训练吞吐或长局收益。

原始成对结果与哈希：`runs/ppo_fast_feature_single_pass_branch_20261024/`；函数实验脚本与旧版副本：`/tmp/bench_feature_single_pass.py`、`/tmp/fast_candidate_features_single_pass_baseline.py`。未使用新种子、确认组或训练步。
