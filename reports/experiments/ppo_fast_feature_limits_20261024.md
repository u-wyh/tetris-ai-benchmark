# 离线 PPO 候选特征值域校验优化

2026-10-24 16:27 +0800。三个独立 episode 的既有 profile 均显示值域校验生成器耗时约 0.116 秒/1000 块。将评测专用 `fast_candidate_features` 的 NumPy 浮点 scale 事先转为 Python 整数上界，循环内比较整数；所有上界均不超过 255，故移除重复的 255 比较。正式 Core/JS、冻结模型、网页默认模型和训练实现未改。

从既有清行节奏实验恢复 22 个归档公开状态，逐状态比较旧版与新版的完整特征矩阵和合法动作掩码，均逐项相同。单线程预热后交替计时 8 轮，每轮对全部 22 状态各调用 25 次：旧版中位 0.2306 秒，新版 0.2036 秒，特征函数加速 **1.132×**；8 轮新版均快于对应旧版。`tests/test_fast_candidate_env.py` 与 `tests/test_fast_counterfactual.py` 共 177 项通过。

计时仅覆盖候选特征函数，不能外推为完整 PPO 分支或训练吞吐；归档数据的来源哈希不修改，现行代码与归档执行源码不同。无新开发或确认 seed、无新训练步。原始计时在 `runs/ppo_fast_feature_limits_20261024/result.json`；一次性实验脚本 `/tmp/bench_feature_limits.py`，旧版源码副本 `/tmp/fast_candidate_features_baseline.py`。
