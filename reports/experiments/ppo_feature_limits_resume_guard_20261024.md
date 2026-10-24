# 候选特征分支基准恢复校验

2026-10-24 22:14 +0800。在既有特征值域优化基准的恢复入口增加逐局校验：归档候选与快照哈希、原动作、结果摘要格式、窗口上限，以及四次计时的顺序和正有限耗时。损坏或混入其他输入的记录会在纳入汇总前被拒绝。

只读验证接受了现有 3 个归档 episode 的逐局记录；分别修改动作、候选哈希、结果哈希或计时后，12 个篡改样例均被拒绝。`git diff --check` 通过。没有重跑 PPO 分支、使用新 seed 或训练。此前完整归档审计在历史 `fast_counterfactual.py` 源码哈希检查处失败：当前分支源码已在归档后更新；本轮没有将当前源码误认为当时的执行源码，也没有改写原始结果。

代码：`training/evaluation/{audit_feature_limits_branch,benchmark_feature_limits_branch}.py`；原始数据：`runs/ppo_fast_feature_limits_branch_20261024/`。
