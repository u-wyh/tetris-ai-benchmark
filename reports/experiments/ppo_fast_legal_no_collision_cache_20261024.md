# 快速合法落点碰撞缓存消融：未采纳

2026-10-24 16:16 +0800。在临时载入的源码副本中移除 `collides` 的 `lru_cache`。仓库实现、正式 Core、模型和网页均未修改。

从既有清行节奏数据恢复 22 个归档快照，逐状态比较完整合法落点字典，**22/22 一致**。单线程交替运行 8 轮，每轮每版各调用 550 次。原版中位 **0.7294 秒**，无缓存版 **0.8019 秒**，原版/无缓存版为 **0.910×**；移除缓存约慢 9.9%，故保留现有缓存。此结果只适用于这些归档状态的合法落点搜索，不能外推完整 PPO 分支。

原始逐轮计时及复现脚本在 `runs/ppo_fast_legal_no_collision_cache_20261024/`；脚本 SHA256 `bca8745d1fdb753aa8e3be235da0ce7256a77e7b1f2f6d45e947726c6e67b99d`，结果 SHA256 `b56240d6605708e7f49ce8e0e9954fb848998ec31eff60b3dd7f71f86ff912f2`。复现命令：`PYTHONPATH=. /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python runs/ppo_fast_legal_no_collision_cache_20261024/bench.py`。完成后 8080 首页与 8081 `/api/health` 均为 HTTP 200；无新 seed、训练步或反事实分支。
