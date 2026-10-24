# 快速合法落点整数访问状态：未采纳

2026-10-24 16:13 +0800。依据三条独立 episode 剖析中 `fast_legal._branch` 自身约占合法落点累计耗时 55% 的结果，在临时源码副本中把 BFS `visited` 的 `(x, y, rotation)` 元组改成唯一整数编码。未修改仓库实现、正式 Core、冻结模型或网页。

用既有清行节奏数据的 22 个归档快照恢复公开棋盘，逐状态比较完整合法落点字典，全部一致。单线程对每个实现交替做 8 轮计时，每轮对 22 个状态各调用 25 次。原版中位 **0.7177 秒**，整数版中位 **0.8149 秒**，原版/整数版为 **0.881×**。整数编码增加算术开销，故不采纳。此处只测合法落点函数，不外推完整 PPO 分支吞吐。

原始逐轮计时和可重跑脚本位于 `runs/ppo_fast_legal_integer_visited_20261024/`；脚本 SHA256 `335e15310d18035d9f140b8749a3161ef05deafeea2a01d8c201335d61160730`，结果 SHA256 `fbe37d8a13df1076c9e40f24ab623f08554144871d9b4c1a02878bca70595562`。运行命令：`PYTHONPATH=. /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python runs/ppo_fast_legal_integer_visited_20261024/bench.py`。完成后 8080 首页和 8081 `/api/health` 均为 HTTP 200；无新种子或训练步。
