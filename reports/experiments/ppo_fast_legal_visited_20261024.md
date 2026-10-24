# 评测合法落点 visited 查询消除：负结果

2026-10-24 16:39–16:40 +0800。在评测专用 `fast_legal.py` 的左右扩展路径，外层已经确认状态不在 `visited`，尝试删掉内层重复查询。候选源码保存在 `runs/ppo_fast_legal_visited_20261024/candidate_fast_legal.py`；正式 Core/JS、冻结 PPO 和网页模型均未改动。

三个已归档独立开发 episode（seed 1550002/1550005/1550007）各以原 PPO 动作续控 500 块，旧/新各两次，按旧/新/新/旧交替，单 CPU 线程。旧、新六对完整分支输出均相同。旧版累计 6.455 秒，候选累计 6.459 秒，速度比 **0.9993×**；六对速度比范围 0.9926–1.0053。这个小改动未带来可信的端到端加速，已恢复原代码。

定向 Core 对照测试 `tests/test_fast_candidate_env.py` 为 175/175 通过。协议、各分支结果、汇总和候选源码在 `runs/ppo_fast_legal_visited_20261024/`；复核脚本为 `training/evaluation/benchmark_fast_legal_visited.py`。恢复调用校验协议并直接复用已有结果。8080 首页及 8081 `/api/health` 均 HTTP 200。无新 seed、训练步或确认组使用。
