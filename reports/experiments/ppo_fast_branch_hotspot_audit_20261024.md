# 冻结 PPO 反事实分支热点细分审计

2026-10-24 16:06 +0800。只读分析已有的三个独立 episode、每条 1000 块的 `cProfile` 原件；没有重跑分支或使用新种子。轨迹身份、冻结模型哈希及逐块历史结果校验见 [跨 episode 剖析](ppo_fast_branch_profile_replication_20261024.md)。

| seed / 状态步号 | 合法落点累计秒 | `_branch` 自身秒 | `collides` 自身秒 / 次数 | `rotate` 自身秒 / 次数 | 特征累计秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1550002 / 3875 | 3.177 | 1.764 | 0.353 / 1,291,841 | 0.299 / 960,657 | 1.254 |
| 1550005 / 4900 | 3.256 | 1.811 | 0.361 / 1,324,142 | 0.309 / 983,657 | 1.246 |
| 1550007 / 6075 | 2.989 | 1.642 | 0.331 / 1,187,564 | 0.278 / 873,111 | 1.275 |

`_branch` 自身耗时约占合法落点累计耗时的 55%，而 `collides` 与 `rotate` 的自身耗时各约占 10%。三条轨迹均呈现相同分布。因此下一次有界实验应先测量搜索循环中的状态构造、集合查询和节点路径构造，避免仅凭碰撞调用次数推断收益。已有的预位移碰撞掩码实验仅得 1.011×，已撤回；这份审计不构成新的加速结果。表中的累计耗时包含子调用，不能把列相加，也不能外推无剖析完整吞吐。

原件及 SHA256：

- `runs/ppo_branch_profile_replication_20261024/seed_1550002_step_3875.prof` — `22cb7869a9dc3db3a49eb44a433b8066005113b2c2874d8348cad1685e3dabfa`
- `runs/ppo_branch_profile_replication_20261024/seed_1550005_step_4900.prof` — `cd1fc02ab9cfdaf98701b06a862f5b06d6f61d152f5e1d79d597299cf44e2da7`
- `runs/ppo_branch_profile_replication_20261024/seed_1550007_step_6075.prof` — `70ef8cb432aa0564081be8e563b0032472ba2aab69dba17d6e75ec7e57fa28f7`

可复核命令：`python -m pstats runs/ppo_branch_profile_replication_20261024/seed_1550002_step_3875.prof`，查看 `fast_legal.py` 的 `_branch`、`collides`、`rotate` 及 `fast_candidate_features.py` 的 `candidate_features`。三份原件均按相同方式读取。
