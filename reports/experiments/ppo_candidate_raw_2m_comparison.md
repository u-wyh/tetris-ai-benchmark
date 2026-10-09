# Candidate-Scoring PPO Raw 2M 对照

同一批 validation seeds `100000..100031`、每局最多 5000 块、确定性推理与官方 Action Mask。两个模型均训练到 2,002,944 步，使用相同 Raw Reward；候选模型从 seed42 随机初始化，使用版本化公开特征。

| 指标 | Raw MLP | Candidate-Scoring |
|---|---:|---:|
| 平均存活 | 57.94 | 3242.75 |
| 中位存活 | 56.50 | 3532.50 |
| P90 存活 | 76 | 5000 |
| 平均消行 | 10.25 | 1285.38 |
| 平均正式分数 | 1315.62 | 19944809.38 |
| 每 100 块新增洞 | 151.24 | 0.75 |
| 死亡时平均最高列 | 20.00 | 19.33 |
| 中央列最高比例 | 3.1% | 55.6% |

配对平均存活差（Candidate − Raw）为 3184.81 块；固定种子、按整局重采样 10,000 次的 95% bootstrap 区间为 [2566.87265625, 3769.1890625]。逐 seed 差保存在 `runs/ppo_candidate_pipeline_seed42/paired_validation_2m.json`。

8-worker CUDA smoke 吞吐：[482.45, 482.9] steps/s；PyTorch 峰值预留显存：[134.0, 134.0] MiB。正式最后 Task 峰值：{'allocated': 117.65, 'reserved': 142.0} MiB。

配置：`CandidateScoringPolicy`、`candidate-dict-state237-features16-v1`、`candidate-placement-uint8-v1`、seed 42、Raw Reward。训练代码 commit `f365081505548e0626037e265769d8c86a5f871a`。

本次只用 validation seeds 评估，不使用 final-test seeds 调参，也未启动 10M。

## 5000 块截断审计

Candidate 14/32 局达到 5000 块上限（43.75%），18/32 局 Game Over；存活中位数 3532.5 块。平均存活 3242.75 块受到右截断影响，不能解释为无上限平均寿命。14 局的 `pieces_survived` 均恰为 5000，且 `survived_cap=true`、`game_over=false`；另外 18 局均记录 Game Over。逐 seed 检查了 seed 对齐、1–5000 块范围与结束标志互斥，未见异常评测行为。

| Seed | Raw 存活 | Candidate 存活 | Candidate 结局 |
|---:|---:|---:|---|
| 100000 | 69 | 1554 | Game Over |
| 100001 | 40 | 5000 | 上限截断 |
| 100002 | 46 | 5000 | 上限截断 |
| 100003 | 57 | 5000 | 上限截断 |
| 100004 | 46 | 5000 | 上限截断 |
| 100005 | 44 | 5000 | 上限截断 |
| 100006 | 83 | 4047 | Game Over |
| 100007 | 76 | 5000 | 上限截断 |
| 100008 | 46 | 5000 | 上限截断 |
| 100009 | 48 | 5000 | 上限截断 |
| 100010 | 39 | 2522 | Game Over |
| 100011 | 58 | 3688 | Game Over |
| 100012 | 56 | 242 | Game Over |
| 100013 | 66 | 444 | Game Over |
| 100014 | 55 | 691 | Game Over |
| 100015 | 48 | 3377 | Game Over |
| 100016 | 57 | 450 | Game Over |
| 100017 | 42 | 2493 | Game Over |
| 100018 | 52 | 1222 | Game Over |
| 100019 | 42 | 5000 | 上限截断 |
| 100020 | 75 | 2591 | Game Over |
| 100021 | 89 | 5000 | 上限截断 |
| 100022 | 59 | 1761 | Game Over |
| 100023 | 53 | 5000 | 上限截断 |
| 100024 | 68 | 2740 | Game Over |
| 100025 | 38 | 1260 | Game Over |
| 100026 | 72 | 1232 | Game Over |
| 100027 | 45 | 5000 | 上限截断 |
| 100028 | 65 | 5000 | 上限截断 |
| 100029 | 64 | 1311 | Game Over |
| 100030 | 64 | 5000 | 上限截断 |
| 100031 | 92 | 2143 | Game Over |
