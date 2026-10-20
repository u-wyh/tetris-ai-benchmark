# 自主强化学习阶段结果

状态：503808 步开发短局未通过预设门槛。

CPU 训练使用加长折扣和 GAE、冻结 PPO KL 锚定；触顶右截断。

## smoke

```json
{
  "ppo": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 6353400,
    "score_per_1000": 3176700.0,
    "four_fraction": 0.9418886198547215
  },
  "B12": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 6324650,
    "score_per_1000": 3162325.0,
    "four_fraction": 0.9436274509803921
  },
  "bc": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 3371900,
    "score_per_1000": 1685950.0,
    "four_fraction": 0.0
  },
  "candidate": {
    "games": 2,
    "mean_pieces": 2000,
    "p10_pieces": 2000.0,
    "capped": 2,
    "early_under_10000": 2,
    "mean_score": 6335950,
    "score_per_1000": 3167975.0,
    "four_fraction": 0.9253012048192771,
    "common_score_ratio_to_ppo": 0.9972534391034722
  }
}

```

## dev_20k

```json
{
  "ppo": {
    "games": 16,
    "mean_pieces": 18682.5,
    "p10_pieces": 15991.5,
    "capped": 11,
    "early_under_10000": 1,
    "mean_score": 570211956.25,
    "score_per_1000": 30521180.583433695,
    "four_fraction": 0.9284749034749035
  },
  "B12": {
    "games": 16,
    "mean_pieces": 16172.9375,
    "p10_pieces": 5092.0,
    "capped": 12,
    "early_under_10000": 4,
    "mean_score": 486958500,
    "score_per_1000": 30109465.271846876,
    "four_fraction": 0.9159586257850019
  },
  "bc": {
    "games": 16,
    "mean_pieces": 20000,
    "p10_pieces": 20000.0,
    "capped": 16,
    "early_under_10000": 0,
    "mean_score": 333544312.5,
    "score_per_1000": 16677215.625,
    "four_fraction": 0.00029338418659234265
  },
  "candidate": {
    "games": 16,
    "mean_pieces": 18472.6875,
    "p10_pieces": 15389.0,
    "capped": 13,
    "early_under_10000": 1,
    "mean_score": 560643043.75,
    "score_per_1000": 30349836.41389484,
    "four_fraction": 0.8825672379383533,
    "common_score_ratio_to_ppo": 0.9935890770378796
  }
}

```
