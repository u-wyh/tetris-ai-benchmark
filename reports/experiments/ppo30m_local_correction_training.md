# PPO 30M 局部纠错：训练

状态：`v1_failed_short_validation`。

从冻结 PPO 30M 初始化新 Candidate Policy，只更新 Actor。A 类用合法动作交叉熵，C 类用原 PPO→新模型的 masked KL；Critic 冻结，未加入 PPO Policy Gradient。

## V1

保持系数 1.0；设备 cuda；已完成 3 epoch；每个 epoch 的模型、优化器、RNG 均独立保存。

| Epoch | 纠错损失 | 保持损失 | 验证 A 动作一致率 | 验证 C 对 PPO 一致率 | 验证 C masked KL |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.4513 | 0.0013 | 0.00% | 97.03% | 0.00587 |
| 2 | 4.4502 | 0.0091 | 0.00% | 94.18% | 0.02272 |
| 3 | 4.5292 | 0.0199 | 0.00% | 92.07% | 0.04756 |
