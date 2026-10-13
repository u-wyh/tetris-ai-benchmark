# 冻结 BC 参考的 KL-PPO 2M

状态：**实现与单位 smoke 测试完成；正式 CUDA 8-worker 训练排队，结果待完成。**

独立 run `runs/candidate_bc_klppo_2m` 从冻结 BC Actor 和其未被 BC 更新的 20M Critic 初始化，使用全新优化器、worker seed `242..249`、Raw Reward、Task 4096 与原 PPO 参数。计划 2M 环境交互步；不会覆盖纯 PPO 20M/30M、BC 或普通 BC+PPO 模型。正式训练前，分别用 `beta=0.01`、`0.03` 从冻结 BC 训练约 0.25M 步，在独立调参种子 `1260000..1260007` 的 20,000 块游戏中按平均存活、P10、截断率预先固定的顺序选参；精确相同时选约束较强的系数。两条短程试验各自计入 PPO 环境交互预算，均不当作正式 2M 的续训。正式 run 的参考模型 SHA256 与选定系数写入配置，恢复时再次校验。

损失为原 SB3-Contrib MaskablePPO 的 clipped policy loss、value loss、entropy loss，再加 `beta × KL(π当前 ∥ π冻结BC)`。KL 从当前与冻结 BC 的**完整 masked Categorical 概率分布**计算，只对合法动作求和；参考策略无梯度。GAE、Value clip、梯度裁剪及 PPO early-stop 行为沿用原实现。训练 CSV/TensorBoard 记录 BC KL、Explained Variance、近邻 PPO KL、clip fraction 和奖励。

已通过测试：相同参数时 KL≈0；扰动当前 Actor 后 KL 非负且 Actor 接收梯度；冻结参考参数不接收梯度；一次真实 Candidate 环境 rollout 后 Actor/Critic 均更新；Save/Load 后 KL 一致。正式 2M 的生存表现与是否优于普通 PPO **尚无结果**。不会根据保留的 30 个全新泛化测试种子调系数。
