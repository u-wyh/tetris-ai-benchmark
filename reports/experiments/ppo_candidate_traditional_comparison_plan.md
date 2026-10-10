# Candidate Champion 与传统算法公平比较方案

本方案在 Candidate 正式测试完成前固定。Candidate Champion 先由 32 个 validation seeds、每局 20,000 块的结果选定并冻结；正式测试结果不得用于改选模型。

未来配对比较使用相同的 100 个 seeds（200000–200099）、每局最多 50,000 块。Candidate、Legacy V1、V2 Hold-only 和 V2 Beam-8 都使用同一版 Python TetrisCore、同一 seed 派生的方块序列、相同公开 Hold/Next 信息边界和正式计分规则。Candidate 使用合法 Action Mask；传统算法通过同一合法落点集合选择。逐 seed 同时记录存活、消行、分数、Game Over、上限截断和运行时间。达到上限按右截断处理。评测一组接一组运行，避免并发负载干扰。

现有传统实验是另外 10 个 seeds（100016–100025）、每局上限 100,000 块，不能直接与 Champion 的 100-seed、50,000 块正式测试判胜负。其历史吞吐可用于粗估成本：

| 策略 | 历史端到端速度 | 100 局 × 50,000 块上限估计 |
|---|---:|---:|
| Legacy V1 | 17.0 块/秒 | 最多约 81.7 小时 |
| V2 Hold-only | 97.1 块/秒 | 最多约 14.3 小时 |
| V2 Beam-8 | 19.6 块/秒 | 最多约 70.9 小时 |

以上为按全部跑满上限推算的单进程墙钟时间；提前 Game Over 会缩短实际时间，机器负载变化也会影响速度。依据为 [Legacy V1](../traditional_longlife/validation10_100000_20261008/v1/aggregate.json)、[V2 Hold-only](../traditional_longlife/validation10_100000_20261008/hold/aggregate.json) 和 [V2 Beam-8](../traditional_longlife/validation10_100000_20261008/beam/aggregate.json) 的端到端吞吐。100 个正式 seeds 此前已用于 Raw PPO 报告，因此它们不是完全未使用的新测试集。

本轮只估算成本，不启动完整传统算法评测。若随后执行，应固定运行顺序与协议，逐局保存并支持恢复；不能根据正式结果再调整 Champion。
