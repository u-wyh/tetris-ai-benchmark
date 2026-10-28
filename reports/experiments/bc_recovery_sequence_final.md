# 序列级安全纠错：最终结论

结束 2026-10-29T01:21:27+08:00；原因 `insufficient_high_confidence_labels`。

1. **可验证多步恢复优势？** 强证据和跨episode统计：`{'train': {'states': 175, 'strong_states': 2, 'episodes': [3102821023], 'frames': 4, 'multistep_episodes': [3102821023], 'first_same_multistep': 1, 'normal': 1121, 'strong_branches': {'B': 1, 'C': 2, 'D': 2}, 'harm_branches': 3, 'both_censored': 515}, 'validation': {'states': 60, 'strong_states': 0, 'episodes': [], 'frames': 0, 'multistep_episodes': [], 'first_same_multistep': 0, 'normal': 384, 'strong_branches': {'B': 0, 'C': 0, 'D': 0}, 'harm_branches': 0, 'both_censored': 180}}`。
2. **单步为何不足，有无实证？** 只把B不强而C/D强的案例视为有限窗口多步证据；首步相同但后续产生关键分歧另计。没有这样的案例时不宣称已证明单步不足。
3. **高可信标签足够？** False；冻结准入检查 `{'train_states': False, 'val_states': False, 'train_episodes': False, 'val_episodes': False, 'train_frames': False, 'val_frames': False, 'train_multi': False, 'val_multi': False}`。不足即停止。
4. **真实训练新模型？** False；冻结历史模型未更新。
5. **得分保留多少？** 未训练/未完成新开发评测，不外推保分效果。
6. **生存指标改善？** 局部反事实仅适用于所选状态，不是新模型生存改善或长期安全验证。
7. **独立确认通过？** False；没有通过确认，不登记冠军。
8. **值得继续投入？** 本轮不追加训练或采样；先解决证据覆盖/学习准入或实战门槛失败原因，再考虑单独预注册研究。

[冻结设计](bc_recovery_sequence_design.md) · [完整结果](bc_recovery_sequence_results.md) · [逐状态证据](bc_recovery_sequence_evidence.json)

全部原始失败、拒绝、反事实、RNG与哈希永久保留于runs/bc_recovery_sequence/。

Python/Parity与JS回归退出码：[0, 0]。
