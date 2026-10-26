# 清行节奏采样重放记录计数校验

2026-10-27 00:48 +0800。恢复校验原先只要求各层 `eligible - spacing_excluded - quota_excluded` 等于入选数。若同时增加合格数与间距排除数，守恒仍成立，篡改的记录会被接受。现逐项要求重放审计计数与原子保存的 episode 计数一致。

既有 8/8 份审计记录通过校验；对每局在内存中同时增加恢复对照层合格数与间距排除数，8/8 份篡改均被拒绝。`git diff --check` 通过；8080 首页、8081 `/api/health` 均为 HTTP 200。证据：`training/evaluation/audit_clear_cadence_selection.py`、`runs/ppo_clear_cadence_counterfactual/{episodes,selection_audit}/seed_*.json`。没有重跑 PPO、反事实分支或训练，也没有使用新 seed。

此校验将两个归档记录对齐，不独立证明历史重放过程。原结果仍为 0/8 局明确救援、1/8 局明确误伤，证据不足。
