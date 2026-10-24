# 清行节奏归档模型文件身份复核

2026-10-24 16:55 +0800。归档审计此前核对 episode、候选与分支记录之间的 PPO 哈希，但未核对当前磁盘上的冻结模型文件。审计现先计算模型文件 SHA256，与研究脚本固定的 PPO 哈希比较，再逐局核对 episode 中的模型哈希。任一不符即拒绝审计结果。

使用主项目虚拟环境运行 `PYTHONPATH=. /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python -m training.evaluation.audit_clear_cadence_records`，通过：8 局、22 状态、44 分支、明确救援 0、明确误伤 1。`git diff --check` 通过。本次只读归档数据和模型文件，没有重跑 PPO 分支或训练；校验只证明当前文件与记录中的固定哈希相符，不补造历史运行时来源。
