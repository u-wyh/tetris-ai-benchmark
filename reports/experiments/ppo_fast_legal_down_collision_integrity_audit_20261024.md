# 向下碰撞检查基准的归档完整性审计

2026-10-24 16:49 +0800。只读核对 `runs/ppo_fast_legal_down_collision_20261024/` 的协议、三个逐例原子记录和汇总：冻结模型哈希、旧源码（提交 `03a2513`）、归档候选源码、分支源码、快照与候选哈希均匹配；每例两次旧/新/新/旧计时为有限正数，6 对计时重算速度比为 **1.0418480789520603×**，与归档汇总一致。三个逐例记录 SHA256 分别为 `271f791d3c880fcb622e4abc42946f7f3a4dafaa6e63db499a706fddf6bd2949`、`e7e737e88cb72434285347a61463387906280ee1a160a144eb0dce03579dfc3d`、`210b09f15bee65dac6c15ff484712c2c3040149be20e652fcb0f0182eb071bc0`。

复核命令：`PYTHONPATH=. /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python -m training.evaluation.audit_fast_legal_down_collision`。旧落点排序审计也通过（1.0415474969285516×）；其历史候选源码改按归档副本核验，因为当前源码已包含后续优化。审计仅验证保存记录的结构和算术，不独立证明当时运行时的分支输出相同。未重跑 PPO、训练或使用新种子。8080 首页与 8081 `/api/health` 均 HTTP 200。
