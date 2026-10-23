# 评测合法落点旋转与边墙差分结果

2026-10-23 13:06 +0800，按[冻结协议](ppo_fast_legal_rotation_parity_design.md)验证了 168 个状态：7 种方块、4 种起始朝向、左/中/右 3 个有效横坐标、空棋盘及下部错列障碍棋盘。评测专用 `fast_legal` 与正式 Core 的完整有序落点字典在全部状态逐项相同，包括动作 ID、占格和移动路径。

命令：`/home/wyh/projects/tetris-ai-benchmark/.venv/bin/python -m pytest -q tests/test_fast_candidate_env.py -k 'rotated_wall_states or fixed_edge_boards or spawn_is_blocked'`。输出：`161 passed, 13 deselected in 1.92s`；其中 56 个新增参数化测试各检查 3 个横坐标，另有 105 个既有边界测试。运行前 8080 首页与 8081 `/api/health` 均返回 HTTP 200。

只增加正确性回归；未训练、未修改正式 Core、冻结模型或网页。该结果不声称性能收益，也不覆盖所有任意棋盘。
