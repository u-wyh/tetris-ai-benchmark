# PPO 评测候选特征多行消除一致性

2026-10-23：为评测专用快速候选特征增加双、三、四行消除的确定性差分测试。每个局面固定为 I 方块及底部 N 行仅第 4 列留空；验证合法落点确实包含 N 行消除，再逐项比较正式实现与快速实现的完整动作特征矩阵及合法 mask。源代码哈希和协议见 `runs/ppo_feature_multiline_parity_20261023/protocol.json`。

定向测试 `tests/test_fast_candidate_env.py` 全部 13 项通过；完整输出见 `runs/ppo_feature_multiline_parity_20261023/tests.log`。本次没有性能收益主张或新训练步；正式 Core、冻结模型及网页默认模型未修改。8080 根路径和 8081 健康端点均返回 HTTP 200。
