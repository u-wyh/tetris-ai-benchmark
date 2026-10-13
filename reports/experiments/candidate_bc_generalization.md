# BC Champion 全新种子泛化验证

状态：**已冻结模型，30 局测试在后台等待已有 V2 配对评测完成。结果待完成。**

冻结模型为 `20M PPO + BC epoch 3`，SHA256 `5f5613cb5eb4e80a02406b70902ad3d5c571eb8fe38bec4b40f73b34289ca663`。来源是 20,000,768 PPO steps 的 Candidate checkpoint，随后用 214,809 条 V2 Hold 专家示范进行 3 epoch Actor-only BC，BC 不增加 PPO 环境步数。来源、教师配置哈希、数据分片哈希、Observation/Feature/Action/Reward 版本写入 `runs/candidate_bc_next_stage/champion/provenance.json`。独立 Python 进程加载与冻结前后的 masked logits 相等检查已通过。

原建议种子 `400000..400029` 已用于既有独立测试，因此冻结新种子 `1234000..1234029`。协议是双方同一 `TetrisCore`、每局最多 100,000 块、确定性 BC、正式 Action Mask、传统 V2 `mode="hold"`。两方各 30 局，每局原子保存，可中断恢复。结果会记录均值、中位数、P10/P90、截断数、消行、分数、逐种子配对胜负和耗时。双方均到上限仅说明寿命下界相同，不能算真实寿命平局。该种子集禁止用于 KL/DAgger 超参数选择。

已有但**非本轮新种子**的 10 局、100,000 块验证：BC 与 V2 均 10/10 到上限，BC 平均分 8,336,668,120，V2 平均分 8,334,245,430。这说明该上限内二者接近，不能说明超过 100,000 块后谁更强。
