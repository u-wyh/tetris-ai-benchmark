# 正确run RNG与可恢复评测的集成验证

## 已确认的缺陷

本地SB32.9.0的BaseAlgorithm.load先恢复保存数据再调用_setup_model；MaskablePPO._setup_model调用set_random_seed(saved_seed)。因此初次迁移源模型时，会重置Python/NumPy/Torch CPU及全部CUDA生成器，覆盖新run已建立的随机状态。原训练事务的run_seed元数据又硬编码为42，无法准确反映model.seed。

修复在初次加载源模型前保存目标run的完整全局RNG，finally中恢复（含加载失败路径）；权重/Adam迁移不变。trainer元数据现在记录实际model.seed。正常事务恢复仍以已提交trainer_state最后恢复RNG，不另设新种子。旧来源已有manifest的事务不迁移；旧数值源码保留于7a1d27b及此前提交。

26项受保护测试通过，包括真实SB3checkpoint加载前后Python/NumPy/CPU/CUDA RNG保持、异常路径、Adam状态及兼容性。运行库/比较身份/接力传播另12项轻量测试通过。该缺陷的修复解决可复现性，不证明其是旧模型长局死亡的原因。

## 新评测路径

参数控制器、额外长局配对、条件续训的长局检查可由新协议显式选择evaluate_resumable；默认保持legacy。白名单选择校验完整评测来源及Python/运行库/诊断身份，比较器拒绝跨运行库或诊断设置混比。接力传播所选入口与身份。周期短局检查继续使用其匹配legacy基线，避免把旧结果改写身份。旧“Review before extending”文字已改为预声明自动门槛说明，不再暗示人工逐项审批。

当前新协议的两阶段验证：

1. CUDA＋8环境4096步→退出→恢复8192步，使用base seed1700000–1700007，KL0.3/LR3e-5/保留Adam/原奖励与10K训练局上限。实际model.seed及trainer元数据均1700000，8worker快照、GPU峰值分配、586000→586160的18份Adam计数、参考SHA全部验证。
2. 真实参数控制器复用已提交8192步（不增加训练），用新入口跑2个种子64块；24个来源文件与运行库身份匹配，controller_smoke通过。随后串行采集新入口8×20000冻结基线，要求逐局与原评测器的历史基线完全一致，才能自动开启262144步新筛选。

新基线是为扩展来源身份与完整长局一致性验证实际采集；未修改旧结果的身份。该新run改变了正确RNG协议及训练seed，属于新开发协议，不是与旧run单变量因果对照。仍只使用开发种子，不使用最终测试、不发布模型。

## 自动执行与接手

- 原Adam候选额外验证未通过，已自动止损，未追加训练。
- 新目录runs/autonomous_rng_isolation_20261105/；持久试运行tetris-rng-isolation-pilot-20261105.service，design.json在训练前冻结来源/设置/预算。
- 17:45:15 CUDA8恢复通过；17:45:38新控制器短局接力通过；随后baseline_guard采集完整8×20000基线，已有完成局逐个持久提交。
- 基线完全一致后，脚本才创建主协议并启用tetris-parameter-screen@autonomous_rng_isolation_20261105.service及对应parameter-followup实例。后者仅对合格者启动额外8配对，过门槛才恢复原事务分段至2M。
- 当前来源与支持文件冻结，运行中不要修改。查看pilot_status.json、实际systemd MainPID及只读observe_development.py，避免重复启动。

所有代码、文件和用户服务留在lightbook；共用CPU250%/内存6GiB/swap0、原温控与时间预算。基线采集及其后新筛选尚未完成，不能提前宣称新模型改善。
