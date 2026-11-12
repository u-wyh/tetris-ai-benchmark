# 单次末层GPU偏好训练：完成256更新，验证准入失败

实验2026-11-07完成，2026-11-12额度恢复后封闭报告。分类progress；未重跑训练或推理。

## 固定协议及执行

仅4个当前GPU强有益train pair，4个validation case/3episode全程不拟合；normal train/validation各316状态/4episode。复用640状态公开h，原模型CUDA一次batch8封存新validation特征。64维末层位移d，其余网络/bias冻结；fresh Adam lr.01、betas.9/.999、eps1e-8、固定256更新、GPU float64，最终映射float32。batch8=4pair＋4normal（train seed100000..3,chunk=t mod79）；pair softplus＋cached-logit教师前向KL＋.01||d||²。无epoch/参数选择或追加更新。

source635732，15:58:28 receipt完成、guard15:58:31 completed、8.198秒。实际256 head optimizer步/256 head batch8 forward，1源policy forward＋81更新后完整policy forward，1原模型load，0新环境/0PPO优化器。源PPO计数30003200/36625、完整源policy/Adam/RNG和ZIP不变，候选只有score_head.2.weight改变。每64步保存检查点，序列化fresh Adam真实step256已校验，源PPO计数单独保留。

12项定向测试1.18秒通过：损失精确梯度/掩码稳定性、训练wholeepisode隔离、当前GPU标签和模型准入、仅修改拷贝末层、CUDA不可用拒绝。没有CPU策略训练fallback。旧证据eligible_for_training=false未改，readiness仅允许事前限定单次试验，不授予PPO恢复或推广资格。

## 固定终点实际GPU结果

| 指标 | train | validation |
|---|---:|---:|
| 纠错动作成为全局Top1 | 2/4 | 0/4 |
| good超过base | 3/4 | 0/4 |
| 按episode等权pair softplus，原值 | 2.884307580 | 4.857334129 |
| 固定终点pair softplus | 0.476059711 | 1.341886832 |
| 正常状态实际GPU动作变化 | 78/316 | 89/316 |
| 正常cached教师KL | 0.295141965 | 0.366017629 |

train新GPU模式[1647,40,984,152]，good动作[1047,40,63,152]；pair超过base不是全局Top1。validation新模式[613,201,272,429]，good[107,84,134,336]，good-base margin[-.315533,-.453944,-2.157216,-.135882]。val seed992027的偏好loss反而增加，其余下降；按3episode等权，992011两case没有多算为两独立局。

d L1=16.413986231608813、L2=3.8032194067508476。更新后完整GPUh/公开输入/合法动作与缓存相同，最大score重建误差3.74187061424891e-6<事前1e-4。所有632 normal变化完整保留；变化本身不证明棋力下降。KL是明确定义的缓存logit教师目标，不冒充原CUDA概率逐位一致。

预定准入要求train/val各至少1个good全局Top1并且val loss下降。val0/4，因此eligible_for_quality_evaluation=false；没有生成evaluation_candidate.zip、没有完整棋力rollout、没有推广。原≥.99pieces/.99score、死亡不增/minseed≥.9门槛未改。本试验已终止，不延长256、调β/LR/epoch补救，也不重试已封闭LP。

## 保护与跨轮恢复

正式原250%CPU/6GiB/swap0、六核、原CPU82/90℃GPU75/85℃/最低可用1GiB/GPU/host/STOP与120command600wall保护。4次采样最高CPU64.75℃、GPU38℃、1047.094MiB，全部hold；是采样最高值，非连续真峰。unit100%2GiB、2.085秒单采样。14/312冻结输入在Nov12收尾再次SHA相符。

旧native turn01a1b4f9-5790-7760-8e99-a8f74ede18f4完成实际实验及completion_audit后API因usageLimitExceeded失败，提交与报告未完。Nov12 15:39:25既有小时流程ordinary_usage_allowed恢复原goal，当前新原生turn真实工具续办；期间未声称持续开发，没有改账户/付费/使用重置券。Nov12排队消息仍pending，不能将accepted当消费。后台13827→1仍驻留、四unit enabled active/linger/sleep保护；当前桌面proxy存在，不称本轮无桌面验收或OS重启验收。

证据runs/autonomous_head_preference_20261107/{unit,probe,completion_audit.json}；probe_result SHA29adf1200930716d317d7b759c24c01f6032b46b6c63d7dfa81739526b5527c6，checkpoint SHAe87588e54249b725e07652c599666d5b18f0a6eec267bd677778a65de03e3d49。跨轮native_goal_head_preference_execution_20261107.json和native_goal_head_resume_20261112.json。

## 下一步

目前仅4个training episode；现有证据不能区分训练覆盖不足、目标局限和模型能力。先扩展当前GPU验证的训练episode覆盖，固定旧train切分中尚未验证的最早4seed[992010,992012,992014,992015]，每seed从旧TopKbeneficial按(step,state_id,action)取最早1，先封存具体case再新GPU结果。不是把旧177标签直接当当前真值。保持val992011/016/027隔离；这些validation已用于开发反馈，不能当最终独立确认集。

复用8配对分支事务，独立新job，原30M/5000branch/40000job、256round有界phase与原保护，0策略优化器、不重做512oracle、不替换失败case。完成来源验证后，再事前定义下一学习方案，不从失败checkpoint继续调参。当前目标保持active。
