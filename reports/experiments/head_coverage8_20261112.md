# 八episode覆盖对照：固定head偏好目标仍未通过验证

## 事前协议与来源

2026-11-12，原四episode试验已失败封闭，按此前计划只扩大训练episode覆盖：旧4＋新block1四个当前GPU已核验强有益pair，共8pair/8episode。保持同一原30M SHA与实际PPO30003200/36625；fresh d0/Adam，lr.01、256固定更新、GPU float64、pair softplus T1＋normal forward teacher KL β1＋.01 L2。最终64维weight转float32，bias及其余网络不变。

batch8=交替old4/new4（t%2）＋4normal训练状态（chunk=t%79），每episode128曝光。validation4case/3episode仅终点、seed内均值后episode等权，不回流训练；已是开发反馈，不能称独立最终确认。旧eligible_for_training=false工件不改，另限定本次单个probe准入。

复用封存旧640 h和旧validation h/mode，只new block1一次源CUDA8导出。仅内存append新8state（640..647），normal原索引不变；旧缓存未覆写。旧helper在before_changes归档，不能将改后路径SHA冒充Nov7旧执行源码。

## 实际结果

|项目|原4episode|新8episode|
|---|---:|---:|
|固定GPU head更新|256|256|
|每训练episode曝光|256|128|
|train preferred globalTop1|2/4|1/8（old4:0/4，new4:1/4）|
|train pair领先base|3/4|3/8，其中2例仍输第三动作|
|validation globalTop1|0/4|0/4|
|validation episode等权pair loss|4.857334→1.341887|4.857334→1.528766|
|normal train动作变化/316|78|96|
|normal validation动作变化/316|89|104|
|normal train teacher KL|.295142|.465948|
|normal validation teacher KL|.366018|.507369|

新train loss4.84015777→.83647140，d L1=24.25274576/L2=4.75035711。val各preferred raw rank3/4/3/3；只有训练case94真正成为全局第一。实际完整CUDA策略mode计数，非仅缓存仿射预测。增加训练覆盖在这次固定曝光预算下未改善val，不推断更多数据普遍无效。

**预定train/val各≥1全局Top1且val loss下降的门槛失败。** 无evaluation_candidate.zip、无全局棋力评测或策略推广，不延长/扫参数/重跑本试验。normal变化不是棋力失败证明；原完整≥.99pieces/.99score、死亡不增/minseed≥.9门槛保持。

## 工作量、保护与证据

真实source1830044于16:07:48完成，controller16:07:52终态inactive/success/MainPID0。GPU source1＋candidate82=83fullpolicy forward，另外head256forward/256Adam步，1model load、0env、0PPO optimizer。源完整policy/Adam/RNG/ZIP及实际30003200/36625不变，候选只score_head.2.weight变化。max score重建3.894893e-6＜1e-4。

19定向测试/1.23秒通过：旧4兼容、缓存前缀逐值不变且不共享内存、8episode各128曝光、validation扰动不影响训练batch、坏seed/mode/duplicate/masked label拒绝、val外部SHA-before-NPZ与forward超额拒绝。第一次unit controller缺original_context在guard/测试前停止，实际0计算；显式pytest命令的新unit_r2成功，旧attempt保留。

原250%/6GiB/swap0/六核/aggregate/温控/最低内存/GPU及lightbook主机/goal STOP/120command600wall保持。正式8.235秒，4采样最高CPU66.125℃/GPU40.0℃、cgroup1044.391MiB，全hold；采样值非连续真峰。unit_r2的frozen及正式323输入完成后SHA全部相符。

根：`runs/autonomous_head_coverage8_20261112/`：protocol、probe/readiness、extension_features、execution、checkpoint、coefficients、probe_result及completion_audit。结果SHA `930f81a0afb870bcf2b5e84cede4186f6323eae77d9f84e7eb3b555149c0b882`，checkpointSHA `01df1d6497aad4465150e2f64557f76f07fe575df0d9b690f24d586bc39561f9`。旧val npzSHA6910609d…a461、旧640 npzSHA2a992048…411，外部封存后复用。

## 下一有限开发

先审计新结果中pair领先但未全局第一的竞争动作，复用旧/新公开h和封存endpoint系数、对照实际GPU mode，不增加模型forward/环境/优化器。区分训练覆盖不足、pair目标未约束第三动作与局间泛化；这次证据不证明唯一原因。

若确认目标与全局Top1存在具体差距，可单独事前设计一次全合法动作交叉熵偏好目标替代pair softplus，保持lr/步数/KL/L2/8episode与终点门槛，不做参数网格。旧两次失败probe不续训；val继续仅开发验证。必须先封存新协议并完成针对padding/第三动作梯度/隔离的测试；不能直接用失败checkpoint恢复PPO或推广。
