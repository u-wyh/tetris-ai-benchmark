# 合并列统计的训练特征后端：未采用

隔离副本`/home/wyh/projects/tetris-feature-fusion-validation`，分支codex/fused-candidate-features，实现da4586b。main保持已验证行位图搜索、27训练/20推理身份。新后端bitboard_fused合并holes/aggregate/height/bumpiness/wells/top_four遍历，并将等价整数范围预转Python tuple；仅训练显式选择，要求compact搜索，原推理实现未改。

142项通过，包括固定棋盘/hold、256随机列的逐格统计参考、消行/锁出、溢出/重复action错误、配置拒绝及原环境轨迹/快照。27状态×30 ABBA静态特征核对照，40输入冻结，预声明中位≥1.20/每对更快；实际中位1.679165042、最小1.606928375。静态不含搜索/IPC/策略/PPO，不能当训练收益。

## GPU8预声明结果

同compact搜索、同28训练来源/初始配方，两新seed2300000/2300100 AB/BA每臂8192、4096跨进程，要求全状态精确、几何≥1.05且两对更快/全250%无降载。51输入冻结。10:47:17一致性passed，但eligible=false：

| seed | 当前特征秒 | fused秒 | 吞吐比 |
| --- | ---: | ---: | ---: |
| 2300000 | 13.329 | 12.702 | 1.04936231 |
| 2300100 | 13.327 | 12.706 | 1.04887455 |

几何1.049118398（约+4.91%）低于预声明1.05；没有四舍五入过关或重跑计时凑门槛。policy/Adam/PPO更新/全部trainer RNG/8worker/非计时指标均精确。错feature preset恢复拒绝，config/metadata/transaction_state字节不变；只排除wall_seconds及gpu_peak_allocated_mib。51输入完成时未改，CPU250%不降载，资源见decision_and_resource_audit.json。

10:47:20父benchmark completed，条件接力自动skipped_ineligible_upstream，validation_started=false、未创建guard/子计算，两个服务MainPID0。**完整上下文扩展/故障恢复未执行，不能声称通过。** 新后端不合入，main保持原row masks+bitboard特征，无质量模型替换/额外训练。隔离代码与证据保存。

协议说明字段training_source_files从旧设计继承为27，实际源码列表/冻结SHA闭包/metadata均28。该字段不参与runner/verifier；为保持冻结协议原文，独立source_count_erratum.json逐项记录28，数值设置与checkpoint身份未改。后续设计须显式设实际计数。

## 后续

连续两轮性能工作已有一次实际+9.44%采用。此负门槛停止后不继续围绕小数门槛调参/重计时；下一转向有界研究训练目标与greedy棋力的对应关系，使用冻结模型和旧greedy结果，先查是否已有同等stochastic诊断，避免重复失败PPO/KL训练或盲扩2M。

证据：runs/autonomous_fused_features_20261106/、runs/autonomous_fused_features_extension_20261106/summary.json。
