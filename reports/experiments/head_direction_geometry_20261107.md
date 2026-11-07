# 冻结末层方向：正常动作先于有益偏好改变

2026-11-07，lightbook；本轮分类 **progress**。先核验小时续办真实消费，再开发并执行冻结评分头诊断。没有训练、候选权重、模型推广或重复启动旧训练。

## 无人值守执行证据

- 小时消息11:13:05进入队列，submission `01a1b3fe-d669-7891-a91d-b057c101fa86`；本轮RPC userMessage client `01a1b3fe-d669-7fa2-8cc5-e5cd3a365b03`。
- 新turn `01a1b40d-c0b1-7e10-bae9-dff6b27f9c9d` 于11:29:22开始；rollout最早工具11:29:30.691+08，`call_96e4318f2b5f4d229d963405c32c0e09`。压缩前后有两个同ID turn_context，采用最早的实际工具时间。
- 本机后台13827→1、keeper active、四个unit enabled/active、linger yes、四个sleep target masked、desktop proxy[]、control enabled/STOP不存在/goal active。11:36:45 health healthy；队列消费后为空。
- 本机后台oneshot service的MainPID0/active exited是启动器状态，实际Codex后台另由13827存活证明。没有据此声称OS重启或整夜/未来额度可用。
- 证据：`runs/autonomous_development_20261105/hourly_head_geometry_execution_20261107.json`；accepted时间不是实际执行时间。

## 固定协议

新增 `scripts/audit_head_direction.py`、`scripts/summarize_head_frontier.py`、25项定向测试。

原模型 `runs/ppo_candidate_raw_30m_seed42/final/model.zip`，SHA256 `eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34`，实际计数30003200/36625。只加载一次，CUDA、最高float32精度、TF32关闭、确定性算法；不创建环境、不reset/step、不调用optimizer，不构造或保存更新后的权重。

末层为Linear64→1。单次前向hook同时取合法动作的输入h和实际raw score；解析重建原w·h+b，误差上限事前固定1e-4。实际概率分布mode作为动作，不能用raw score argmax代替。

偏好使用前轮封存的公开输入120状态：`autonomous_topk_evidence_20261107/audit/public_inputs.npz` SHA `2b75bdb1b60ffdbdeaccccfa1fb03e82d49ce805749945cf06634c181f0b3fa5`，preferences SHA `3ceb7faba892d77ac0795d67dbb211c560b8d1dd3c2851e4938ec6e028705804`。固定Top-K有益177对；有害1/未支持28另列，不作为负例。

全episode按SHA256('topk-head-geometry-v1:'+seed)整数mod4切分；validation为992011/992016/992027，其余16seed train。全部19seed已被探索，不能称独立确认。温度1 pairwise softplus下降方向：各train seed内部平均sigmoid(-margin)·(h_good-h_base)，再对seed等权平均。只算原w处的一个固定方向g，不归一化、不执行参数步。

正常参照原30M的匹配GPU全轨迹 `autonomous_gpu_quality_4k_20261106/baseline`。固定chunk0..78每个8lane，共632状态；next chunk首轮验证原动作、输入摘要、episode。所有80组manifest/state/rows外部SHA先于pickle，检查完整链；重建公开输入匹配snapshot和下一轮原rows。train正常seed100000..3、validation100004..7，各316状态/4episode。同局样本相关。

总 **94次GPU前向（危险15+正常79）、0环境步、0策略优化器步、0参数更新**。原policy完整state_dict、Adam、30003200/36625及模型加载后的全局采样RNG前后精确相等；输入文件SHA前后相等。当前31训练源/20推理源文件也与正常参照声明一致。

## 结果

120/120危险状态当前GPU mode匹配旧CPU基准动作；632/632正常状态匹配原GPU动作。最大raw score重建误差 **4.516084903372075e-6**，小于1e-4；此重建误差不证明实际修改权重后的GPU greedy逐位一致。

g范数0.9851575515761001。按episode等权的偏好损失方向导数：train **-0.9705354014274163**，validation **-0.8815999410634169**；前者匹配-||g||²。方向能局部降低两组偏好损失，但动作交叉发生顺序不利于严格保持。

| 几何量 | train | validation |
|---|---:|---:|
| 有益偏好对 / episode | 146 / 16 | 31 / 3 |
| 有益偏好正斜率 | 136 | 28 |
| 首个正常动作排序交叉 | 0.09067871209411729 | 0.09060038626451325 |
| 正常交叉时间中位数 | 3.2227008216144712 | 3.1886343021145818 |
| 首个有益pair排序交叉 | 0.14014688898691172 | 0.8275321065189366 |
| 在首个train正常交叉之前修复的有益pair | 0/146 | 0/31 |

正常margin≤1e-6保守记零；另用max(1e-6,2×batch实测重建误差)做更保守初始tie包络。两者首个train界相同；本次632状态均无初始tie/零界/无限界。未将有限精度包络作为实际新策略的保持保证。

复用已封存标量结果，再算**全部453个train定义边界**，不追加GPU前向。事件网格只由train正常/有益pair交叉和0决定，validation不参与选点。报告各边界右极限t+，没有选取有限步长或最好参数。

在首个train有益pair交叉0.14014688898691172的右侧：train修复1/146、正常动作已改变1/316；validation修复0/31、正常已改变2/316。在train网格第一次出现validation有益pair超过base的边界0.9171437092214334：train有益6/146、正常改变19/316，validation有益1/31、正常改变27/316、未支持pair超过base1个。这里的“改变”不是已证明棋力下降，“有益pair超过base”不是成为全局Top1。

## 资源及证据

根目录 `runs/autonomous_head_direction_20261107/`：

- unit：25passed/0.23秒；guard11:36:56 completed/2.078秒，100%/2GiB；单采样不称真实峰值。
- audit：进程589763；guard11:37:46启动，模型诊断receipt11:37:50完成，controller11:37:53 passed；6.152秒。479冻结输入/45.873MiB，controller含geometry_design共480；全部前后未变。
- audit GPU批8、CPU250%/6GiB/swap0、六物理核affinity，120秒command/600秒wall；原CPU82/90℃、GPU75/85℃、内存/主机/目标/STOP保护保持。3次采样最高CPU67.375℃、GPU37℃、cgroup899.148MiB，均hold；仅为采样最高值。
- frontier：guard11:39:26 completed/controller passed，100%/2GiB/2.077秒；无模型load/forward/optimizer，单采样。
- `audit/geometry_design.json`、`geometry_result.json`、`execution.json`；三阶段guard/controller；`frontier/frontier.json`、根`completion_audit.json`。

## 判断及下一步

本方向在首个正常动作交叉前不能修复任何已选有益pair，**不支持启动其训练**。这只评价一个方向和抽样状态，不能证明任何末层方向都不可行；严格保持全部正常Top1也不是原棋力门槛，不增设或放宽质量门槛。

旧分支仍缺逐分支模型SHA与旧runtime固定证明；当前mode匹配不能补出历史运行身份。`eligible_for_training=false`保持，原4k/8k质量失败结论不变。

下一先查重和核对可恢复状态，制定并实现一个小型、明确来源的原30M **GPU成对偏好复核**：固定4个train seed各一对（共8初始分支），相同公共/私有core和RNG开始，base/alternative首动作后均由冻结原策略续行，固定5000窗口；不根据复核结果筛掉失败个例。先验证恢复/有界阶段/停止保护和计数，再允许受保护的有限分支计算。该工作补当前GPU标签证据，不冒充历史runtime复现或放行训练，不重做既有4k/8k反事实。
