# GPU训练标签扩展 block1：4个新episode均强有益

2026-11-12，lightbook；progress。失败的4episode/256步head试验已封闭于499b6cb，本轮没有重跑或追加它的优化步。

## 实际恢复与协议

Nov7上轮在实验完成后因额度失败；Nov12 15:39:25原小时机制确认ordinary_usage_allowed恢复原goal。本轮native turn01a1ceb2-7ff5-7f43-b98b-23c61f7799a3首工具15:39:45.269+08/call_4c8a6ee694c04786b332c27392c0e6c3。新queued submission01a1ceb2-8015-74a2-923c-8bc542a9fd6f在本轮初仍pending，不声称消费。后台13827→1保持驻留，主机machine-id6b1b552131804ebc918f30005b4d0204；当前desktop proxy存在，不扩大无桌面/OS重启验收。额度间断没有持续开发或任何账户/付费/重置券操作。

在新GPU结果前，固定旧TopK train切分尚未核验seed数值最早4个992010/012/014/015，各取旧beneficial/ppo_top中(step,state_id,action)最早1。val992011/016/027隔离；旧CPU标签已探索，不能当独立最终质量。protocol.json先于GPU；不按新结果筛选/替换案例或延长5000窗口。

selector新增受限training_block1；只允许block0/1，validation只0，原train/validation选择兼容。输出显式training_block，修改前capture/window源码与SHA在before_changes归档。17项定向测试1.16秒通过（6新block测试＋11选择/driver兼容），没有策略/环境计算。

probe1823853于15:45:39通过8真实native worker完整core/NumPy/action RNG/公开mask配对恢复，0模型load/forward/env/optimizer。新driver1824118自动串行10phase，15:49:20真实completed/terminal、unit inactive/success/MainPID0，21checkpoint/5000round。一次25秒journal观测到期返回124，仅是观察句柄到期，后续核验同driver，没有重启任务。

## GPU证据

原30M SHA eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34，实际PPO30003200/36625；CUDA最高float32/TF32false、8独立base/alt分支，固定首动作后共用冻结policy greedy继续，死亡/窗口停止不auto-reset。

实际26900唯一环境步、5000 GPU forward、10次原模型load、0策略优化器。每阶段≤2×256round/4096新步，job≤40000、branch≤5000；原模型完整policy/Adam/计数/加载及恢复后的采样RNG均未变。没有重做旧512恢复oracle、原4train/4val分支、80forward缓存/LP或256head试验。

| state / seed / snapshot step | base→alt | base新步（真死亡） | alt新步（右删失） | 共同窗口得分比 | 四消比 |
|---|---|---:|---:|---:|---:|
| 94 / 992010 / 25891 | 1169→62 | 2000 | 5000 | 1.008582492 | 1.010309278 |
| 96 / 992012 / 31013 | 364→338 | 2000 | 5000 | 1.017598578 | 1.031914894 |
| 86 / 992014 / 12635 | 959→1328 | 900 | 5000 | 1.046628829 | 1.112500000 |
| 98 / 992015 / 72678 | 1073→842 | 2000 | 5000 | 1.012256473 | 1.005128205 |

4/4按未修改的原preference规则strong beneficial；观测生存增益3000/3000/4100/3000，不能外推5000以外。新4train episode与原4train及3val episode不同；现在共8个当前GPU确认的训练episode/8pair，验证4case/3episode继续隔离。这不是候选策略棋力提升证明，也不自动给PPO恢复/推广资格。

window source1825626于15:50:52完成全21chunk manifest链/完整worker/初态core/RNG/公开输入/合法动作/终止轨迹/10实际receipt审计，0额外模型/forward/env/optimizer。新8branch均显式当前模型SHA。旧8branch已有记录字段相等8/8；仍不能补旧branch缺少的历史模型/runtime/完整tensor身份，不能将全部177旧偏好提升为当前真值。

## 保护与可恢复证据

13guard全部completed/采样hold；正式250%CPU/6GiB/swap0、六核亲和性与aggregate slice、CPU82/90℃GPU75/85℃、最低可用1GiB、GPU telemetry/主机目标STOP/120command600wall。unit/window纯数据100%2GiB，窗口入口strict授权前后检查。

采样最高CPU70℃GPU50℃/单guard cgroup3202.039MiB，非连续峰值。91job/95pipeline/170window及各phase递增外部SHA完成后相符，源ZIP/31训练20推理源未变；readiness不改旧eligible_for_training=false。证据runs/autonomous_training_block1_20261112/{protocol.json,unit,probe,capture,pipeline,phase_01..10,window,completion_audit.json}；窗口SHA d9a4f8ba78668de6663a907818e84bf82a72fa4991d4a66cddfa4aa9bf97e1d3，branches SHA 34ea07a362860f400bf9dd52f50e0491fdbe89a17d6f48a614ec373b174064ee。跨轮回执native_goal_head_resume_20261112.json。

## 下一预定比较：8episode训练覆盖，单次fresh head试验

先封存独立的新准入/协议，检验训练覆盖这一变量；不从失败检查点续调。保持4episode试验的loss/LR/.01L2/β1、CUDA float64、fresh Adam、256更新、bias及其他网络固定。每步batch8仍4pair＋4normal：train pair交替原4与新4（t mod2），normal仍seed100000..3/chunk=t mod79。8train episode各128次暴露；val4/3episode只终点，按episode等权，不选epoch或调超参。既有val已作开发反馈，不称独立最终测试。

原640公开缓存和Nov7 validation_features.npz/readiness的外部SHA、源模型和31/20/runtime身份必须先验证；旧val源mode和原weights证据复用。只为新block1初态做1次原模型CUDA batch8导出h，避免重跑原80和旧val源forward；新train8视图可追加到原640缓存的内存副本，pair索引[0,2,4,6,640,642,644,646]、normal索引保持原位，不能覆写旧工件。输出应明确全新8episode训练数据准入与validation禁止拟合。

事前full-policy预算83：新源train1＋更新后train16状态2batch、val8状态1batch、normal632状态79batch＝候选82；另256 head batch8训练forward、0环境/0PPO更新。所有新h/public digest/模型/counts/原source policy/Adam/RNG与后续actual GPU模式核验保持；输出fresh head优化器与源PPO计数分离，256不会加进源30003200/36625。先定向测试新batch隔离/原4episode兼容/追加缓存与step/fresh optimizer，再受原250%6GiB/温控/120command600wall保护执行。

只评固定终点；沿用train/val各≥1good全局Top1且val episode-equal pairloss下降的准入条件，normal所有动作变化/KL全报告，不把变化当棋力失败或新增正常Top1全同门槛。若失败停止本8episode试验，不扫参数；若通过仅生成evaluation-only候选，再走原完整棋力≥.99pieces/.99score、死亡不增/minseed≥.9门槛。不自动PPO resume或推广，不改旧flag，不GPU/host/额度fallback；不足保留等原每小时。
