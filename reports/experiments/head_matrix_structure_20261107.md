# 固定评分头约束矩阵的数值结构

2026-11-07，lightbook；progress。复用d25c8d7封存的640公开状态h，不重跑GPU、不重求解已到60秒上限的联合LP。本轮0模型load/forward/环境/策略优化器/LP调用，7项定向纯代数测试通过（0.11秒）。

## 小时消息实际消费

15:15:55 queued submission `01a1b4dd-4bee-7de0-a1d7-1a2415387678`，本轮user client `01a1b4dd-4bed-7010-8f6c-2cffb7bbb853`，新turn `01a1b4e5-719e-7071-bc10-d616661dfc7f` started epoch1794036298，最早实际工具15:25:06.217+08/call_4a7a06895d4a4d88b9f492a76115c80f。RPC queue空，上一原生turn已completed；这次有消费和工具证据，不仅accepted。后台13827→1、health15:23:55 healthy、四unit enabled/active、linger yes、sleep masked、proxy空、goal active；固定machine-id 6b1b552131804ebc918f30005b4d0204，无既存live guard。未重启或扩大OS重启/未来额度保证。

## 固定定义和来源

scripts/audit_head_matrix.py只读cache/features.npz（SHA 2a992048409c2691a67af4a6d2fcc125054f0ef5cb4c9088cb0cb46c7963d411）、cache_metadata.json（SHA e10cdf221a4da15783cdb3b5e8a488f654e7292da840008771361f0a742ed72c），外部SHA先于数组加载。复用原constraints函数和4个当前GPU强有益训练pair；train normal seed100000..3、validation100004..7，各316state/4相关episode。validation不参与谱、pair投影或必要下界。

事前固定quantile=[0,.01,.1,.5,.9,.99,1]；near-zero为行maxabs≤1e-12；SVD machine秩阈值=max(m,n)·float64 eps·smax，另1e-8·smax相对秩只描述、不选择拟合参数；pair投影使用train machine-rank行空间。Tanh饱和定义abs(h)≥1−1e-6。必要L1下界由 `row·d≤||row||∞ ||d||1` 得 `max(rhs,0)/||row||∞`；精确零行且rhs>0只对该固定LP构成矛盾，不视作其他表示或模型不可学。

## 结果

正常train差分矩阵 **14769×64，machine rank64，1e-8相对rank64**。最大/最小奇异值509.9469313227543 / 2.6817782069413662e-5，machine阈值1.67230812058289e-9；未缩放矩阵二范数条件数 **19015253.759719424**。每列L2范数最小3.420802944937972e-5、最大204.8000993339931，存在明显尺度差异。完整64个奇异值和列范数保留在matrix_result.json。

14769个train正常行没有精确零行、没有near-zero行（最小row maxabs0.14080214500427246）；4pair也无零行/正rhs矛盾。全部公开h的1936896个元素中1159923个按事前阈值Tanh饱和；统计含4pair重复初始lane，但重复lane不进入训练矩阵两次。

| 正常参照 | train | validation |
|---|---:|---:|
| state / episode | 316 / 4 | 316 / 4 |
| 竞争动作行 | 14769 | 14455 |
| 最小原margin | 0.0732952632255639 | 0.006987691550799369 |
| 竞争margin中位数 | 19.708145719519607 | 20.009856295206436 |
| 每state最小margin中位数 | 2.0322549461132997 | 2.142767221058648 |
| 负margin / ≤1e-3 margin行 | 0 / 0 | 0 / 0 |
| float64重建mode与原GPUteacher不一致 | 0 | 0 |

| 训练pair seed | 原good−base margin | 为达1e-3所需row·d | 必要||d||1下界 |
|---|---:|---:|---:|
| 992000 | -6.143981344046066 | 6.144981344046066 | 4.234073469922266 |
| 992004 | -0.5547557844086057 | 0.5557557844086057 | 0.2778794325544556 |
| 992006 | -3.377430316120899 | 3.378430316120899 | 1.964152318130663 |
| 992008 | -0.49601465976630266 | 0.49701465976630266 | 1.0778048065707813 |

四pair方向到train正常行空间投影残差约1.1e-15..5.2e-15，符合全列秩；全约束必要L1下界4.234073469922266。这不是可行解或系数拟合，normal不等式可能增加实际代价。满秩意味着在此固定表示与数值秩定义下，保持所有正常logit差分严格不变没有非零末层方向；**保持Top1只要求不等式，不能据满秩推不可行**。

## 资源与范围

unit source629307，15:27:30 controller passed，7passed/.11秒；audit source629420，15:27:59 receipt完成/15:28:00 controller passed。guard分别2.080/2.086秒，均100%CPU/2GiB/swap0、原82/90℃CPU、75/85℃GPU、六核亲和性和聚合250%/6GiB限制、最低可用1GiB、GPU telemetry及主机/目标/STOP保护。数据任务仍先查CUDA可用，无CPU模型或跨机fallback。

两个guard各仅1个采样，unit CPU38.375/GPU34℃/12.746MiB，audit CPU41.125/GPU36℃/16.266MiB，均hold。采样可能早于torch/数组加载，不当作真实峰值或完整内存用量。9/20项冻结输入完成后SHA相同，原640缓存及旧LP结果不变。

测试覆盖已知低秩投影/零秩、已知满列秩条件数、必要L1界和精确零行矛盾区别、validation不影响训练谱/pair界、非有限/空矩阵拒绝。未导入或调用HiGHS，未生成/应用候选系数。matrix_result eligible_for_training=false。

## 判断与下一步

数值尺度差异是已测事实，**不是已确认的LP超时原因**；没有零行矛盾或原normal排序/tie问题，不证明联合约束可行。原单次LP仍unresolved，不能延长、换solver/目标/margin重试。原模型/4k及8k棋力门槛结论不变。

新的学习方案缺少当前GPU验证的整局隔离偏好验证集：目前4个GPU pair都属于train。下一从旧Top-K固定validation seeds992011/992016/992027，事前按(step,state_id,action)每seed取最早1个旧beneficial，再按(seed,step,state_id,action)取剩余最早1个，共4case/3episode，封存为validation；若不足则报告，不补选其他seed。这些旧CPU标签已探索，不能称独立最终棋力确认。

复用已验收的8分支事务及保护，先查现存工作并冻结实际cases/来源/预算/完整RNG恢复；每case base/alt配对形成8环境，原30M CUDA greedy、5000上限/branch、全job≤40000唯一env步、0策略优化器。沿用256round有界phase、120command600wall、250%6GiB/原温控；本数据无需重复已完成512恢复oracle。GPU/host/额度不可用保留，不CPU/跨机fallback；按原compare区分强有益/有害/未支持和右删失，不淘汰失败替换，不把validation回流训练。新标签用于以后预先定义的方案评价，不启动PPO网格或推广。

证据runs/autonomous_head_matrix_20261107/{completion_audit.json,unit,audit}，小时真实执行回执hourly_head_matrix_execution_20261107.json。
