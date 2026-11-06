# 原30M KL锚定：固定缓存副本更新对照

记录：2026-11-06T21:31:53+08:00。诊断计算通过，**预声明方案门未过**；不启动相关训练，不提高beta重试，不推广模型。

## 先查重与本轮真实执行

小时submission01a1b0f7-4d16-71e0-b863-b958661cb953（21:05:59）已由新轮01a1b101-81c6-7311-be4e-18e4e4194f2f消费，userMessage.clientId匹配01a1b0f7-4d15-7d23-84ba-fd85c6e338c6；首exec21:17:18.206/call_ff2f3064c30a43eb9f786fb013d0288a，queue空。本机13827→1/桌面proxy空，四驻留及续办单元enabled active、linger yes/睡眠masked。真实工具已执行，未把accepted当执行；未来额度、OS重启或断电恢复范围未扩大。receipt：hourly_original_kl_execution_20261106.json。

已有原30M作reference的KL方案，不能称此前只试过BC参考：

| 历史方案 | 均存活比 | 均正式分比 | 结果 |
|---|---:|---:|---|
| fresh Adam β.03 | .879025 | .8566012942 | 首8seed失败 |
| fresh Adam β.3 | .7910375 | .6840136599 | 首8seed失败 |
| restore Adam β.3 | 1 | 1.0019875537 | 首8seed通过 |
| 上项额外8seed | .9709125 | .9487529084 | 100014/15346死亡，失败 |
| RNG隔离 restore β.3 | .89575 | .8811558348 | 首8seed失败 |

本轮并未重训上述262144步方案。新对照只检验同原策略固定数据、同完整Adam历史且清actor一阶后，默认β.02在一次克隆更新路径中的实际作用；历史初始化/数据/β均不同，不把跨实验成绩当因果对照。

## 冻结协议

复用既有4096样本、8worker的完整obs/action/mask/GAE数据，archive SHA124e0495b682d249942bc535a2b21667e2ad72b6a74934eb95f84bd9050ef3c1，dataset.json外部SHA及每数组shape/dtype/hash先核验，再精确重构已保存GAE。非二值/空mask、masked action、坏archive先验拒绝。源为eb859b…原30M，实际30003200/36625，历史训练源码未知的限制保留；当前31/20仅标离线诊断运行来源，数据仍保留旧来源。

两个副本保留源policy/完整Adam，每个仅清12个actor参数的exp_avg，保留二阶/step/group/critic。LR固定3e-5，5epoch/batch256各80步，唯一区别β=0或.02。使用**既有KLMaskablePPO**，不重写生产算法；reference是另一个独立、requires_grad=false的原30M policy/Adam副本，永不进入训练optimizer。env=None防止采新数据。原全局采样RNG保存/恢复，预声明新排列seed3500000、env-major flatten、五轮完整排列写permutations.json，**不声称恢复历史训练采样顺序**。

预声明诊断门：64个固定训练态上的最终和峰值KL(ref||current)均至少下降5%（比例<.95），greedy一致率不降；它是进入进一步研究的支持检查，**不是棋力门**。β0与普通SB3.train的参数/完整Adam/epoch精确等价在合成CPU检查中验收；正式CUDA两case均使用KLMaskablePPO.train，仅改变β，不扩大该CPU等价检查的范围。

9项测试passed/1.81秒，含真正SB3顺序更新、非空Adam历史、teacher和source隔离、actor梯度/critic无KL梯度、错误mask/action及archive在np.load前拒绝。测试不推进环境；两个人工模型warmup和两个4步克隆路径共**10次合成CPU optimizer步骤**，与正式160步分开计，不冒充原模型训练。

## 正式GPU结果

| case | β | 实际克隆更新 | 最终KL(ref||current) | 路径峰值KL | 最终greedy一致率 |
|---|---:|---:|---:|---:|---:|
| control | 0 | 80 | 0.000236243161492 | 0.000249838860441 | 1.0 |
| original_anchor | .02 | 80 | 0.000232590912694 | 0.000246806397632 | 1.0 |

最终KL比0.984540298332（下降约1.55%），峰值比0.987862325323（下降约1.21%），均未到5%。实际训练目标为KL(current||reference)，64态最终反向KL比0.984699219279，同样只有约1.53%下降。不能因两边greedy一致率都是1就判定整局稳定：4k完整评测已经失败，本探针只是64个训练样本。

最终同一256样本batch：锚定加权actor梯度范数0.00427446701，PPO actor梯度1.03233957，比例0.00414056297（约0.414%）；critic侧KL梯度为0。PPO+熵+critic的全局梯度范数141.537552，对应基础共同裁剪系数0.00353263137；该subtotal不包含锚定项。初始同权重的KL为0，锚定梯度约1e-9，属于近零数值量。只在本固定batch上说明此β锚定作用较小，不推断长期退化因果，也不从比例反推该用更大β。

所有源样本的value/log_prob重算最大差为6.103515625e-5/7.62939453125e-6，处于预声明容差（1e-4与1e-5及相对项），不是跨batch GEMM逐bit相同。源模型policy/完整Adam/计数、teacher policy/Adam、采样RNG逐项未变；所有克隆Adam计数均+80、PPO epoch各+5，源30003200/36625未动。**正式克隆更新160、原更新0、新环境步0**，steps.jsonl160行与两case最终步数一致。没有普通model.zip、模型推广或PPO环境训练。

## 保护与证据

所有执行lightbook，GPU诊断必须CUDA，无CPU/他机fallback；8worker固定数据离线使用，不声称本轮运行了8个实时环境。原CPU250%/6GiB/swap0/温控与GPU遥测/180秒命令600秒wall保持。

| 阶段 | guard/controller完成 | guard秒 | 配额 | 采样CPU℃ | 采样GPU℃ | 采样内存MiB |
|---|---|---:|---:|---:|---:|---:|
| unit | 21:24:29/21:24:30 | 4.120 | 100% | 54.75 | 35 | 375.52 |
| run | 21:25:24/21:25:25 | 8.179 | 250% | 58.50 | 42 | 1359.18 |

run64冻结输入完成后未改，unit和run分别2/4个样本全部hold，未降载。短任务采样不等于连续真实峰值。GPU进程PID442238，guard21:25:24/controller21:25:25终态success。当前主31、推理20及initialization中full-context与kl_reference互斥保护未改。

证据根runs/autonomous_original_kl_20261106：design.json、unit与run guard/controller、kl_audit.json、steps.jsonl、permutations.json、completion_audit.json。不重跑本窗口，不调整beta后翻案，不据此启动PPO训练。

## 下一工作

回到已确认的GPU完整评测实际开销：此前309797步评测的354个稳态（elapsed≥10秒）采样中，CPU used cores中位数0.9984507576，而预算为2.5核。代码own_policy_capture.advance_round按lane逐项同步env_method('capture_step', action, indices=[lane])。这与PPO训练已有并发step不同；**利用率和代码结构尚不证明它是性能瓶颈**。可在新独立wrapper中研究一次发送全部active worker动作、仍按固定lane顺序收集结果；保留旧冻结helper/捕获文件不改。先查重，再预声明有限同状态ABBA GPU对照，验证逐步action/观测/mask/selected feature/outcome及完整worker/RNG/模型Adam精确、停槽/跨进程恢复无重算，并以整体GPU评测round的配对几何加速≥1.05决定是否值得集成；不把局部通信/CPU时间改善当训练加速。GPU8/CPU250%/温控与有限模拟预算保持，失败就保存证据，不重做计时凑门槛。旧IPC .0726秒上界研究是训练侧，不能把它当已测试本串行评测调度；也不能仅凭本假设推翻其结果。
