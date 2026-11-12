# 全合法动作CE：训练纠错2/8，验证0/4；固定Adam终点非收敛证明

## 事前设计与实现

前一8episode pair目标仅1/8全局纠正，3/8领先base中2例输第三动作。单独固定新目标为全legal动作交叉熵；其余原8episode数据、batch8 old/new交替、normal chunk=t%79、fresh d0/Adam lr.01/betas(.9,.999)/eps1e-8、256更新、KLβ1/L2.01和GPU float64/最终float32不变。不从失败checkpoint续训，不扫超参。CE指向已核验的强有益动作，并不把所有其余合法动作宣称为负反事实结果。

validation4case/3episode仅最终一次、episode等权pair指标及全局Top1准入，已作开发反馈非最终独立确认。旧eligible_for_training=false不改，限定单试验准入。候选全局棋力门槛≥.99pieces/.99score、死亡不增/minseed≥.9保持。

旧640、val8及新block1八源状态h/mode全部外部SHA/readiness/原完整完成receipt封存复用；0源forward。新CUDA候选train16/2batch＋val8/1＋normal632/79=82fullforward。1原model load用于真实完整候选mode，无新环境。旧8episode helper与测试归档before_changes。

## 实际GPU终点

|指标|8episode pair|8episode CE|
|---|---:|---:|
|train good全局Top1|1/8|2/8（case91、93）|
|train pair领先base|3/8|3/8|
|val good全局Top1|0/4|0/4|
|val pair领先base|0/4|1/4（992027，仍输第三动作）|
|train episode等权pair loss|.8364714|1.1753568|
|val episode等权pair loss|1.5287665|1.4768619|
|normal train mode变化/316|96|77|
|normal val mode变化/316|104|79|
|normal train teacher KL|.4659484|.3419302|
|normal val teacher KL|.5073691|.3856997|

两个loss目标不同，pair loss只作统一评价，不能将其直接当CE优化值。验证preferred raw ranks2/4/2/3，0/4实际全局Top1，**仍失败**；无evaluation_candidate.zip、无完整棋力评测/推广，不续跑固定256Adam或扫参数。normal变化不等于棋力损失；训练改善不代表泛化或棋力改善。

source1832010于16:14:36完成，controller16:14:40终态inactive/success/MainPID0。0source＋82candidatefullforward，另256headforward/256freshAdam步骤，1model load/0env/0PPO optimizer。源完整policy/Adam/RNG/ZIP/实际30003200与36625及31训练20推理源不变；候选只末层64weight变化。d L1=22.58075415/L2=4.49826163，重建max4.086083e-6＜1e-4。resultSHA `d4c0899423855d453c46a99d4c3afdbf4e0fefa51521a87133410a16b7b96598`，checkpointSHA `2659b4ee876fa2d542ae545f2c4e89655cc2aa57bc37587e9338456705c8a82e`。

## 单独只读终点审计

scripts/audit_head_endpoint.py只加载封存缓存及已完成head断点（weights_only/map_location=cpu；没有CPU策略模型或优化器）。serializedAdam实际step256、唯一64维float64参数/一组moments、固定超参、PPO源计数、coefficient逐值相等通过。

按实际256调度出现频次构造经验目标：8pair每个权重128/1024=1/8；normal每个chunk0..18权重4/1024、19..78权重3/1024，共316trainnormal；不把重复曝光当新独立episode。无任何val梯度。

|训练经验目标|d0|终点|
|---|---:|---:|
|全动作CE|5.0760038757|1.8483940459|
|normal forward KL|0|.3440839659|
|.01 L2|0|.2023435769|
|合计|5.0760038757|2.3948215887|

终点梯度L2 **.1540731153**，不是梯度归零。由于.01||d||²使固定经验目标强凸系数.02，理想精确算术下目标次优界≤||g||²/.04≈**.5934631**、到唯一最小点距离≤||g||/.02≈**7.70366**。此界相对当前d范数4.498仍宽，不能证明已充分收敛，也不证明继续优化会改善验证。这里是float64诊断，非区间认证或泛化保证。终点使用未转float32的d；实际GPU模式另用最终float32权重审核。

审计audit_result.json于16:19:36 passed/controller37终态；0模型load/策略forward/env/优化器。3额外测试/1.17秒验证pair和CE解析梯度有限差分、真实频次及validation扰动隔离。该audit不重复新候选训练。

## 保护与工件

25训练工具定向测试/1.25秒通过（第三动作梯度、padding/极值、来源完成/外部封存才复用，以及旧4/8兼容），另3数值审计测试。4个guard全completed/hold；正式8.188秒4采样最高CPU66.375℃/GPU39.0℃、cgroup1043.363MiB，采样非真峰。原250%6GiB/swap0/六核/aggregate/温控/GPU及lightbook/goal STOP/120command600wall；测试和纯数组审计100%2GiB。15/332/17/346冻结输入各阶段完成后全部相符。

根 `runs/autonomous_head_global_ce_20261112/` 包含事前protocol、unit/probe/audit_unit/audit、completion_audit。旧8pair结果已提交c785318，不覆盖旧缓存/结果。小时消费turn及首工具见hourly_head8_execution_20261112.json；本轮工具实际执行，不拿accepted或keeper健康代替开发证据。当前桌面proxy存在，不称本轮新无桌面/OS重启验收。

## 下一有限实验的理由和范围（未开始）

先前256Adam终点不支持把验证失败归结为已收敛的固定head最优解。下一应隔离**优化算法**，而非继续改变LR/β/训练数据或调validation：事前封存一次fresh d0、同一频次加权CE/KL/L2目标的CUDA float64阻尼Newton求解；用64×64 Hessian＋.02I、train梯度无穷范数≤1e-6停止，最多32个接受Newton步骤，每步最多12次Armijo(c1e-4、halving.5)回溯，失败/预算到则停止，不提高预算。不会恢复/延长已封闭Adam checkpoint。

所有梯度/Hessian/试探loss均CUDA batch8、缓存公开h，精确保留原固定调度的经验权重；最后不足8补mask零权重，不能混入val。每个完整训练目标pass=8pair＋316normal填充至328状态/41batch；完整pass/求解/line-search次数分开计数，无PPO优化器。只有预定收敛终点才做一次完整候选82forward；未收敛如实报告，不看val挑步数。原源state/计数不变、val仅终点和原准入/全局棋力门槛保持。先测试解析Hessian/方向/Armijo失败和计数边界，再按原120command/250%6GiB/温控/GPU8/主机STOP执行；一次有限新算法对照，不构造参数网格，不宣称收敛即泛化。GPU/host/额度不可用保留既有小时机制，不CPU/跨机fallback。
