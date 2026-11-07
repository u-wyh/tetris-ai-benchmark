# PPO Top-K反事实偏好数据证据审计

## 去重与有限窗口结果

旧报告的325是跨1000/5000窗口、可重叠来源的强纠错比较，不是325个独立训练样本。本轮只固定5000窗口，按原规则重新读取分支逐步得分/四消轨迹，核对原1000→5000条件筛选（最多两替代动作），不重放、不新增环境、不加载模型。

541条1000步分支与349条5000步分支完全符合历史条件扩展；120个源状态中最终有232个5000步状态—替代动作比较。去重未发现重复或同公开输入冲突。它们同局相关，不能当232个统计独立样本；原19个死亡seed也是条件选中的已探索案例。

| 来源（可重叠） | 有益比较/状态/seed | 明确有害比较/状态/seed | 窗口内未获支持比较 | 双方均删失 |
|---|---:|---:|---:|---:|
| 全部 | 200/104/19 | 1/1/1 | 31 | 0 |
| PPO Top-K | 177/101/19 | 1/1/1 | 28 | 0 |
| BC | 26/26/14 | 0 | 6 | 0 |
| 同时Top-K与BC | 12/12/8 | 0 | 3 | 0 |

有益200与旧汇总的strong_5000_comparisons精确一致。复用原compare规则：原分支真死、替代至少多活250且≥1.5倍、共同存活得分≥95%、原共同四消≥3时保留≥90%。有害是交换两分支后也满足同一强规则，**未获支持不等于负例**。所有长分支经1000窗口存活/得分门后择优延长，正例比例和有害稀少不能当无偏发生率或分类器充分训练证据。原有限候选/固定PPO续控/局部窗口不说明全局动作最优或长期策略质量。

## 真实性与限制

封存2238个输入、195.28MiB源文件，审计前后SHA全未变。原protocol的PPO=eb859b…、BC=5f5613…与现存冻结ZIP字节哈希一致。逐状态核对selection/candidate/meta/源episode trace SHA，快照同时核验外部SHA与旧meta SHA后才解压读取自己本地pickle；不直接读取未核验状态。所有120快照的episode/piece/cap/版本/公开hash、实际候选动作合法性通过。

从公开board/current/hold/next重建当前状态237 float32、候选1840×16 uint8与正式合法mask1840 bool，未推进Core、未读取策略或采样动作。输出public_inputs.npz仅有三类公开输入和state_id；preferences.json的seed/step/结果为离线分组/标签，不能作在线特征。隐藏RNG只做来源一致性校验，没有导出到模型输入。

**分支模型身份与旧运行环境仍有缺口**：541/349分支均没有逐分支model_sha256。suite声明模型与当前ZIP一致，不等于每条旧分支运行时的独立证明。原suite commit b382a44b…保留，不重标当前31/20；没有原完整观测tensor记录，当前公开重建通过不冒充历史tensor逐位相同。结论为可追溯的条件性旧偏好证据/当前公开输入完整性，不把来源缺口填成验证成功。eligible_for_training=false，本轮不训练/推广。可先用于有明确限制的离线排序几何诊断，另需episode隔离、正常动作保持集与独立质量评测协议。

## 实现与验证

新增scripts/audit_topk_preference_evidence.py、19项tests/test_topk_preference_evidence.py（1.14秒），覆盖原阈值边界、删失/累计轨迹、对称有害、条件延长、双来源去重/跨窗口与冲突、两个seal先于pickle、真实Core公开输入静态重建、错误显式模型身份拒绝。旧反事实代码和数据未改。

unit guard11:18:56/controller57 completed，2.090秒/48输入/100%2GiB；正式guard11:20:17/controller18 completed，4.152秒/2238输入/100%2GiB/swap0、原温控/GPU遥测/内存/主机/目标/STOP/120秒command300秒wall。正式仅2采样，最高CPU58.5℃/GPU37℃/312.07MiB，unit单采样不能当真实峰；均hold。全轮**0新环境/0GPU forward/0optimizer/无模型加载**，不属于CPU训练fallback。所有有限service终态。

证据runs/autonomous_topk_evidence_20261107：protocol、unit/audit design/guard/controller、audit/evidence_audit/preferences/public_inputs和completion_audit。原生轮01a1b3fd-3c0d-7432-b319-8456c937cf79实际首工具11:11:29.081+08/call_53098648ff53444d97c20cd453dbe8fa，native_goal_topk_evidence_execution_20261107.json记录开始/结果，不代表新hourly已消费。

下一先在冻结原策略上检查Top-K偏好在最后线性评分层的固定梯度方向，与正常原策略状态的排序保持是否冲突；只做CUDA前向/代数，不改参数或开始PPO。该新方向与旧BC全actor纠错或失败LR/KL扫描不同，仍不能因有限数据看起来正例多就宣称可提高棋力。
