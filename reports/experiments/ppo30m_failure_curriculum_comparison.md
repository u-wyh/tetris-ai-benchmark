# PPO 30M 危险课程完整对局比较

开发评测：相同 16 个全新种子、每局 50K，全部从空棋盘开始。

| 方案 | 平均存活 | P10 | 触顶 | 平均总分 | 每千块得分 | 四消比例 |
|---|---:|---:|---:|---:|---:|---:|
| ppo | 47496 | 44050 | 14/16 | 3658046950 | 77017798 | 92.95% |
| B12 | 45178 | 33518 | 12/16 | 3355688981 | 74277470 | 91.14% |
| normal | 36358 | 15080 | 8/16 | 2502386181 | 68827235 | 93.73% |
| curriculum | 37480 | 8648 | 10/16 | 2728255269 | 72792539 | 91.30% |

冻结门槛：`{'rules': 'four drop <=3pp, common score >=95%, P10 >=95%, mean +>=1000, no extra <10K death', 'decisions': {'normal': {'four_drop_pp': -0.7798085658424547, 'common_score_ratio': 1.0012106312119897, 'p10_ratio': 0.34234571684770887, 'mean_survival_gain': -11138.625, 'early_death_excess': 1, 'pass': False}, 'curriculum': {'four_drop_pp': 1.6497634354421642, 'common_score_ratio': 0.9978868721969195, 'p10_ratio': 0.1963314831840728, 'mean_survival_gain': -10016.25, 'early_death_excess': 2, 'pass': False}}, 'extend_curriculum': False, 'extend_normal': False}`。触顶局按右截断；开发组轻微差异不代表独立泛化优势。

开发门槛未通过时，不启动独立确认；不把未评测说成训练失败或胜出。

逐 seed、配对共同得分、策略保持样本和模型 SHA256 在 `runs/ppo30m_failure_curriculum/`，大型模型未提交 Git。
