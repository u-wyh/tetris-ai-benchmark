# BC Champion 得分强化学习

更新时间：2026-10-27T21:28:22+08:00；停止/运行结果：`stopped_gate_32`。

实际新增训练步：**131,072**（包含 Critic 预热）。

历史 Raw Reward 的消行部分与正式得分除以等级近似相同。本轮实质区别是重新初始化与新目标匹配的 Critic、前 16,384 步只更新 Critic、死亡惩罚 -20、较小学习率/clip/epoch 以及固定 BC KL 0.1。正式计分和游戏规则不变。

历史 KL-PPO 2M checkpoint 已核对加载；旧 20K 平均分 331,629,591，BC 333,592,084；旧 100K 平均存活 85,658 对 BC 99,320.5。旧归档无逐块共同窗口分数，不能声称旧方案提高了效率；本轮重新配对测量。

候选 checkpoint：`/home/wyh/projects/tetris-ai-benchmark-bc-score/runs/bc_score_rl/checkpoints/task_000032/model.zip`；SHA256：`3b7b7dae96dfad401a9a556432797d55f9c243001c6e9cf2f49ba28e96b1ae77`。

## task_32_20k

|模型|平均存活|P10|触顶|<10K|平均正式分|四消比例|
|---|---:|---:|---:|---:|---:|---:|
|bc|20000.0|20000.0|16/16|0|333890719|0.03%|
|historical_kl|19206.5|20000.0|15/16|1|321215831|0.15%|
|ppo|19274.6|18661.5|14/16|0|596724656|93.36%|
|B12|16033.0|5412.5|11/16|4|472995562|91.09%|
|candidate|20000.0|20000.0|16/16|0|334029188|0.03%|

候选共同存活窗口正式得分比 BC：1.0004，seed 配对 Bootstrap 95% 区间 [0.9983619048480583, 1.0024668384486881]；正向 seed 8。
每千块得分、推理时间及风险指标见 runs 中的同名 summary.json。达到上限为右截断。

|Seed|共同块数|共同窗口得分比|候选−BC存活|
|---:|---:|---:|---:|
|1561000|20000|1.0010|0|
|1561001|20000|0.9958|0|
|1561002|20000|0.9998|0|
|1561003|20000|1.0025|0|
|1561004|20000|1.0061|0|
|1561005|20000|0.9984|0|
|1561006|20000|0.9995|0|
|1561007|20000|0.9918|0|
|1561008|20000|1.0028|0|
|1561009|20000|1.0033|0|
|1561010|20000|1.0032|0|
|1561011|20000|1.0012|0|
|1561012|20000|1.0090|0|
|1561013|20000|0.9993|0|
|1561014|20000|0.9941|0|
|1561015|20000|0.9990|0|

门槛 `decision_32`：`{
  "passed": false,
  "checks": {
    "mean_survival": true,
    "p10": true,
    "early": true,
    "common_score": false,
    "cross_seed": false,
    "high_board": true,
    "high_streak": true,
    "new_holes": true,
    "short_survival": true,
    "short_cap": true
  }
}`

门槛 `decision_32_short`：`{
  "passed": false,
  "checks": {
    "mean_survival": true,
    "p10": true,
    "early": true,
    "common_score": false,
    "cross_seed": false,
    "high_board": true,
    "high_streak": true,
    "new_holes": true,
    "short_survival": true,
    "short_cap": true
  }
}`

独立确认通过：**False**。仅开发结果不足以登记新冠军；未启动的确认组继续保留。

恢复：在冻结源码和模型哈希一致时运行 `python -m training.bc_score_rl`；已原子提交事务和对局只校验、不重复计算。语义门槛失败属于正常结束，不应恢复扩大预算。

训练遥测：`runs/bc_score_rl/training/metric_history/`；协议、配置、完整逐 seed 配对、错误与测试日志均保存在 `runs/bc_score_rl/`。
