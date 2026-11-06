# 独立冻结策略数据与真实分段恢复

2026-11-06 14:19:48 collection_summary passed，14:19:49 restart_receipt passed，保护器/父控制器14:19:50 completed/MainPID0success。共16384个环境步，CUDA＋8环境、每worker2048步，4段各4096完整buffer。冻结原30M的policy、Adam、PPO更新数未改变，没有新质量模型或优化器更新。

## 数据与分组

原模型SHA `eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34`。新开发seed2600000–2600007，动作Torch RNG2700000，原max_pieces10000/奖励/PPO配方不变。固定预算，不按死亡数扩采样。

worker0–3为后续拟合组，worker4–7为验证组；初始及自动reset后的实际episode_seed均已保存并检查，两组没有共享episode seed。该拆分按worker/轨迹，不能将8192个相关step当8192个独立样本。动作来自共享的批量CUDA RNG，不声称各worker独立随机数流。

| 组 | 样本数 | 真终止/截断 | GAE残差RMSE | 解释方差 |
| --- | ---: | ---: | ---: | ---: |
| 拟合 | 8192 | 2/0 | 7.149209 | 0.585051 |
| 验证 | 8192 | 2/0 | 7.841563 | 0.192270 |

GAE残差=原策略GAE return减原value；目标仍含bootstrap，按4个512步窗口计算，不是完整Monte Carlo真值。这是数据诊断，不是策略质量评测。源模型历史代码SHA不可得的限制仍然存在；本次当前27训练来源与运行库已冻结。

## 持久化与恢复

每段目录包含rollout.npz、dataset.json、state.pkl和checkpoint.json。数据含完整观测/动作/mask/reward/value/return/advantage/log_prob/episode_starts、末bootstrap值/dones、逐step终止/孔洞/高度/episode_seed标记。每段无pickle数据回读、逐数组shape/dtype/SHA和SB3 GAE精确重算通过。

诊断恢复state.pkl只含本地产生且哈希验证后的8个环境状态、父进程Python/NumPy/Torch CPU/CUDA RNG、采样计数与末观测/episode_starts；没有保存或改写模型。完成数据与状态后才原子发布整段；中断片段保留，恢复从最近完整提交继续，拒绝身份变化、数据/状态字节损坏和已完成任务重复启动。

真实进程363063完成前8192步后退出；进程363174加载同一冻结源模型，从chunk2恢复，观测/mask/RNG/计数检查精确，继续剩余8192步。没有重采已提交样本。此次没有额外运行“连续采样vs恢复采样”重复对照，不据此声称做了该对照或模拟突然断电。2项受保护测试另验证真实SB3加载后的fresh seed、环境/RNG快照恢复后动作重放，以及身份/字节损坏拒绝。

已处理一个容易漏掉的初始化问题：SB3.load内部会用源模型seed42重新seed环境；本采样器在load后重新应用2600000起始种子，并在_setup_learn实际reset后检查所有worker的episode_seed及pieces=0。动作RNG也在加载/初始化之后设置。

## 硬件与后续

55个冻结输入结束后逐项未改。CPU峰75.25℃、GPU53℃、组内存3717.0MiB，全250%无降载；6GiB/swap0、原温控/GPU遥测和180秒命令/600秒队列预算保持，实际保护器40.956秒。所有有限服务现已终态，不要重启；原30M/main27/推理20/row masks保持。

下一可以设计仅副本critic拟合的可行性检查：只用拟合worker训练，验证worker仅检查最终结果，不选epoch/追加预算；原actor及其Adam状态冻结。必须区分“固定GAE目标拟合更好”和“策略质量更好”，不能以训练误差下降推广模型。计划中的单次有界可行性检查尚未执行。

证据：runs/autonomous_independent_critic_data_20261106/design.json、collection_summary.json、collection_progress.json、restart_receipt.json、restore_*.json、chunk_0001–0004/、resource_audit.json、summary.json、unit_guard/和guard/。本轮原生goal实际执行证据native_goal_independent_data_execution_20261106.json，不冒充新的小时消息消费。
