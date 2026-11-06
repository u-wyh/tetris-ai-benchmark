# 可恢复自然greedy上下文快检

## 实现与范围

新增`policy_contexts.py`、`evaluate_policy_contexts.py`及两checkpoint校准脚本。固定8个完整上下文、每个512本地决策、max_pieces20000；模型自然greedy，**没有强制首动作**。死亡/TimeLimit停槽，不自动reset；存活到512为horizon删失，环境状态保留但不继续推进。

checkpoint load(env=None)后克隆保存状态，避免seed42重置；local steps/奖励独立从上下文起算，不将已有六千余步prefix当成新工作量。每256全局轮原子提交8worker/Core/bag/游戏及NumPy/action RNG、active观测/mask摘要、父CPU/CUDA采样RNG、实际动作/所选特征/概率及outcome。恢复直接使用已提交末状态，不重采前缀。

输入预检核验ZIP、suite、每context的外部manifest/state SHA及身份。错误ZIP/suite/context或“新有效ZIP+旧缓存”在pickle/数值输出前拒绝；设计/源码/数值runtime/compiled产物/精度和固定槽序均纳入身份。policy/完整Adam/源计数和采样RNG不变，0optimizer。

本suite来自同一开发episode的8个相关历史状态。`evaluation_summary.status=completed`仅表示诊断计算完成，**不是全局质量通过或自动推广**。旧模型训练来源保持原样，当前31训练相关文件/20推理文件仅是本评测运行来源；它们和旧捕获/恢复/四臂源码均未改。

## 定向验证

5项测试通过，包含真实forkserver自然首动作、已有prefix的局部计数、真正死亡槽跨新进程保持停止；错ZIP/suite/context和修改有效checkpoint后复用旧缓存均在pickle或输出前拒绝、前缀不变。单测含少量模拟，正式计算另计。

校准suite与job目录：`runs/autonomous_context_regression_20261106/`。每模型真实运行两个进程：首256轮提交后退出，新进程load checkpoint并恢复worker/RNG，再完成第二段；初始和首段文件SHA未改。

| 冻结checkpoint | 实际新环境步 | GPU批次 | 512步删失 | 死亡 | 真实进程 |
|---|---:|---:|---:|---:|---|
| 原30M `eb859b…` | 4096 | 512 | 8 | 0 | 431093→431170 |
| 拒绝8k `016b6b…` | 2154 | 462 | 0 | 8 | 431249→431305 |

合计6250新环境步、974 GPU批次、0optimizer。两者自然首步和每步全部旧A/D trace字段、完整末worker状态精确；8k额外逐步pre-observation/mask/selected feature与原source D一致。8k死亡动作数为462/351/311/290/244/202/160/134。所有模型policy/Adam/计数/RNG不变；正常退出后的真实断点重载已测，不宣称OS重启或任意断电恢复已实测。

## 已有4096 checkpoint的新诊断

校准通过后先核对已有任务1 ZIP及原manifest，未重训：

- ZIP SHA：`9a11ce79e7ef32c8a2dfa88a713422503cd6635082ef9d4b244eaa854dc65b3d`。
- ZIP内真实计数4096、PPO更新36630（不改写成30007296）；原任务manifest的SHA和2669203字节尺寸一致。
- 自然同suite：4096新环境步/512 GPU批次，**8个均达到512删失，0死亡**，未改变模型或优化器。

证据：`runs/autonomous_context_probe_4k_20261106/`。这将**本固定suite上的**退化定位在4096至8192两个既有checkpoint之间，不证明4096版本整局质量达标；状态由8k失败轨迹选择，存在开发集选择偏差。已搜索原actor probe、actor8k质量根记录及现有quality candidate JSON，未找到该ZIP对应完整20k评测，不声称它已经通过完整质量门。

## 保护记录

| 阶段 | guard/controller完成 | 秒 | 配额/内存上限 | 冻结输入 | 采样CPU℃ | 采样GPU℃ | 采样内存MiB |
|---|---|---:|---|---:|---:|---:|---:|
| 5项定向测试 | 20:32:13/20:32:13 | 8.198 | 100%/2GiB | 196 | 51.5 | 36 | 740.98 |
| 两模型校准 | 20:37:04/20:37:04 | 43.124 | 250%/6GiB | 216 | 73.75 | 52 | 3482.64 |
| 已有4k诊断 | 20:40:28/20:40:28 | 16.409 | 250%/6GiB | 219 | 67.125 | 43 | 3190.13 |

全部完成、冻结输入未改、未触发降载。2秒间隔采样峰值不是连续真实峰值。GPU8/原温控/内存/时间预算保持；GPU不可用拒绝，无CPU训练或其他主机fallback。快检与完整质量评测工作量不同，不把本表当作等价质量评测加速比。

本轮为20:24:19实际首工具的原生目标续轮，前轮`3647e2b`为实质进展。执行receipt：`native_goal_context_regression_execution_20261106.json`；20:44:55后台healthy/goal active，20:46:45 queue_empty。未声称消费了新小时消息或未来额度持续可用。

## 使用

每个新checkpoint新建job目录，参考`runs/autonomous_context_probe_4k_20261106/design.json`，固定新ZIP的path/SHA和真实计数（`expected_counts`可省略，实际计数始终报告），保留已校准suite/源码/数值runtime/compiled身份并更新`frozen_files`。suite不能含重复key，旧缓存不能用于新checkpoint。

通过现有受保护controller，或明确包在资源保护器内运行：

```bash
.venv/bin/python scripts/resource_guard.py \
  --profile configs/training_resources.json \
  --output runs/new_context_job_guard --seconds 180 -- \
  .venv/bin/python scripts/evaluate_policy_contexts.py \
  --run-dir runs/new_context_job --max-new-chunks 1
```

恢复同一个job时使用新的guard输出目录，保持原design/ZIP/suite不变，再执行最多1或2个新chunk。完整job不应重复计算；每个模型唯一新模拟步上限4096。当前支持本项目candidate格式PPO和8槽/512步suite。本机数值环境发生变化时应新建job并重新校准。

下一做同一GPU8协议的完整开发评测，对比原30M与已有4096版本；先读取现有未降低的质量门，不能把此快检代替完整/独立seed验证或直接用于模型推广。
