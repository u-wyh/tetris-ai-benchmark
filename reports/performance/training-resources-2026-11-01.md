# CPU/GPU 训练资源配置与短测

资源策略见 `configs/training_resources.json`；`scripts/resource_guard.py` 将命令放入独立的 systemd 用户服务，限制整组训练进程及其子进程。

- 8 核 16 线程的 Ryzen 7 6800H：保留两个物理核心，选取其余六个物理核心各一个逻辑 CPU（当前为 4、6、8、10、12、14），避免将四个 worker 挤在两个物理核心上。
- 初始 CPU 配额 200%，可在持续低温 30 秒后逐步升至 300%；高温时每次减少 50%，最低 100%。配额按一个逻辑 CPU 为 100% 计算。
- CPU 82°C / GPU 75°C 开始降低 CPU 配额；CPU 90°C / GPU 85°C 停止整个任务进程组。每两秒采样。阈值为保守运行策略，不代表硬件损坏界限。GPU 高负载时降低 CPU 配额不保证立即降低 GPU 温度，因此另设停止阈值。
- 内存上限 6 GiB，禁止该服务使用交换区；系统可用内存低于 1 GiB 时停止。服务有独立的运行时间上限。
- 中止时保留训练器已完整提交的事务。当前正在执行的事务可能回滚，资源保护本身不创建新的模型快照。恢复仍使用现有训练器的身份和哈希校验。

## 使用方法

```bash
.venv/bin/python scripts/resource_guard.py \
  --output runs/my_training_resource_guard --seconds 3600 -- \
  /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python -u \
  -m training.train_vector_transaction \
  --run-dir runs/my_training --config-file configs/my_training.json
```

`--output` 必须使用新目录。已有训练恢复时沿用其冻结配置并加 `--resume`；不能直接将 CPU 断点的设备配置改为 CUDA。该命令的保护只对其启动的训练及子进程生效。

## 校准证据

固定 PPO30M 初始权重、候选策略、默认奖励、缓存环境、相同训练超参数；CPU4 与 CUDA4 使用相同四个性能基准种子。CUDA8 增加为八个种子、保持每事务总采样 4096；轨迹因此不同，不作质量对照。每种配置完成两个事务（8192 步），推荐组额外恢复一个事务。结果只用于选资源配置，不提升或替换正式模型。

三组短测和恢复结果：`runs/resource_calibration_20261101_retry1/summary.json`。

首次隔离服务启动时系统用户服务缺少 shell 的 Node 路径，训练前检查失败；补充显式传递 PATH 后在新目录重跑。原始失败证据保留在 `runs/resource_calibration_20261101/`。

保护逻辑的定向测试：`tests/test_resource_guard.py`；实际超时与进程组清理测试：`runs/resource_guard_deadline_smoke_20261101/`。超时测试预期为非零退出，不是训练故障。

持续测试结果见下文。采样峰值受两秒采样间隔影响，不能视为硬件历史最高温度；短测不代表已验证数小时运行。若外部强制杀死监控器，实时温控会中断，systemd 的配额、内存和时间限制仍然保留。

## 实测结果

| 配置 | 训练步/秒 | CPU采样峰值°C | GPU采样峰值°C |
|---|---:|---:|---:|
| cpu4 | 290.6 | 75.8 | 47.0 |
| cuda4 | 366.0 | 70.1 | 52.0 |
| cuda8 | 409.7 | 72.2 | 52.0 |

初始相同200%配额下，CUDA8短测吞吐比CPU4高41.0%。持续测试共65536步、16个完整事务，运行136.5秒；CPU/GPU采样峰值80.8/56.0°C；最近一分钟CPU范围74.8–80.8°C。自动配额从200%升至250%后保持，平均训练521.5步/秒，最后四个事务551.8步/秒。持续测试训练轨迹更长且配额改变，不将其与短测的差值全部归因于GPU。16个事务身份/文件哈希及冻结模型未变检查通过；短测中另有一次4096步恢复通过。

推荐CUDA＋8环境，使用资源保护启动；配置记录见`configs/training_resources_recommended.json`。所有校准进程已正常结束，没有启动新的长期训练。

## 后续配置更新

本文记录首次200%起始、最高300%的历史校准。后续六组对照后，当前推荐已改为GPU＋8环境、初始与最高CPU配额250%。最新对照见 `reports/performance/training-resource-comparison-2026-11-01.md`，以当前配置文件为准。

2026-11-05已修复上文提及的“监控被强制结束后子任务可能继续”的缺口：增加pidfd与30秒心跳看门狗，真实systemd测试已确认监控被SIGKILL后子服务停止。证据见 `reports/experiments/autonomous_development_20261105.md`。
