# 位图评测与推理缓存身份验证

## 结果

本机两组AB/BA、每次2worker×2seed×4096块、固定CPU150%的受保护命令完成计时：

| 配对 | official整条命令秒 | bitboard整条命令秒 | 吞吐比 |
|---|---:|---:|---:|
| 0，official→bitboard | 25.301 | 23.201 | 1.090513 |
| 1，bitboard→official | 25.293 | 23.199 | 1.090263 |

几何平均吞吐提高9.0388%，耗时约减少8.29%。计时包括导入、模型加载、spawn、guard启动和轮询；不把特征微基准当作端到端收益。两组都更快且达到预声明的≥1.05门槛，全部采样保持CPU150%。这是本机这组有限评测的证据，不是所有训练或长局任务的统一加速比。

全部逐局字段、聚合及孔洞诊断与归档旧评测器精确一致。另验证bitboard的单worker、spawn和legacy CLI路径结果相同；真实缓存尝试混入另一后端时在启动worker前拒绝，原缓存不变。采样最高CPU79℃、GPU47℃、内存786.10MiB。

证据：`runs/autonomous_evaluation_backend_20261105/summary.json`、`cache_isolation.json`、`legacy_reference.json`及各guard目录。无训练、最终测试或模型发布。

## 接入

两个评测器新增显式 `--candidate-feature-backend official|bitboard`，默认仍为official。推荐新实验加载 `configs/development_evaluation_bitboard.json`，或调用：

```bash
.venv/bin/python -m training.evaluation.evaluate_resumable \
  --model runs/ppo_candidate_raw_30m_seed42/final/model.zip \
  --seeds-file training/evaluation/seeds.json --seed-set validation \
  --seed-count 8 --max-pieces 20000 --workers 2 \
  --candidate-feature-backend bitboard --protocol development \
  --committed-steps 0 --output runs/new_development/baseline.json
```

重任务应通过既有resource_guard执行；以上只说明评测器参数。GPU8训练、总CPU250%/6GiB和原温控不变；评测使用已确认的2worker及较低配额。

筛选、额外验证及可选扩训控制器传递模式，要求基线与协议一致。跨模式结果不直接比较；模式、诊断、Python及torch/numpy/gym/SB3/contrib/cloudpickle版本都记录在结果/缓存身份中。旧数据没有重新标记为新方法。

## 推理来源闭包

新身份扫描所有本地静态import，包括函数内部、条件分支、相对import、包初始化；对由checkpoint动态载入的CandidateScoringPolicy登记显式根。当前resumable身份覆盖20文件，包含策略、观测、两个特征后端、核心规则、缓存及实际调用的公共工具。新增自定义动态策略需要登记对应根。

计算身份只用stdlib，不导入Torch或NumPy。无关训练器/初始化代码改动不再使推理缓存失效；实际推理依赖或新增包初始化改动仍会改变身份。对当前依赖图的改动检测、缓存及控制器传播共54项定向测试通过，包含离开项目cwd时的后台脚本导入检查。

## 完整长基线

`tetris-bitboard-baseline.service`于22:00:46成功完成新的8×20000可恢复基线。逐局及聚合与历史冻结模型结果精确一致，8/8均触顶。`runs/autonomous_bitboard_baseline_20261105/summary.json`通过，结果SHA为`02831deef2907f016404873432f9beda5257af8683508a6b2d68b5da83c4807c`。

后续修改纯训练器时，可复用这个具有当前推理身份的基线；模型、推理依赖、种子/上限、模式、诊断和运行库仍必须匹配。该长基线用于正确性及后续配对，不把与不同长度任务的耗时差当作性能证据。
