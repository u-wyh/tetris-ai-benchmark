# 完整向量化清行特征：一致性通过，静态速度未过门

隔离副本 `/home/wyh/projects/tetris-vectorized-feature-validation`，分支 `codex/vectorized-candidate-features`，提交 `a2efe9d`，显式 bitboard_vectorized/compact，训练来源28。main仍27；推理20在隔离/main/准确归档43cc26f逐项SHA一致。

## 改动与验证

在新的来源中批量压缩满行、将保留行移至底部/顶部补0，并用整数行权重重建列。visible=false的放置不会清除底板已有满行。底板holes/max直接整数统计，正常合法清行不回到Python参考函数；异常输入仍参考回退以保留原语义。

133项受保护差分/配置测试通过：128种独立逐格清行压缩/列重建、正常清行及全不可见不调用参考函数、随机20-bit列/2的幂边界、完整矩阵/mask/非法行零、负y/顶出/非连续多清行、异常与溢出/重复cells/负action索引、200步轨迹与快照恢复。

固定3份快照24状态加3边界，共27状态；固定相同预计算落点，只计特征构造。30对AB/BA，38冻结输入未改。不是完整训练吞吐。

| 指标 | 结果 |
| --- | ---: |
| 配对特征速度比中位数 | **1.378821** |
| 最小配对速度比 | 1.351846 |
| 参考每批CPU秒中位 | .0121766305 |
| 候选每批CPU秒中位 | .0088322285 |
| 预声明中位门槛 | **1.8，失败** |

完整feature/mask一致、每对更快，但eligible_for_gpu_benchmark=false。保留负结果，不降低门槛或重计时，不合入main，不启动该版CUDA8对照、真实错误feature恢复或完整上下文扩展/故障恢复；这些均未测。没有训练或环境推进/模型更新。旧batch/delta/fused失败路线保持停止。

## 资源和无人值守证据

测试CPU峰60.625℃/GPU36℃/内存300.26MiB，静态41.75℃/37℃/11.16MiB；CPU100%/2GiB及原温控/GPU遥测，均completed，无降载。静态15:06:33完成、保护器15:06:35完成；133项测试保护器15:05:39完成。未将遥测GPU可用称为本轮CUDA训练。

本次15:01:55小时请求对应新轮01a1afab-8872-7cb3-8449-3c075c6a9290，15:03:36开始、15:03:47首个工具执行。匹配依据为唯一待办及相同提示/新轮/queue清空，历史submission/clientId并非相同ID。15:09:55后台RPC/goal active、4服务定时器enabled/active、linger、防休眠正常，桌面proxy为空。accepted本身不作为执行或额度恢复证据；未测试OS重启或整夜连续运行。

证据：runs/autonomous_vectorized_features_20261106/{runtime_archive.json,unit_guard/,micro/design.json,micro/microbenchmark.json,micro/qualification.json,resource_audit.json,inference_audit.json}；runs/autonomous_development_20261105/hourly_vectorized_execution_20261106.json。

## 后续

停止这组纯Python/NumPy特征微调。下一先检查既有原生整数特征内核；如无，可在新隔离副本做C完整特征原型，仅研究可行性，先固定编译器/参数/C源和二进制SHA与ABI，再差分和一次预声明静态门。当前没有该原型或性能结论，不改变训练配置。静态合格前不引入训练后端；合格后仍需严格运行时身份、GPU8/250%端到端和跨进程恢复验收，才能考虑采用。
