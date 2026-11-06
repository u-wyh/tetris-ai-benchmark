# C特征训练验收与采用

隔离实现 `/home/wyh/projects/tetris-native-feature-validation`，原型d229934、训练集成66f3266。主副本采用与测试版本**逐文件相同的31项训练来源**；旧27项准确归档43cc26f保留，旧断点不重标来源。推理20项在主副本/隔离/归档逐项SHA不变。

## 训练接入与保护

显式bitboard_native，要求普通fast Candidate、bitboard_compact搜索和固定native_feature_runtime声明。父进程和每worker校验manifest SHA、C/wrapper SHA、二进制SHA/ABI；完整编译器/参数/平台身份存入config、run metadata和task metadata。普通旧后端无声明不加载.so或依赖编译器。

恢复/扩展在锁或目录写入前拒绝改变的feature/编译身份；已有source校验也移至写入前。31来源加入C、wrapper、运行时校验及扩展逻辑。旧来源恢复使用准确归档，不能删改旧metadata以继续。

49项定向检查最终整体通过，包括真实环境200步轨迹/cache/state、config/metadata/task身份、默认无artifact兼容、source闭包，以及拒绝错误后整个运行目录文件逐字节不变。**首次49断言虽过，整进程因测试原地截断已加载.so触发SIGBUS(-7)，记为失败。** 仅将夹具改为原子替换后重跑，15:28:06整体passed/退出0；原失败证据保留。

## 一次预声明真实GPU对照

同来源31、同编译身份声明，控制臂bitboard、候选臂bitboard_native，均compact搜索/CUDA8。seed2900000/2900100，AB/BA顺序，每臂8192、4096处跨进程恢复；未重新计时或放宽门槛。

| seed | 控制训练秒 | C训练秒 | 速度比 |
| --- | ---: | ---: | ---: |
| 2900000 | 13.350 | 12.662 | 1.054336 |
| 2900100 | 13.476 | 12.551 | 1.073699 |

几何速度比**1.063973508，约提升6.4%**，通过预声明≥1.05且每对更快。8个任务全250%无降载；policy/Adam/PPO计数/trainer RNG/8worker及非计时指标逐项精确。排除指标仅wall_seconds和gpu_peak_allocated_mib。计量为任务训练耗时，不含进程启动/检查点IO；两组短程本机证据，不保证长程同幅提升。静态6.94倍仅特征内核，不能用于整体训练结论。

实际错误feature及requested manifest SHA恢复被拒绝，整真实运行目录SHA不变；另克隆checkpoint并配置损坏的复制二进制，确认在执行C之前拒绝，克隆/真实运行文件SHA均不变。没有破坏冻结原artifact。58冻结输入未改。

## 完整上下文扩展/故障恢复

上游合格后条件服务15:32:16实际开始；test_extend_candidate.py -k bitboard_native：**1 passed, 5 deselected，77.32秒**，15:33:36保护器completed。合成源完整上下文恢复，完成短源后扩展至12288，task2注入中断，再恢复；与连续控制的policy/完整Adam/trainer RNG/worker状态/PPO计数精确，原上下文偏移和lifetime steps正确。60冻结输入未改，全250%无降载。这不是原30M长程质量训练或突然断电测试。

| 阶段 | CPU峰℃ | GPU峰℃ | 内存峰MiB |
| --- | ---: | ---: | ---: |
| GPU配对及核验 | 72.375 | 53 | 3585.97 |
| 完整上下文扩展恢复 | 70.625 | 51 | 3884.29 |

原82/90℃ CPU、75/85℃ GPU软/硬温控、6GiB/无swap/时间预算保护保持，采样峰不等于连续物理温度上界。

## 主副本采用

两门通过后复制准确代码，guarded模板显式设为native并固定本机artifact manifest路径/SHA；GPU＋8环境和资源配额不改。主副本受保护加载检查确认31来源、编译运行时身份与测试版本一致，实际native reset feature/mask与控制完全一致。原30M模型SHA保持；**未启动模板中的2M训练，没有质量模型推广**。该模板依赖本机固定artifact；丢失/不匹配即拒绝，不能隐式重编或回退训练。重新编译必须另存manifest并重新审计/显式配置，旧检查点继续使用原固定产物。

后续回到质量退化诊断，复用现有独立持久化数据，避免盲目LR/GAE/KL/critic网格或重复采样。此次采用仅提高安全配置下的吞吐。

证据：runs/autonomous_native_features_20261106/{integration/guard/watchdog_result.json,integration_atomic_artifact/,gpu_design.json,gpu_summary.json,resume_rejection.json,native_runtime_resume_rejection.json,gpu_resource_audit.json,adoption_audit.json,adoption_guard/}；runs/autonomous_native_extension_20261106/{design.json,summary.json,guard/command.log,resource_audit.json}。
