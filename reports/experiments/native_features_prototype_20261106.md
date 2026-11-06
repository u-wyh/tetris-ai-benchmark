# C整数候选特征原型：通过静态门，尚未接训练

隔离 `/home/wyh/projects/tetris-native-feature-validation`，分支 `codex/native-candidate-features`，提交 **d229934**。没有修改main训练代码或默认配置，没有启动该原型的GPU训练。

## 实现及身份

完整C整数内核计算16项特征、行清除与列重建。检查元数据及cell坐标后才索引或移位；正常4-cell放置、负y/不可见、已有满行均在C中处理。Python一次打包连续int64数组，通过ctypes调用；异常/变长cells等回退原参考函数，保留输出/异常语义。

只显式加载预先编译的.so，构造函数必须接收外部固定manifest SHA。manifest固定ABI、GCC13.3路径/版本/driver SHA、flags、平台、C源和wrapper SHA、二进制SHA。缺失/不匹配拒绝，不隐式编译或替换。编译参数为-std=c11 -O3 -fPIC -shared -Wall -Wextra -Werror；无安装包。

二进制SHA：bff6f8e64a1516ad577ceaee7c5d9b79ece4800147bf33a718ead6f9c2b9642f。

**当前仅原型加载身份，并非训练检查点身份已经集成。** 训练来源仍27；C和wrapper尚未进入训练来源清单。未来接入必须同时固定checkpoint/config/worker加载身份，不能只记C源而漏掉编译产物。

## 正确性与一次静态对照

129项受保护测试通过。包括1024组随机20-bit列及2的幂边界、128组逐格行压缩参考、多/非连续清行、已有满行/不可见、整数极值/异常类型/负action别名、200步真实轨迹/快照，以及manifest/ABI/flags/C源/wrapper/二进制篡改和缺失拒绝。正常清行不回退有专门断言。轨迹使用测试专用子类，未当成正式训练后端。

固定同3份快照24状态+3边界，共27状态、30组AB/BA，只计特征构造，包含Python打包、输出分配与ctypes调用，不包含一次manifest校验/动态加载。参考为当前bitboard特征，双方使用相同预计算落点。

| 指标 | 结果 |
| --- | ---: |
| 配对特征速度比中位数 | **6.935353** |
| 最小配对速度比 | 6.336642 |
| 参考每27状态CPU秒中位 | .012261304 |
| C候选每27状态CPU秒中位 | .0017665925 |
| 预声明静态门 | ≥1.8且每对>1，**通过** |

所有feature/mask逐项精确。build/unit/micro分别37/41/45冻结输入未改，推理20在main/隔离/准确归档43cc26f逐项同。eligible_for_training_backend_integration=true；**eligible_for_gpu_benchmark=false**，须先完成显式后端/检查点运行时身份集成。没有将静态6.94倍称为整体训练加速，亦无模型质量改善结论。

## 保护与后续

| 阶段 | CPU峰℃ | GPU峰℃ | 内存峰MiB | 配额 |
| --- | ---: | ---: | ---: | ---: |
| 编译 | 39 | 34 | 12.38 | 100% |
| 129项测试 | 62.5 | 37 | 300.34 | 100% |
| 静态 | 43.125 | 39 | 14.46 | 100% |

全程2GiB、原温控与GPU遥测，均completed/无降载，无环境训练或模型更新。短时遥测采样峰不代表连续物理温度上界。复用准确main27归档，无重复创建；旧NumPy/fused/delta负结果保持停止。

下一在该隔离副本集成显式训练后端及checkpoint-pinned manifest/二进制身份；普通旧后端不要求.so。先测试默认兼容性、错误feature/编译身份恢复拒绝且提交不变，再一次新CUDA8/250% AB/BA端到端门（每对更快/几何≥1.05/状态全精确/无降载），合格才完整上下文扩展/故障恢复。当前这些尚未开始，main不采用、训练模板不改。

证据：runs/autonomous_native_features_20261106/{runtime_archive.json,artifact/manifest.json,build/design.json,unit/guard/command.log,micro/design.json,micro/microbenchmark.json,micro/qualification.json,resource_audit.json,inference_audit.json}。本轮小时消息实际执行见runs/autonomous_development_20261105/hourly_vectorized_execution_20261106.json。
