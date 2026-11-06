# 行位图候选落点搜索（隔离验收）

## 算法与范围

在lightbook隔离副本`tetris-search-compact-validation`，e3c4da5将每个旋转/纵坐标的允许横坐标和访问集合压成位图；以board行位移/或运算预计算碰撞，减少逐位置collision/LRU查询。保留BFS Left/Right/RotateCW/Down、kick首个无碰撞选择、action排序/去重与路径；异常初始坐标转原搜索。显式bitboard_compact，formal默认与旧bitboard保留，推理20来源不变。

旧9144a36的27文件已先归档`tetris-runtime-9144a36`，全部SHA与旧GPU metadata相同。此前compact仅+2.1%的证据和失败保留，没有复跑原计时或重标旧模型。新实验目录autonomous_row_masks_20261106及新的源码SHA。

## 静态与安全门

233项通过，含43680组独立逐格碰撞与3136负y/墙位姿，及placement/path/feature/mask/奖励/轨迹/快照差分。27状态×30 ABBA静态完整构造，预声明门槛中位≥1.10且每对更快；实际中位1.376835685、最小1.342313352，39输入完成时未改。CPU100%/2GiB/原温控。cProfile主进程此前显示采样等待主要在step_wait，单独mask RPC仅约.227秒，因而未重写mask通信或packed rollout。

2ace0c1给串行验收增加显式上游性能门，7项保护回归通过：status须passed且eligibility为bool；false记录skipped_ineligible_upstream且不创建guard/子进程，缺失/模糊/失败拒绝。旧微基准的controller支持原文归档support_at_micro，随后才修改。

## 实际CUDA8 AB/BA结果

预声明seed2100000/2100100，每臂8192、4096跨进程恢复，49输入冻结，训练全程250%、6GiB/原温控，每命令300秒/总1800秒。基线旧bitboard、候选row masks，都用同bitboard features/同27来源/同初始化与配方。只是性能/确定性对照；不是新的棋力训练假设。

| seed | 旧bitboard训练秒 | row masks训练秒 | 吞吐比 |
| --- | ---: | ---: | ---: |
| 2100000 | 14.431 | 13.379 | 1.07863069 |
| 2100100 | 14.815 | 13.342 | 1.11040324 |

10:28:19 gpu_summary passed/eligible=true，几何1.094401668（约+9.44%），超过预声明≥5%且两对更快。全部policy/Adam/PPO更新/全部trainer RNG/8worker/非计时指标精确；只排除wall_seconds与gpu_peak_allocated_mib。错backend preset恢复被拒绝，config/metadata/transaction_state字节不变。49输入完成时未改。

此为两组本地固定配方吞吐，不代表普遍加速、棋力改善或可与历史其他实验简单叠加。

## 当前

benchmark10:28:22 completed，自动条件接力10:28:27启动完整上下文完成→扩展→故意中断→恢复验收（240秒命令/1800秒总队列）。通过前不合入main；main26/推理20仍保持。后续查真实summary/guard，不把服务启动当通过。

证据：runs/autonomous_row_masks_20261106/、runs/autonomous_row_masks_extension_20261106/。

## 完成与采用

10:29:52完整上下文合成CUDA8夹具通过：1 passed/4 deselected，82.41秒；source正式搜索→新row masks完整上下文降LR→已完成4096扩展至12288→故意中断→恢复，与连续控制policy/Adam/PPO/全部RNG/8worker精确同。父服务MainPID0/Resultsuccess。此为合成事务/恢复验收，不将其称为原30M历史代码重演。

采用前归档主副本原26文件至`/home/wyh/projects/tetris-runtime-8c1b411`，全部SHA与原actor上下文metadata相同。随后合入bc18167/5ddd8ec/b41dc69；当前main27数值文件与实测隔离副本逐项SHA相同，推理20与原bitboard长基线身份完全相同。旧26/旧27运行代码均可在本机准确恢复，旧断点和缓存未重标，未重算长基线。

`configs/ppo30m_guarded_bitboard.json`现显式candidate_search_backend=bitboard_compact；保留其GPU8及PPO配方，资源仍用training_resources.json。该模板变更没有启动2M训练；新质量实验仍须自己的步数预算、冻结协议和筛选。formal默认与旧bitboard仍可显式选用。

GPU配对CPU峰75.25℃/GPU53℃/组内存3584.4MiB；扩展验收CPU71.25℃/GPU51℃/内存3900.5MiB。两套训练采样均全250%、未触发温控降载，保护未放宽。

行位图性能配置已采用；原PPO30M仍为冻结参考，未发布或替换质量模型，+9.44%是本地配对吞吐证据。见integration_audit.json、old_main_runtime_archive.json、gpu_resource_audit.json。
