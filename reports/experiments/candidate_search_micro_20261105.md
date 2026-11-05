# 既有位图落点搜索复用验证

2026-11-05；仅复用已有training/evaluation/fast_legal.py，不改正式Core或训练/推理后端。将归档分支5e7483b的边界/可达轨迹用例移入独立测试，与现有环境回归共177项通过。覆盖完整有序动作、occupiedCells及path，特征和mask逐项一致。

原30M及低LR候选各8个提交worker，加清行/顶出边界共19状态；30次AB/BA固定CPU100%/2GiB/原温控，比较预先给定公开观察→合法落点搜索→相同位图特征。中位吞吐1.16418667倍（约+16.42%），每对均更快，范围1.09332–1.23191；达到事前median≥1.10且每对>1门槛。只计静态构造，不含IPC/模型/PPO，不能当作整体训练提速。

源码与输入SHA前后不变，原推理20身份仍匹配，未重标缓存、迁移旧断点或重复基线。下一步在独立轻薄本副本验证显式可选的训练搜索后端，再做真实CUDA8两组AB/BA及4096/8192跨进程恢复精确对照；未通过前不启用。

证据：runs/autonomous_candidate_search_20261105/design.json、summary.json、construction_microbenchmark.json、guard/。
