# 修改日志

## 2026-10-06

- 简化 AI placement BFS：仅以 `(x,y,rotation)` 去重，移除搜索中的 50 ms 计时与 Lock Reset 维度，保留完整 SRS 路径可达性；人工 Lock Delay 不变。基准从 232.39 ms/mask 提升至 4.64 ms/mask。
- 新增 1840 固定动作空间、encode/decode、真实可达 placement 搜索及 Action Mask；支持 SRS、Hold、15 次 Lock Reset，并固定去重与标准路径。
- 固化 Next Preview = 3，新增只读公开观察接口；隐藏 Next 4+、Bag 和 RNG 内部状态，为统一 Agent 接口建立信息边界。
- 固化 Benchmark v1.0 计分：消行按 `100/300/500/800 × level`，Soft Drop 每格 +1，Hard Drop 每格 +2；暂不加入 T-Spin、Combo、Back-to-Back。
- Lock Delay 为 450 ms；每块最多 15 次计时重置，仅成功的接地移动或旋转消耗次数；Hard Drop 仍立即锁定。
- Gameplay RNG 改用可复现的 Mulberry32；7-Bag 支持固定 seed，并保存、恢复当前 RNG 状态。
- 视觉随机性与游戏随机性分离；传统 AI 使用独立随机源，不影响出块顺序。
- 明确 Benchmark v1.0 Top Out：保留出生碰撞 Block Out，加入整块在棋盘上方锁定时的 Lock Out。
- Partial Lock 暂时允许继续，仍正常合并可见部分并消行。
- 将自定义旋转规则替换为标准 SRS，明确维护 `0 → R → 2 → L` 旋转状态。
- JLSTZ 与 I 方块分别使用对应的 Wall Kick 规则；O 方块原地旋转。
- 保留 Hold、Next、计分等已有规则，并兼容旧版游戏存档。
- 完成贴墙、地面、碰撞、旋转失败和现有玩法的可重复测试。
