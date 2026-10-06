# 修改日志

## 2026-10-06

- Gameplay RNG 改用可复现的 Mulberry32；7-Bag 支持固定 seed，并保存、恢复当前 RNG 状态。
- 视觉随机性与游戏随机性分离；传统 AI 使用独立随机源，不影响出块顺序。
- 明确 Benchmark v1.0 Top Out：保留出生碰撞 Block Out，加入整块在棋盘上方锁定时的 Lock Out。
- Partial Lock 暂时允许继续，仍正常合并可见部分并消行。
- 将自定义旋转规则替换为标准 SRS，明确维护 `0 → R → 2 → L` 旋转状态。
- JLSTZ 与 I 方块分别使用对应的 Wall Kick 规则；O 方块原地旋转。
- 保留 Hold、Next、计分等已有规则，并兼容旧版游戏存档。
- 完成贴墙、地面、碰撞、旋转失败和现有玩法的可重复测试。
