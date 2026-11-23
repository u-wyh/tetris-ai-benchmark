# 俄罗斯方块 AI 的长期生存与得分评测

[论文 PDF](paper.pdf) · [LaTeX 源文件](paper.tex) · [逐局数据](episodes.csv)

本文按独立种子组回顾性比较冻结的 BC Champion、PPO 30M、Hybrid B12 与传统 V2 Hold-only。论文记录评测上限、右截断、配对 Bootstrap 及模型复现限制。文稿暂未填写作者和单位，采用通用中文学术技术报告排版。

## 编译和重算

使用支持中文的 XeLaTeX。Ubuntu 可安装 `texlive-xetex`、`texlive-lang-chinese`、`fonts-noto-cjk` 和 `fonts-urw-base35`。已归档生成的表格与矢量图，可直接编译。

```sh
cd reports/paper
xelatex -interaction=nonstopmode -halt-on-error paper.tex
xelatex -interaction=nonstopmode -halt-on-error paper.tex
```

重算表格和生存曲线需要 Python 和 Matplotlib。从本目录运行 `python3 analysis.py`，然后重新编译。分析未启动训练或评测。

## 数据来源和字段

CSV 包含 224 行数据。每行对应一项策略在一个种子上的既有终局记录。`pieces` 为观测块数，`game_over` 与 `capped` 区分实际终止和触顶，`score` 为正式累计分。`clears_1` 至 `clears_4` 为对应的消行事件数。`source_sha256` 校验整个原始 JSON 文件。

| 协议 | 原始 JSON 路径模板 |
| --- | --- |
| A1 | `runs/ppo30m_independent_confirmation/evaluation/{policy}/seed_{seed}.json` |
| A2 | `runs/hybrid_v2/confirmation_100k/{policy}/seed_{seed}.json` |
| B（PPO、BC、B12） | `runs/ppo30m_failure_curriculum/post_curriculum_3m_fastbfs/{policy}/seed_{seed}.json` |
| B（V2） | `runs/post_curriculum_3m/traditional_v2_hold/seed_{seed}.json` |

A1 和 A2 的历史报告包含其他探索策略，本论文取两个协议共有的三种冻结基线。B 组增列长局传统基线。B 组 PPO 种子 `1400000`、`1400004`、`1400006` 的落点来自原 BFS 的已完成同协议对局，三个导入文件按内容哈希核对。Lookahead 仅引用独立协议的既有报告，未纳入本 CSV。

- [A1 评测报告](../experiments/ppo30m_independent_confirmation.md)
- [A2 评测报告](../experiments/hybrid_v2_comparison.md)
- [B 评测代码](../../training/evaluation/post_curriculum_3m.py)与[成绩汇总](../experiments/model_leaderboard.md)
- [模型权重哈希与本机路径](../releases/v1.0.0.md)

| 模型或策略身份 | SHA-256 |
| --- | --- |
| PPO 30M | `eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34` |
| BC Champion | `5f5613cb5eb4e80a02406b70902ad3d5c571eb8fe38bec4b40f73b34289ca663` |
| Hybrid B12 | 使用上述 PPO 与 BC 权重，B 模式，触发高度 12。 |
| B 组 V2 Hold-only Python 源码身份 | `d1def3c5c4948853f468a2254765e6fb3aae1adc1e5f9e692a43514a9ca295a2` |

权重与完整原始对局记录保存在本地 `runs/`，重算本文图表仅需归档的 CSV。评测逻辑以实验报告和固定提交为准。
