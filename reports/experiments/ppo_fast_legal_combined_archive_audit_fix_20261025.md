# 合并落点基准归档审计来源修复

2026-10-25 08:19 +0800。原审计把归档协议中的新版落点和分支源码 SHA256 同当前工作树比较；后续合法源码变更会使旧归档无法复核。现在分别从提交 `d40bc756e0fb` 和 `4ff338010bfd` 读取当时的源码，再与冻结协议核对；旧版落点仍来自固定提交 `5cf0c74`。这些提交的对应文件字节与原协议哈希吻合，归档数据未改写。

运行 `python3 -m training.evaluation.audit_fast_legal_combined` 通过：3 个独立开发 episode、6 对计时，汇总倍率 1.079276×。`git diff --check` 通过；8080 首页和 8081 健康接口均 HTTP 200。本轮无分支重算、新种子或训练步。此审计仍不能从归档独立重建四次完整分支输出，原有证据限制不变。

证据：`runs/ppo_fast_legal_combined_20261024/{protocol,summary}.json`；审计入口：`training/evaluation/audit_fast_legal_combined.py`。
