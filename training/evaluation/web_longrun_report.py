"""Summarize sealed browser HTTP/Core parity chunks without rerunning games."""

import json
import math
import statistics
from pathlib import Path

from training.evaluation.web_longrun_verify import committed, sealed
from training.train_ppo import ROOT

RUN = ROOT / "runs/web_longrun_validation"
REPORT = ROOT / "reports/experiments/web_longrun_validation.md"
TASKS = (("ppo_10k", "PPO 30M"), ("bc_10k", "BC Champion"),
         ("b12_10k", "Hybrid B12"), ("bc_100k", "BC 100K"))


def percentile(values, rate):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(rate * len(ordered)) - 1]


def describe(run):
    result = json.loads((run / "result.json").read_text())
    count = committed(run)
    if count != result["chunks"]:
        raise RuntimeError(f"Committed chunk count changed: {run}")
    summaries = [json.loads(sealed(run / "summaries" / f"{i:06d}.json"))
                 for i in range(1, count + 1)]
    if sum(s["steps"] for s in summaries) != result["pieces"]:
        raise RuntimeError(f"Piece count changed: {run}")
    metrics = {}
    for field in ("server_ms", "http_ms", "bfs_ms", "path_ms"):
        values = [value for s in summaries for value in s["metrics"][field]]
        metrics[field] = {"mean": statistics.mean(values), "p95": percentile(values, .95),
                          "p99": percentile(values, .99)}
    final = summaries[-1]
    return {**result, "score": final["score"], "lines": final["lines"],
            "metrics": metrics,
            "failed_requests": sum(s["metrics"]["failed_requests"] for s in summaries),
            "source_counts": {source: sum(s["metrics"]["source_counts"][source]
                                           for s in summaries) for source in ("ppo", "bc")},
            "node_rss_peak_mb": max(s["metrics"]["node_rss_bytes"] for s in summaries) / 2**20,
            "server_rss_peak_mb": max((s["metrics"]["server_usage"] or {}).get("rss_kb", 0)
                                      for s in summaries) / 1024,
            "node_cpu_ms": sum(s["metrics"]["node_cpu_ms"] for s in summaries),
            "server_cpu_sample_max": max((s["metrics"]["server_usage"] or {}).get("cpu_percent", 0)
                                         for s in summaries),
            "seconds_per_1000": result["wall_ms"] / result["pieces"]}


def main():
    rows = {name: describe(RUN / name) for name, _ in TASKS}
    (RUN / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    lines = ["# 网页神经 AI 长局稳定性验证", "",
             "Seed `12345` 为已使用过的历史功能回归种子。三种 AI 各最多 10000 块；通过后 BC 再最多 100000 块。每一步由网页代码向 8081 发出真实 HTTP 请求，网页按正式 BFS 路径落子，独立 Python Core 从同一 seed 执行返回的 Action ID，并逐步核对完整棋盘、Current、Next、Hold、得分、等级、消行、结束状态及合法动作列表。每 500 块原子提交浏览器/Core 恢复快照与 SHA256 封印，失败即停止。未使用 `1452000..1452031` 或 `1441000..1441031`。", "",
             "| 模式 | 实际放置 | 结束 | 正式得分 | 推理平均/P95/P99 ms | HTTP 平均/P95/P99 ms | 网页 BFS 平均/P95/P99 ms | 全链路每千块秒 | 失败请求 |",
             "|---|---:|---|---:|---|---|---|---:|---:|"]
    for name, title in TASKS:
        r = rows[name]
        fmt = lambda field: "/".join(f"{r['metrics'][field][key]:.2f}" for key in ("mean", "p95", "p99"))
        outcome = "真正 Game Over" if r["game_over"] else "上限右截断"
        lines.append(f"| {title} | {r['pieces']} | {outcome} | {r['score']:,} | {fmt('server_ms')} | "
                     f"{fmt('http_ms')} | {fmt('bfs_ms')} | {r['seconds_per_1000']:.1f} | "
                     f"{r['failed_requests']} |")
    lines += ["", "上表每千块秒使用整轮实际墙钟耗时，包含 HTTP、网页路径执行、Python Core 逐步验证及原子 checkpoint。无渲染测试通过大步进驱动路径，**不等于用户观看动画时的播放速度**；真实页面仍按演示速度滑块逐步播放。到达 10000/100000 上限的局属于右截断，不是该位置死亡。", "",
              "## 资源与恢复", ""]
    lines.append(f"B12 的策略来源：PPO {rows['b12_10k']['source_counts']['ppo']} 次，BC {rows['b12_10k']['source_counts']['bc']} 次。")
    for name, title in TASKS:
        r = rows[name]
        lines.append(f"- {title}：Node 峰值 RSS {r['node_rss_peak_mb']:.1f} MiB，8081 服务采样峰值 RSS {r['server_rss_peak_mb']:.1f} MiB；Node CPU 累计 {r['node_cpu_ms']/1000:.1f} 秒，服务采样 CPU 最高 {r['server_cpu_sample_max']:.1f}%。")
    lines += ["", "短窗专门测试另验证：神经 AI 刷新后从网页保存恢复；暂停、继续、切换模式、重开游戏后旧请求不会作用于新棋盘；请求错误停止自动重试并由按钮恢复；服务重启后从 650 块 checkpoint 续跑至 700 块；PPO 与 BC 两个浏览器会话同时各运行 200 块，均逐步通过 Core 校验。静态 8080 服务在测试期间保持可用。", "",
              "## 结论", "",
              "若全部封印及回归通过，则本轮在既定历史 seed 和上限内未观察到状态漂移、非法动作或服务故障，具备持续演示条件。结论不扩展为不同 seed 上的模型泛化能力，也不代表触顶后的无限生存。完整原子结果与异常日志在 `runs/web_longrun_validation/`。", ""]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
