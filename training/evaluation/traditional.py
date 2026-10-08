"""Resumable CPU V1 benchmark. Each completed seed is atomically persisted."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone

from training.tetris_core import TetrisCore
from training.traditional.v1 import V1Adapter, VERSION, WEIGHTS

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def code_digest():
    paths = sorted((ROOT / "training/tetris_core").glob("*.py")) + [
        ROOT / "training/traditional/v1.py", Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def play(seed, max_pieces, progress=None):
    core, policy = TetrisCore(seed), V1Adapter(seed)
    times, actions = [], []
    began = time.perf_counter()
    while core.phase != "over" and len(actions) < max_pieces:
        observation = core.get_public_observation()
        started = time.perf_counter_ns()
        placements = core.get_legal_placements()
        mask = core.get_action_mask()
        action = policy.choose(observation, placements)
        if not mask[action] or action >= len(mask) // 2:
            raise AssertionError("Policy chose an illegal or Hold action")
        times.append((time.perf_counter_ns() - started) / 1e6)
        core.step(action)
        actions.append(action)
        if progress and len(actions) % 100 == 0:
            progress(len(actions))
    elapsed = time.perf_counter() - began
    return dict(seed=seed, pieces_survived=len(actions), lines=core.lines, score=core.score,
                game_over=core.phase == "over", truncated=core.phase != "over" and len(actions) == max_pieces,
                decision_ms=times, actions=actions, wall_seconds=elapsed,
                completed_at=now())


def aggregate(rows):
    result = dict(num_seeds=len(rows))
    for source, label in (("pieces_survived", "pieces"), ("lines", "lines"), ("score", "score")):
        values = [row[source] for row in rows]
        result.update({f"mean_{label}": statistics.mean(values),
                       f"median_{label}": statistics.median(values),
                       f"p90_{label}": percentile(values, .9)})
    times = [value for row in rows for value in row["decision_ms"]]
    result.update(game_over_rate=statistics.mean(row["game_over"] for row in rows),
                  cap_rate=statistics.mean(row["truncated"] for row in rows),
                  mean_decision_ms=statistics.mean(times), p95_decision_ms=percentile(times, .95),
                  decisions_per_second=1000 / statistics.mean(times),
                  end_to_end_pieces_per_second=len(times) / sum(row["wall_seconds"] for row in rows),
                  lines_per_piece=sum(row["lines"] for row in rows) / len(times))
    result["evaluation_saturated"] = result["cap_rate"] >= .5
    result["metric_censored_by_cap"] = result["cap_rate"] > 0
    return result


def report(config, rows, metrics):
    text = ["# Traditional V1 Adapter 实测报告", "", "这是合法动作适配版实测，不能视为原始网页 V1 的实测。", "",
            f"配置：{len(rows)} seeds，最多 {config['max_pieces']} 块；CPU；代码版本 {VERSION}。",
            f"Git 基线：`{config['git_commit']}`；实际代码 SHA-256：`{config['code_sha256']}`。", "",
            "保留原 V1 全部七项权重、下一块 0.58 系数及独立 Mulberry32 的 0.002 扰动。",
            "当前与下一块候选均用正式 BFS，禁止 Hold；原版几何枚举不验证路径。候选集合/顺序变化使 RNG 消耗及最终选择可能不同。",
            "策略 RNG 用 CLI seed 初始化，与游戏 RNG 隔离；策略只接收公开 Observation，不读取 Bag、RNG 或 Next 4+。",
            "原网页 Down 使用 softDrop(true) 每格 +1；Core BFS Down 无加分，仅剩余 HardDrop 每格 +2，消行按正式等级计分。",
            "保留原 V1 对任何负 y 格子的模拟拒绝；全被拒绝时选最小合法非 Hold 动作避免停滞，原网页可能停滞。",
            "前瞻无候选时仍只用当前评分，未额外添加死亡惩罚。Core 的 partial lock/lock out/block out 规则不变。", "",
            "决策时间包含当前合法动作及 mask 构造、全部一块前瞻和策略选择；不含 Core.step 与记录 I/O。每秒决策数为总决策次数/总决策时间。",
            "P90/P95 使用 nearest-rank；逐 seed 文件包含完整动作及逐决策耗时。存活块计数沿用 Env 的已执行 placement 次数（含最后致死块）。", "",
            "| 指标 | 实测 |", "|---|---:|"]
    text += [f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |" for key, value in metrics.items()]
    text += ["", "| seed | 块数 | 消行 | 正式分数 | Game Over | 截断 |", "|---|---:|---:|---:|---|---|"]
    text += [f"| {r['seed']} | {r['pieces_survived']} | {r['lines']} | {r['score']} | {r['game_over']} | {r['truncated']} |" for r in rows]
    text += ["", "达到上限是截断，不能将上限解释为真实寿命。" + ("至少半数截断，评测已饱和；应使用独立 validation 种子提高上限后再判断寿命。" if metrics["evaluation_saturated"] else "本次没有达到半数截断的饱和阈值。"),
             "本阶段未运行原网页 V1 或 V2，不能推断它们的成绩或算法提升。16 seeds 也不足以描述所有出块分布。", ""]
    return "\n".join(text)


def run(output, seeds, max_pieces, resume=False):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be unique unsigned 32-bit integers")
    if max_pieces < 1:
        raise ValueError("max_pieces must be positive")
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "config.json").is_file():
        raise ValueError("Resume requires an existing config")
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = dict(seeds=seeds, max_pieces=max_pieces, strategy=VERSION, weights=WEIGHTS,
                        lookahead=.58, jitter=.002, hold=False, code_sha256=code_digest())
        if resume:
            config = json.loads((output / "config.json").read_text())
            if any(config.get(k) != v for k, v in expected.items()):
                raise ValueError("Resume configuration/code mismatch")
        else:
            config = dict(expected, created_at=now(), python=platform.python_version(),
                          platform=platform.platform(), git_commit=subprocess.check_output(
                              ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                          git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                          timing="placement generation + mask + policy; excludes execution/I/O")
            atomic(output / "config.json", config)
        rows = []
        try:
            for seed in seeds:
                destination = output / f"seed_{seed}.json"
                if destination.exists():
                    row = json.loads(destination.read_text())
                    if (row["seed"] != seed or len(row["actions"]) != row["pieces_survived"]
                            or len(row["decision_ms"]) != row["pieces_survived"]
                            or row["game_over"] == row["truncated"]
                            or not 0 < row["pieces_survived"] <= max_pieces
                            or (row["truncated"] and row["pieces_survived"] != max_pieces)):
                        raise ValueError("Invalid completed seed record")
                else:
                    def progress(pieces):
                        atomic(output / "status.json", dict(state="running", seed=seed, pieces=pieces,
                               completed_seeds=len(rows), total_seeds=len(seeds), updated_at=now()))
                    progress(0)
                    row = play(seed, max_pieces, progress)
                    atomic(destination, row)
                rows.append(row)
            metrics = aggregate(rows)
            atomic(output / "aggregate.json", metrics)
            report_path = output / "report.md"
            if not report_path.exists():
                report_path.write_text(report(config, rows, metrics))
            atomic(output / "status.json", dict(state="complete", completed_seeds=len(rows),
                   total_seeds=len(seeds), updated_at=now()))
            return metrics
        except BaseException as error:
            # Unique filenames retain every failed attempt. Partial episodes never
            # enter seed JSON or aggregate; resume restarts that seed from scratch.
            atomic(output / f"error_{time.time_ns()}.json", dict(type=type(error).__name__,
                   message=str(error), traceback=traceback.format_exc(), timestamp=now()))
            atomic(output / "status.json", dict(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   completed_seeds=len(rows), total_seeds=len(seeds), updated_at=now()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-pieces", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print((args.output / "status.json").read_text())
        return
    seeds = args.seeds
    if seeds is None:
        seeds = json.loads((ROOT / "training/evaluation/seeds.json").read_text())["validation"][:16]
    print(json.dumps(run(args.output, seeds, args.max_pieces, args.resume), indent=2))


if __name__ == "__main__":
    main()
