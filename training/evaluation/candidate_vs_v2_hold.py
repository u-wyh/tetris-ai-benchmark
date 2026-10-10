"""Recoverable 100-seed formal comparison of frozen Candidate and V2 Hold-only."""

import argparse
import fcntl
import json
import statistics
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from training.evaluation.champion_pipeline import bootstrap, percentile
from training.evaluation.traditional_v2 import code_digest, run as run_v2
from training.long_run import SEEDS_FILE
from training.train_ppo import ROOT, atomic_json, now
from training.train_vector_transaction import sha256
from training.traditional.v2 import V2Config, VERSION

OUT = ROOT / "runs/ppo_candidate_vs_v2_hold_final100"
TRADITIONAL = OUT / "traditional"
CHAMPION = ROOT / "runs/ppo_candidate_champion"
REFERENCE = ROOT / "reports/traditional_longlife/validation10_100000_20261008/hold/config.json"
REPORT = ROOT / "reports/experiments/ppo_candidate_vs_v2_hold_final100.md"


def preflight():
    seeds = json.loads(SEEDS_FILE.read_text())["final_test"]
    if seeds != list(range(200000, 200100)):
        raise RuntimeError("Frozen final-test seeds changed")
    champion = json.loads((CHAMPION / "champion.json").read_text())
    final = json.loads((CHAMPION / "final_100.json").read_text())
    source = json.loads((CHAMPION / "selection_decision.json").read_text())
    if (champion["selected_step"] != source["selected_step"]
            or final["checkpoint"]["model_sha256"] != champion["model_sha256"]
            or final["seeds"] != seeds or final["max_pieces"] != 50000
            or len(final["per_seed"]) != 100
            or [row["seed"] for row in final["per_seed"]] != seeds
            or sha256(CHAMPION / "model.zip") != champion["model_sha256"]):
        raise RuntimeError("Frozen Champion or existing final test failed integrity checks")
    historical = json.loads(REFERENCE.read_text())
    if (historical["strategy"] != VERSION or historical["policy"]["mode"] != "hold"
            or historical["code_sha256"] != code_digest()):
        raise RuntimeError("V2 Hold-only implementation differs from the validated reference")
    policy = V2Config(**historical["policy"])
    if asdict(policy) != historical["policy"]:
        raise RuntimeError("V2 Hold-only parameter mismatch")
    protocol = {"seeds": seeds, "max_pieces": 50000,
                "candidate_champion_sha256": champion["model_sha256"],
                "candidate_committed_steps": champion["selected_step"],
                "candidate_final_result_sha256": sha256(CHAMPION / "final_100.json"),
                "traditional_version": VERSION, "traditional_mode": "hold",
                "traditional_policy": historical["policy"],
                "traditional_code_sha256": code_digest(),
                "tetris_core": "training.tetris_core.TetrisCore",
                "deterministic": True, "same_seed_cap_score_rules": True,
                "candidate_final_reused": True}
    return final, policy, protocol


def freeze_protocol(protocol):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("Saved paired-test protocol differs")
    else:
        atomic_json(path, protocol)


def summarize(rows):
    pieces = [row["pieces_survived"] for row in rows]
    return {"mean_pieces": statistics.mean(pieces),
            "median_pieces": statistics.median(pieces),
            "p10_pieces": percentile(pieces, .1),
            "p90_pieces": percentile(pieces, .9),
            "capped_count": sum(row["survived_cap"] for row in rows),
            "game_over_count": sum(row["game_over"] for row in rows),
            "mean_lines": statistics.mean(row["lines"] for row in rows),
            "mean_score": statistics.mean(row["score"] for row in rows),
            "mean_capped_survival_bootstrap_95ci": bootstrap(pieces)}


def comparison(candidate, traditional, protocol):
    candidate_rows = candidate["per_seed"]
    traditional_rows = [json.loads((TRADITIONAL / f"seed_{seed}.json").read_text())
                        for seed in protocol["seeds"]]
    if ([row["seed"] for row in traditional_rows] != protocol["seeds"]
            or any(row["game_over"] == row["truncated"]
                   or not 1 <= row["pieces_survived"] <= 50000
                   or row["truncated"] and row["pieces_survived"] != 50000
                   for row in traditional_rows)):
        raise RuntimeError("Incomplete or invalid traditional seed results")
    standardized = [dict(row, survived_cap=row["truncated"])
                    for row in traditional_rows]
    candidate_summary = summarize(candidate_rows)
    traditional_summary = summarize(standardized)
    survival = [a["pieces_survived"] - b["pieces_survived"]
                for a, b in zip(candidate_rows, traditional_rows)]
    scores = [a["score"] - b["score"] for a, b in zip(candidate_rows, traditional_rows)]
    paired = {"mean_capped_survival_difference": statistics.mean(survival),
              "survival_difference_bootstrap_95ci": bootstrap(survival),
              "candidate_survival_wins": sum(value > 0 for value in survival),
              "survival_ties": sum(value == 0 for value in survival),
              "traditional_survival_wins": sum(value < 0 for value in survival),
              "mean_score_difference": statistics.mean(scores),
              "score_difference_bootstrap_95ci": bootstrap(scores),
              "candidate_score_wins": sum(value > 0 for value in scores),
              "score_ties": sum(value == 0 for value in scores),
              "traditional_score_wins": sum(value < 0 for value in scores),
              "both_capped": sum(a["survived_cap"] and b["truncated"]
                                 for a, b in zip(candidate_rows, traditional_rows)),
              "per_seed": [{"seed": a["seed"], "candidate_pieces": a["pieces_survived"],
                            "traditional_pieces": b["pieces_survived"],
                            "candidate_score": a["score"], "traditional_score": b["score"],
                            "candidate_capped": a["survived_cap"],
                            "traditional_capped": b["truncated"],
                            "survival_difference": a["pieces_survived"] - b["pieces_survived"]}
                           for a, b in zip(candidate_rows, traditional_rows)]}
    total_seconds = sum(row["wall_seconds"] for row in traditional_rows)
    total_pieces = sum(row["pieces_survived"] for row in traditional_rows)
    traditional_config = json.loads((TRADITIONAL / "config.json").read_text())
    completed_at = now()
    elapsed_hours = (datetime.fromisoformat(completed_at)
                     - datetime.fromisoformat(traditional_config["created_at"])).total_seconds() / 3600
    result = {"protocol": protocol, "candidate": candidate_summary,
              "traditional": traditional_summary, "paired": paired,
              "traditional_runtime": {"sum_game_wall_seconds": total_seconds,
                                      "sum_game_wall_hours": total_seconds / 3600,
                                      "queue_elapsed_hours": elapsed_hours,
                                      "pieces_per_second": total_pieces / total_seconds,
                                      "total_pieces": total_pieces},
              "completed_at": completed_at}
    atomic_json(OUT / "paired_result.json", result)
    return result


def report(result):
    a, b, p = result["candidate"], result["traditional"], result["paired"]
    lines = ["# Candidate PPO Champion 与 V2 Hold-only 正式公平对照", "",
             "两者使用相同 seeds 200000–200099、同一 Python TetrisCore 的 7-Bag 与正式计分、同一合法落点规则，"
             "每局最多 50,000 块。Candidate 使用既有已冻结 7M Champion 的 100 局正式结果，未重复推理；"
             "V2 Hold-only 使用历史验证过的固定权重和配置，逐 seed 独立运行。达到上限是右截断。", "",
             "| 指标 | Candidate Champion | V2 Hold-only |", "|---|---:|---:|",
             f"| 平均存活 | {a['mean_pieces']:.1f} | {b['mean_pieces']:.1f} |",
             f"| 中位存活 | {a['median_pieces']:.1f} | {b['median_pieces']:.1f} |",
             f"| P10 / P90 存活 | {a['p10_pieces']} / {a['p90_pieces']} | {b['p10_pieces']} / {b['p90_pieces']} |",
             f"| 达到 50,000 块上限 | {a['capped_count']}/100 | {b['capped_count']}/100 |",
             f"| Game Over | {a['game_over_count']}/100 | {b['game_over_count']}/100 |",
             f"| 平均消行 | {a['mean_lines']:.1f} | {b['mean_lines']:.1f} |",
             f"| 平均正式得分 | {a['mean_score']:.1f} | {b['mean_score']:.1f} |", "",
             f"配对截断存活差（Candidate − V2 Hold-only）平均 {p['mean_capped_survival_difference']:.1f} 块；"
             f"固定 seed 的 10,000 次整局 bootstrap 95% 区间为 {p['survival_difference_bootstrap_95ci']}。"
             f"逐 seed 存活胜/平/负：{p['candidate_survival_wins']}/{p['survival_ties']}/{p['traditional_survival_wins']}；"
             f"双方都截断 {p['both_capped']} 局。截断后的真实寿命无法由这些差值确定。", "",
             f"配对得分差平均 {p['mean_score_difference']:.1f} 分，95% 区间 {p['score_difference_bootstrap_95ci']}；"
             f"得分胜/平/负：{p['candidate_score_wins']}/{p['score_ties']}/{p['traditional_score_wins']}。", "",
             f"V2 Hold-only 100 局合计计算墙钟 {result['traditional_runtime']['sum_game_wall_hours']:.2f} 小时；"
             f"从启动到完成的队列历时 {result['traditional_runtime']['queue_elapsed_hours']:.2f} 小时。"
             f"实际端到端平均速度 {result['traditional_runtime']['pieces_per_second']:.2f} 块/秒。"
             "逐 seed 记录包括完整耗时；中断停机时间不计入合计计算墙钟。", "",
             "| Seed | Candidate 存活 | V2 存活 | Candidate 分数 | V2 分数 | 结局 |",
             "|---:|---:|---:|---:|---:|---|"]
    for row in p["per_seed"]:
        lines.append(f"| {row['seed']} | {row['candidate_pieces']} | {row['traditional_pieces']} | "
                     f"{row['candidate_score']} | {row['traditional_score']} | "
                     f"{'双截断' if row['candidate_capped'] and row['traditional_capped'] else 'Candidate 截断' if row['candidate_capped'] else 'V2 截断' if row['traditional_capped'] else '双方 Game Over'} |")
    lines += ["", "正式测试 seeds 曾用于 Raw PPO 报告，不是完全未使用的新测试集；"
              "本轮没有根据结果改变 Candidate Champion 或传统算法参数。", ""]
    REPORT.write_text("\n".join(lines))
    changelog = ROOT / "CHANGELOG.md"
    text = changelog.read_text()
    entry = (f"- Candidate Champion 与 V2 Hold-only 使用相同 100 seeds、每局 50000 块完成正式配对："
             f"平均截断存活 {a['mean_pieces']:.1f} vs {b['mean_pieces']:.1f} 块，"
             f"上限截断 {a['capped_count']}/100 vs {b['capped_count']}/100。"
             "详见 `reports/experiments/ppo_candidate_vs_v2_hold_final100.md`。\n")
    if entry not in text:
        marker = "## 2026-10-10\n\n"
        if marker not in text:
            raise RuntimeError("CHANGELOG date marker missing")
        changelog.write_text(text.replace(marker, marker + entry, 1))


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            candidate, policy, protocol = preflight()
            freeze_protocol(protocol)
            atomic_json(OUT / "status.json", {"stage": "traditional_eval", "state": "running",
                                              "updated_at": now()})
            run_v2(TRADITIONAL, protocol["seeds"], 50000, policy,
                   resume=(TRADITIONAL / "config.json").exists())
            atomic_json(OUT / "status.json", {"stage": "paired_report", "state": "running",
                                              "updated_at": now()})
            result = comparison(candidate, TRADITIONAL, protocol)
            report(result)
            atomic_json(OUT / "status.json", {"stage": "completed", "state": "completed",
                                              "updated_at": now()})
        except Exception as error:
            atomic_json(OUT / "status.json", {"stage": "failed", "state": "failed",
                                              "error": repr(error), "traceback": traceback.format_exc(),
                                              "updated_at": now()})
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status_path = OUT / "status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        count = len(list(TRADITIONAL.glob("seed_*.json"))) if TRADITIONAL.exists() else 0
        live = TRADITIONAL / "status.json"
        print(json.dumps({"queue": state, "completed_games": count,
                          "current_game": json.loads(live.read_text()) if live.exists() else None},
                         ensure_ascii=False, indent=2))
    else:
        execute()


if __name__ == "__main__":
    main()
