"""Write the 10M Candidate versus Raw validation comparison after training."""

import argparse
import json
from pathlib import Path

from training.evaluation.evaluate import evaluate_model
from training.train_ppo import ROOT, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--raw-run", type=Path,
                        default=ROOT / "runs/ppo_raw_10m_seed42")
    args = parser.parse_args()
    candidate_state = json.loads((args.candidate_run / "transaction_state.json").read_text())
    raw_state = json.loads((args.raw_run / "transaction_state.json").read_text())
    if (candidate_state["status"] != "completed" or raw_state["status"] != "completed"
            or candidate_state["committed_steps"] != raw_state["committed_steps"]):
        raise RuntimeError("Both runs must be complete at the same committed step")
    step = candidate_state["committed_steps"]
    seeds = json.loads((ROOT / "training/evaluation/seeds.json").read_text())["validation"][:32]
    outputs = args.candidate_run / "evaluations" / "paired_10m"
    outputs.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, run in (("Raw MLP", args.raw_run), ("Candidate-Scoring", args.candidate_run)):
        path = outputs / ("raw.json" if label == "Raw MLP" else "candidate.json")
        if path.exists():
            result = json.loads(path.read_text())
            if (result["seeds"] != seeds or result["max_pieces"] != 5000
                    or result["committed_steps"] != step):
                raise RuntimeError("Existing 10M validation result has a different protocol")
        else:
            model = run / "milestones" / f"step_{step:09d}" / "model.zip"
            result = evaluate_model(model, seeds, 5000, "paired_validation_10m", step,
                                    diagnostics=True)
            atomic_json(path, result)
        results[label] = result
    left, right = (results[name] for name in ("Raw MLP", "Candidate-Scoring"))
    lines = ["# Candidate-Scoring PPO 与 Raw PPO 10M 对照", "",
             f"双方均训练至 {step:,} committed steps。使用同一批 32 个 validation seeds（100000–100031）、每局上限 5000 块、确定性推理及 Action Mask；未使用 final-test seeds。", "",
             "| 指标 | Raw MLP | Candidate-Scoring |", "|---|---:|---:|"]
    for label, key in (("平均存活", "mean_pieces"), ("存活中位数", "median_pieces"),
                       ("平均消行", "mean_lines"), ("平均分数", "mean_score"),
                       ("上限截断率", "survival_cap_rate"),
                       ("每 100 块新增洞", "new_holes_per_100_pieces")):
        lines.append(f"| {label} | {left['aggregate'][key]:.2f} | {right['aggregate'][key]:.2f} |")
    lines += ["", "逐 seed 结果：", "", "| Seed | Raw 存活 | Candidate 存活 | Candidate 结局 |",
              "|---:|---:|---:|---|"]
    for raw, candidate in zip(left["per_seed"], right["per_seed"]):
        if raw["seed"] != candidate["seed"]:
            raise RuntimeError("Paired validation seed mismatch")
        lines.append(f"| {raw['seed']} | {raw['pieces_survived']} | {candidate['pieces_survived']} | "
                     f"{'上限截断' if candidate['survived_cap'] else 'Game Over'} |")
    lines += ["", "20,000 块长寿命评测单独保存在 Candidate run 的 `evaluations/validation/*_longlife.json`；其截断率不与 5000 块协议混合比较。", ""]
    (ROOT / "reports/experiments/ppo_candidate_raw_10m_comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
