"""Independent, restartable next-stage BC generalization/KL/DAgger queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from training.imitation_dataset import digest
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/candidate_bc_next_stage"
CHAMPION = RUN / "champion/model.zip"
KL = ROOT / "runs/candidate_bc_klppo_2m"
DAGGER = ROOT / "runs/candidate_bc_dagger_v1"
KL_TUNING = RUN / "kl_tuning"
KL_BETAS = {"beta_001": 0.01, "beta_003": 0.03}
STAGES = ("new_seed_generalization_test", "analyze_ppo_degradation", "kl_ppo_smoke_tests",
          "tune_kl_beta", "train_kl_ppo_2m", "evaluate_kl_ppo", "collect_dagger_data",
          "collect_dagger_validation", "dagger_supervised_training",
          "evaluate_dagger_tuning", "evaluate_dagger", "comparative_analysis",
          "long_validation_if_saturated", "generate_reports", "publish_reports")


def command(*args):
    result = subprocess.run([sys.executable, "-m", *map(str, args)], cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str, args))}")


def wait_generalization():
    path = RUN / "generalization/status.json"
    while True:
        if path.exists():
            state = json.loads(path.read_text())
            if state["state"] == "completed":
                return
            if state["state"] == "failed":
                raise RuntimeError(f"Fresh-seed test failed: {state.get('error')}")
        session = subprocess.run(["tmux", "has-session", "-t", "tetris-bc-generalization"],
                                 capture_output=True, check=False)
        if session.returncode:
            raise RuntimeError("Fresh-seed tmux session exited without completed status")
        time.sleep(30)


def prepare_kl(beta, target_steps, run_seed, path):
    config = json.loads((ROOT / "configs/ppo_candidate_raw_2m_seed42.json").read_text())
    model_hash = digest(CHAMPION)
    config.update({"base_seed": run_seed, "target_total_steps": target_steps,
                   "step0_baseline": False,
                   "initial_model": {"path": str(CHAMPION.resolve()), "sha256": model_hash,
                                     "actor": "frozen 20M PPO + BC epoch 3",
                                     "critic": "20M PPO critic unchanged by actor-only BC",
                                     "optimizer": "fresh", "ppo_steps_before_run": 0},
                   "kl_reference": {"path": str(CHAMPION.resolve()), "sha256": model_hash,
                                    "beta": beta, "direction": "KL(current||BC)",
                                    "legal_actions_only": True}})
    if target_steps < 1_000_000:
        config.update({"validation_interval": 1_000_000, "milestone_interval": 1_000_000})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Frozen KL-PPO config changed")
    if not path.exists():
        atomic_json(path, config)
    return path


def train_transaction(run_dir, config):
    state = run_dir / "transaction_state.json"
    command("training.train_vector_transaction", "--run-dir", run_dir,
            *( ["--resume"] if state.exists() else ["--config-file", config]))
    result = json.loads(state.read_text())
    if result["status"] != "completed":
        raise RuntimeError(f"KL-PPO run did not complete: {run_dir}")
    return result


def model_evaluation(label, model, seed_set="validation", cap=20000):
    command("training.evaluation.next_stage_models", "--label", label,
            "--model", model, "--seed-set", seed_set, "--cap", cap)


def tune_kl_beta():
    # These eight seeds are disjoint from the held-out 30-seed BC/V2 comparison.
    model_evaluation("tuning_bc", CHAMPION, "tuning")
    scores = {}
    for index, (label, beta) in enumerate(KL_BETAS.items()):
        run_dir = KL_TUNING / label
        config = prepare_kl(beta, 250_000, 342 + 100 * index,
                            KL_TUNING / f"{label}_config.json")
        state = train_transaction(run_dir, config)
        if state["committed_steps"] < 250_000:
            raise RuntimeError(f"KL tuning run stopped early: {label}")
        comparison_label = f"tuning_kl_{label}"
        model_evaluation(comparison_label, run_dir / "final/model.zip", "tuning")
        scores[label] = json.loads((RUN / "comparisons" / comparison_label / "summary.json").read_text())
    # Pre-registered survival priority; prefer the stronger anchor on exact ties.
    selected = max(KL_BETAS, key=lambda label: (scores[label]["mean_pieces"],
                                                 scores[label]["p10_pieces"],
                                                 scores[label]["capped_count"],
                                                 KL_BETAS[label]))
    atomic_json(KL_TUNING / "selection.json",
                {"selected": selected, "beta": KL_BETAS[selected], "scores": scores,
                 "tuning_seeds": list(range(1260000, 1260008)),
                 "pilot_environment_steps_per_beta": 253952,
                 "selection_rule": "mean survival, P10, cap count, then stronger KL",
                 "fresh_30_seed_test_excluded": True, "selected_at": now()})


def train_kl():
    selection = json.loads((KL_TUNING / "selection.json").read_text())
    config = prepare_kl(selection["beta"], 2_000_000, 242, RUN / "kl_ppo_config.json")
    result = train_transaction(KL, config)
    if result["committed_steps"] < 2_000_000:
        raise RuntimeError("KL-PPO did not commit 2M steps")


def collect_dagger():
    command("training.dagger_dataset", "--output", DAGGER / "data",
            "--first-seed", 1240100, "--seeds", 50, "--target", 25000)


def collect_dagger_validation():
    command("training.dagger_dataset", "--output", DAGGER / "validation_data",
            "--first-seed", 1250000, "--seeds", 8, "--target", 2000)


def train_dagger():
    command("training.dagger_train", "--dagger-data", DAGGER / "data",
            "--dagger-validation", DAGGER / "validation_data",
            "--output", DAGGER / "bc", "--device", "cuda")


def tune_dagger():
    model_evaluation("tuning_bc", CHAMPION, "tuning")
    for epoch in (1, 2):
        model_evaluation(f"tuning_dagger_epoch_{epoch}", DAGGER / "bc" / f"epoch_{epoch:02d}.zip",
                         "tuning")
    candidates = {}
    for epoch in (1, 2):
        path = RUN / "comparisons" / f"tuning_dagger_epoch_{epoch}" / "summary.json"
        candidates[epoch] = json.loads(path.read_text())
    selected = max(candidates, key=lambda epoch: (candidates[epoch]["mean_pieces"],
                                                  candidates[epoch]["p10_pieces"], -epoch))
    atomic_json(DAGGER / "selected.json", {"epoch": selected, "rule": "tuning mean, then P10",
                                           "tuning_seeds": list(range(1260000, 1260008)),
                                           "final_test_seeds_excluded": True,
                                           "selected_at": now()})


def evaluate_dagger():
    selected = json.loads((DAGGER / "selected.json").read_text())
    model_evaluation("dagger_selected", DAGGER / "bc" / f"epoch_{selected['epoch']:02d}.zip")


def compare():
    from training.evaluation.next_stage_models import RUN as comparison_run
    source = ROOT / "runs/imitation_v2_hold_20m/validation"
    existing = {}
    for label in ("pure_30m", "bc", "bc_ppo_2m"):
        rows = list((source / label).glob("seed_*.json"))
        if len(rows) != 32:
            raise RuntimeError(f"Incomplete old matched comparison: {label}")
        pieces = [json.loads(path.read_text())["pieces_survived"] for path in rows]
        existing[label] = {"mean_pieces": sum(pieces) / 32,
                           "capped_count": sum(value == 20000 for value in pieces)}
    new = {label: json.loads((comparison_run / "comparisons" / label / "summary.json").read_text())
           for label in ("kl_ppo_2m", "dagger_selected")}
    atomic_json(RUN / "comparison_summary.json", {"existing": existing, "new": new,
                "protocol": "same 32 fixed validation seeds, deterministic, mask, 20k cap",
                "completed_at": now()})


def long_validation():
    summary = json.loads((RUN / "comparison_summary.json").read_text())
    selected = json.loads((DAGGER / "selected.json").read_text())["epoch"]
    models = {
        "bc": CHAMPION,
        "pure_30m": ROOT / "runs/ppo_candidate_raw_30m_seed42/final/model.zip",
        "kl_ppo_2m": KL / "final/model.zip",
        "dagger_selected": DAGGER / "bc" / f"epoch_{selected:02d}.zip",
    }
    counts = {name: data["capped_count"] for group in ("existing", "new")
              for name, data in summary[group].items()}
    chosen = [name for name in models if counts[name] >= 24]
    atomic_json(RUN / "long_validation_protocol.json",
                {"models": chosen, "selection_rule": "at least 24/32 capped at 20k",
                 "seeds": list(range(100000, 100032)), "max_pieces": 100000,
                 "same_seeds_for_all_selected_models": True})
    for name in chosen:
        model_evaluation(f"long_100k_{name}", models[name], cap=100000)


def generate_reports():
    general = json.loads((RUN / "generalization/summary.json").read_text())
    comparison = json.loads((RUN / "comparison_summary.json").read_text())
    long_protocol = json.loads((RUN / "long_validation_protocol.json").read_text())
    def append_report(filename, heading, body):
        import re
        path = ROOT / "reports/experiments" / filename
        existing = path.read_text()
        existing = re.sub(r"状态：\*\*[^\n]*?\*\*", "状态：**正式实验已完成。**",
                          existing, count=1)
        if heading in existing:
            existing = existing[:existing.index(heading)].rstrip() + "\n"
        path.write_text(existing + "\n" + heading + "\n\n" + body + "\n")

    bc = general["bc"]
    v2 = general["v2_hold"]
    append_report("candidate_bc_generalization.md", "## 正式实测结果",
                  f"BC：平均 {bc['mean_pieces']:.1f}，P10 {bc['p10_pieces']}，"
                  f"中位数 {bc['median_pieces']}，P90 {bc['p90_pieces']}，"
                  f"100k 截断 {bc['capped_count']}/30，平均消行 {bc['mean_lines']:.1f}，"
                  f"平均分 {bc['mean_score']:.1f}；"
                  f"V2 Hold：平均 {v2['mean_pieces']:.1f}，P10 {v2['p10_pieces']}，"
                  f"中位数 {v2['median_pieces']}，P90 {v2['p90_pieces']}，"
                  f"100k 截断 {v2['capped_count']}/30，平均消行 {v2['mean_lines']:.1f}，"
                  f"平均分 {v2['mean_score']:.1f}。"
                  f"双方均截断 {general['paired']['both_capped']} 局，"
                  f"BC 胜 {general['paired']['bc_wins']} 局、V2 胜 "
                  f"{general['paired']['v2_wins']} 局；双方都截断的局仅有相同存活下界。"
                  f"平均每局总耗时 BC {bc['mean_wall_seconds']:.1f} 秒、"
                  f"V2 {v2['mean_wall_seconds']:.1f} 秒。"
                  "现有结果未单独记录 BC 推理调用耗时，总耗时包含 Core 和特征计算。")
    kl = comparison["new"]["kl_ppo_2m"]
    append_report("candidate_kl_ppo_2m.md", "## 正式实测结果",
                  f"相同 32 个 validation seeds、20k 上限：KL-PPO 平均存活 "
                  f"{kl['mean_pieces']:.1f}，P10 {kl['p10_pieces']}，"
                  f"跑满 {kl['capped_count']}/32，平均消行 {kl['mean_lines']:.1f}。"
                  "与冻结 BC 的 32/32 及普通 BC+PPO 的 16/32 应按同一上限配对比较。")
    manifests = [json.loads(p.read_text()) for p in (DAGGER / "data").glob("seed_*.json")]
    totals = {key: sum(row[key] for row in manifests) for key in
              ("samples", "height_ge_12", "height_ge_16", "holes_ge_4", "learner_teacher_agreement")}
    dagger = comparison["new"]["dagger_selected"]
    append_report("candidate_dagger_v1.md", "## 正式实测结果",
                  f"共采集 {totals['samples']} 条 learner-visited 状态；高度≥12："
                  f"{totals['height_ge_12']}，高度≥16：{totals['height_ge_16']}，"
                  f"洞数≥4：{totals['holes_ge_4']}；师生动作相同 "
                  f"{totals['learner_teacher_agreement']}/{totals['samples']}。"
                  f"相同 32 个 validation seeds、20k 上限：DAgger 平均存活 "
                  f"{dagger['mean_pieces']:.1f}，P10 {dagger['p10_pieces']}，"
                  f"跑满 {dagger['capped_count']}/32。")
    lines = ["", "## 后台完成后的实测结果", "",
             "以下结果由可恢复队列在所有模型训练/评测完成后生成。", "",
             f"全新 30-seed × 100,000 块：BC {general['bc']['capped_count']}/30，"
             f"V2 Hold {general['v2_hold']['capped_count']}/30；"
             f"双方均截断 {general['paired']['both_capped']} 局。", "",
             "| 模型 | 20k 平均存活 | 20k 截断局数 |", "|---|---:|---:|"]
    for group in ("existing", "new"):
        for name, row in comparison[group].items():
            lines.append(f"| {name} | {row['mean_pieces']:.1f} | {row['capped_count']}/32 |")
    lines += ["", f"100k 长验证模型：{', '.join(long_protocol['models']) or '无'}。", ""]
    for name in long_protocol["models"]:
        row = json.loads((RUN / "comparisons" / f"long_100k_{name}" / "summary.json").read_text())
        lines.append(f"- {name}: 平均存活 {row['mean_pieces']:.1f}，P10 {row['p10_pieces']}，"
                     f"100k 截断 {row['capped_count']}/32。")
    lines.append("")
    report = ROOT / "reports/experiments/candidate_imitation_next_stage.md"
    append_report(report.name, "## 后台完成后的实测结果", "\n".join(lines[3:]).strip())
    atomic_json(RUN / "report_status.json", {"state": "generated", "path": str(report),
                                              "generated_at": now()})


def publish_reports():
    files = ["candidate_bc_generalization.md", "candidate_bc_ppo_degradation.md",
             "candidate_kl_ppo_2m.md", "candidate_dagger_v1.md",
             "candidate_imitation_next_stage.md"]
    paths = [f"reports/experiments/{name}" for name in files]
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    changes = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths],
                             cwd=ROOT, check=False)
    if changes.returncode == 1:
        subprocess.run(["git", "commit", "-m", "docs(imitation): publish completed KL and DAgger results",
                        "--", *paths], cwd=ROOT, check=True)
    elif changes.returncode:
        raise RuntimeError("Could not verify staged report changes")
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    atomic_json(RUN / "report_status.json", {"state": "published", "git_commit":
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "published_at": now()})


ACTIONS = {"new_seed_generalization_test": wait_generalization,
           "analyze_ppo_degradation": lambda: command("training.evaluation.bc_ppo_degradation"),
           "kl_ppo_smoke_tests": lambda: command("pytest", "-q", "tests/test_kl_maskable_ppo.py",
                                                  "tests/test_dagger_dataset.py",
                                                  "tests/test_kl_transaction.py"),
           "tune_kl_beta": tune_kl_beta,
           "train_kl_ppo_2m": train_kl,
           "evaluate_kl_ppo": lambda: model_evaluation("kl_ppo_2m", KL / "final/model.zip"),
           "collect_dagger_data": collect_dagger,
           "collect_dagger_validation": collect_dagger_validation,
           "dagger_supervised_training": train_dagger,
           "evaluate_dagger_tuning": tune_dagger,
           "evaluate_dagger": evaluate_dagger,
           "comparative_analysis": compare,
           "long_validation_if_saturated": long_validation,
           "generate_reports": generate_reports,
           "publish_reports": publish_reports}


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        marker = RUN / "stages" / f"{stage}.json"
        if marker.exists() and json.loads(marker.read_text())["state"] == "completed":
            continue
        marker.parent.mkdir(exist_ok=True)
        atomic_json(marker, {"state": "running", "started_at": now()})
        atomic_json(RUN / "queue_status.json", {"stage": stage, "state": "running", "updated_at": now()})
        print(f"STAGE {stage} started", flush=True)
        try:
            ACTIONS[stage]()
        except Exception:
            error = traceback.format_exc()
            atomic_json(marker, {"state": "failed", "error": error, "updated_at": now()})
            atomic_json(RUN / "queue_status.json", {"stage": stage, "state": "failed",
                                                    "error": error, "updated_at": now()})
            print(error, flush=True)
            raise
        atomic_json(marker, {"state": "completed", "completed_at": now()})
        print(f"STAGE {stage} completed", flush=True)
    atomic_json(RUN / "queue_status.json", {"stage": "completed", "state": "completed",
                                            "updated_at": now()})


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
