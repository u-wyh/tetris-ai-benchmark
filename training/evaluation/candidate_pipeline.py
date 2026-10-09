"""Recoverable Candidate-Scoring PPO smoke, 2M training and paired validation."""

import json
import pickle
import subprocess

import torch
from sb3_contrib import MaskablePPO

from analysis.ppo_five_model_board import audit, old_rows, paired, replay_rows, summarize
from training.evaluation.evaluate import evaluate_model
from training.train_ppo import ROOT, atomic_json, now
from training.train_vector_transaction import verify_task

PIPELINE = ROOT / "runs/ppo_candidate_pipeline_seed42"
STATE = PIPELINE / "status.json"
SMOKE = ROOT / "runs/ppo_candidate_smoke_v2_seed42"
CONTROL = ROOT / "runs/ppo_candidate_smoke_uninterrupted_v2_seed42"
RUN = ROOT / "runs/ppo_candidate_raw_2m_seed42"
CONFIG = ROOT / "configs/ppo_candidate_raw_2m_seed42.json"
SMOKE_CONFIG = ROOT / "configs/ppo_candidate_smoke_seed42.json"
RAW_MODEL = ROOT / "runs/ppo_raw_10m_seed42/milestones/step_002002944/model.zip"
REPORT = ROOT / "reports/experiments/ppo_candidate_raw_2m_comparison.md"
SEEDS = list(range(100000, 100032))
STAGES = ("feature_tests", "performance_smoke", "train_2m",
          "paired_evaluation", "report", "completed")


def command(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def run_state(run_dir):
    path = run_dir / "transaction_state.json"
    return json.loads(path.read_text()) if path.exists() else None


def do_feature_tests():
    command(ROOT / ".venv/bin/python", "-m", "pytest", "-q",
            "tests/test_candidate_ppo.py", "tests/test_ppo_five_model_board.py",
            "tests/parity/test_parity.py")
    command("node", "--test", *sorted((ROOT / "tests").glob("*.test.cjs")))


def do_performance_smoke():
    if run_state(SMOKE) is None:
        command(ROOT / ".venv/bin/python", "-m", "training.train_vector_transaction",
                "--run-dir", SMOKE, "--config-file", SMOKE_CONFIG, "--max-tasks", "1")
    while (run_state(SMOKE) or {}).get("committed_task", 0) < 2:
        command(ROOT / ".venv/bin/python", "-m", "training.train_vector_transaction",
                "--run-dir", SMOKE, "--resume", "--max-tasks", "1")
    state = run_state(SMOKE)
    if state["status"] != "completed" or state["committed_steps"] != 8192:
        raise RuntimeError("Candidate smoke did not finish two committed Tasks")
    config = json.loads((SMOKE / "config.json").read_text())
    rows = [verify_task(SMOKE / "committed" / f"task_{number:06d}", number, config)
            for number in (1, 2)]
    if any(row["gpu_peak_mib"] is None or row["gpu_peak_mib"]["reserved"] >= 4096
           for row in rows):
        raise RuntimeError("CUDA memory smoke failed")
    rates = []
    for number in (1, 2):
        metrics = json.loads((SMOKE / "committed" / f"task_{number:06d}" / "metrics.json").read_text())
        rates.append(4096 / metrics["wall_seconds"])
    if run_state(CONTROL) is None:
        command(ROOT / ".venv/bin/python", "-m", "training.train_vector_transaction",
                "--run-dir", CONTROL, "--config-file", SMOKE_CONFIG, "--max-tasks", "2")
    elif (run_state(CONTROL) or {}).get("committed_task", 0) < 2:
        command(ROOT / ".venv/bin/python", "-m", "training.train_vector_transaction",
                "--run-dir", CONTROL, "--resume", "--max-tasks", "2")
    if (run_state(CONTROL) or {}).get("status") != "completed":
        raise RuntimeError("Uninterrupted candidate control did not complete")
    a = MaskablePPO.load(str(SMOKE / "final/model.zip"), device="cpu")
    b = MaskablePPO.load(str(CONTROL / "final/model.zip"), device="cpu")
    if any(not torch.equal(value, b.policy.state_dict()[key])
           for key, value in a.policy.state_dict().items()):
        raise RuntimeError("Resumed and uninterrupted candidate parameters differ")
    a_opt, b_opt = a.policy.optimizer.state_dict(), b.policy.optimizer.state_dict()
    if a_opt["param_groups"] != b_opt["param_groups"] or a_opt["state"].keys() != b_opt["state"].keys():
        raise RuntimeError("Resumed and uninterrupted optimizers differ")
    for key in a_opt["state"]:
        for field, value in a_opt["state"][key].items():
            other = b_opt["state"][key][field]
            if not (torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other):
                raise RuntimeError("Resumed and uninterrupted optimizer values differ")
    for name in ("trainer_state.pkl",):
        a_state = pickle.loads((SMOKE / "committed/task_000002" / name).read_bytes())
        b_state = pickle.loads((CONTROL / "committed/task_000002" / name).read_bytes())
        if (a_state["num_timesteps"] != b_state["num_timesteps"]
                or not torch.equal(a_state["torch_cpu_random"], b_state["torch_cpu_random"])
                or any(not torch.equal(x, y) for x, y in zip(
                    a_state["torch_cuda_random"], b_state["torch_cuda_random"]))):
            raise RuntimeError("Resumed and uninterrupted RNG states differ")
    control_config = json.loads((CONTROL / "config.json").read_text())
    control_meta = verify_task(CONTROL / "committed/task_000002", 2, control_config)
    if rows[-1]["worker_digests"] != control_meta["worker_digests"]:
        raise RuntimeError("Resumed and uninterrupted worker states differ")
    atomic_json(PIPELINE / "performance_smoke.json", {
        "steps_per_second": rates,
        "gpu_peak_mib": [row["gpu_peak_mib"] for row in rows],
        "resumed_from_task_1": True,
        "identical_to_uninterrupted": True,
        "completed_at": now()})


def do_train():
    state = run_state(RUN)
    if state and state["status"] == "completed":
        command(ROOT / ".venv/bin/python", "-m", "training.train_vector_transaction",
                "--run-dir", RUN, "--resume")
        return
    args = [ROOT / ".venv/bin/python", "-m", "training.train_vector_transaction",
            "--run-dir", RUN]
    args += ["--resume"] if state else ["--config-file", CONFIG]
    command(*args)
    state = run_state(RUN)
    if state["status"] != "completed" or state["committed_steps"] != 2_002_944:
        raise RuntimeError("Candidate 2M run did not fully commit")


def do_paired_evaluation():
    output = PIPELINE / "paired_validation_2m.json"
    if output.exists():
        data = json.loads(output.read_text())
        if data["seeds"] == SEEDS and data["max_pieces"] == 5000:
            return
        raise RuntimeError("Existing candidate evaluation protocol differs")
    candidate_model = RUN / "milestones/step_002002944/model.zip"
    if not candidate_model.is_file():
        raise FileNotFoundError(candidate_model)
    raw = evaluate_model(RAW_MODEL, SEEDS, 5000, "candidate_paired_validation_2m", 2_002_944,
                         diagnostics=True)
    candidate = evaluate_model(candidate_model, SEEDS, 5000,
                               "candidate_paired_validation_2m", 2_002_944, diagnostics=True)
    raw_board, candidate_board = old_rows("raw"), replay_rows(candidate_model)
    audit(raw_board, raw)
    audit(candidate_board, candidate)
    atomic_json(output, {"seeds": SEEDS, "max_pieces": 5000, "deterministic": True,
                         "raw": raw, "candidate": candidate,
                         "raw_board": summarize(raw_board),
                         "candidate_board": summarize(candidate_board),
                         "paired_pieces": paired(candidate_board, raw_board)})


def do_report():
    data = json.loads((PIPELINE / "paired_validation_2m.json").read_text())
    perf = json.loads((PIPELINE / "performance_smoke.json").read_text())
    raw, candidate = data["raw"]["aggregate"], data["candidate"]["aggregate"]
    rb, cb = data["raw_board"], data["candidate_board"]
    comparison = data["paired_pieces"]
    config = json.loads((RUN / "config.json").read_text())
    metadata = json.loads((RUN / "metadata.json").read_text())
    task = json.loads((RUN / "committed/task_000489/task_meta.json").read_text())
    lines = ["# Candidate-Scoring PPO Raw 2M 对照", "",
             "同一批 validation seeds `100000..100031`、每局最多 5000 块、确定性推理与官方 Action Mask。两个模型均训练到 2,002,944 步，使用相同 Raw Reward；候选模型从 seed42 随机初始化，使用版本化公开特征。", "",
             "| 指标 | Raw MLP | Candidate-Scoring |", "|---|---:|---:|",
             f"| 平均存活 | {raw['mean_pieces']:.2f} | {candidate['mean_pieces']:.2f} |",
             f"| 中位存活 | {raw['median_pieces']:.2f} | {candidate['median_pieces']:.2f} |",
             f"| P90 存活 | {raw['p90_pieces']} | {candidate['p90_pieces']} |",
             f"| 平均消行 | {raw['mean_lines']:.2f} | {candidate['mean_lines']:.2f} |",
             f"| 平均正式分数 | {raw['mean_score']:.2f} | {candidate['mean_score']:.2f} |",
             f"| 每 100 块新增洞 | {rb['new_holes_per_100_pieces']:.2f} | {cb['new_holes_per_100_pieces']:.2f} |",
             f"| 死亡时平均最高列 | {rb['mean_death_max_height']:.2f} | {cb['mean_death_max_height']:.2f} |",
             f"| 中央列最高比例 | {rb['central_tallest_rate']:.1%} | {cb['central_tallest_rate']:.1%} |",
             "",
             f"配对平均存活差（Candidate − Raw）为 {comparison['mean_difference']:.2f} 块；固定种子、按整局重采样 10,000 次的 95% bootstrap 区间为 {comparison['bootstrap_95ci']}。逐 seed 差保存在 `runs/ppo_candidate_pipeline_seed42/paired_validation_2m.json`。", "",
             f"8-worker CUDA smoke 吞吐：{[round(x, 2) for x in perf['steps_per_second']]} steps/s；PyTorch 峰值预留显存：{[x['reserved'] for x in perf['gpu_peak_mib']]} MiB。正式最后 Task 峰值：{task['gpu_peak_mib']} MiB。", "",
             f"配置：`{config['policy']}`、`{config['observation_version']}`、`{config['candidate_feature_version']}`、seed {config['run_seed']}、Raw Reward。训练代码 commit `{metadata['git_commit']}`。", "",
             "本次只用 validation seeds 评估，不使用 final-test seeds 调参，也未启动 10M。", ""]
    REPORT.write_text("\n".join(lines))


def main():
    PIPELINE.mkdir(parents=True, exist_ok=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "stage": "feature_tests", "status": "running", "created_at": now(),
        "git_commit": git_commit}
    while state["stage"] != "completed":
        stage = state["stage"]
        if stage not in STAGES:
            raise ValueError(f"Unknown candidate pipeline stage: {stage}")
        state.update(status="running", updated_at=now())
        atomic_json(STATE, state)
        try:
            {"feature_tests": do_feature_tests,
             "performance_smoke": do_performance_smoke,
             "train_2m": do_train,
             "paired_evaluation": do_paired_evaluation,
             "report": do_report}[stage]()
        except Exception as error:
            state.update(status="failed", error=repr(error), updated_at=now())
            atomic_json(STATE, state)
            raise
        state.pop("error", None)
        state["stage"] = STAGES[STAGES.index(stage) + 1]
        state.update(status="completed" if state["stage"] == "completed" else "running",
                     updated_at=now())
        atomic_json(STATE, state)


if __name__ == "__main__":
    main()
