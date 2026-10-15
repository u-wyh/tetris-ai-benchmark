"""Resumable, resource-aware overnight A/B curriculum PPO experiment."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from training.env.failure_curriculum_env import file_sha256
from training.evaluation.failure_curriculum_eval import DEV, CONFIRM, audit_seeds, evaluate, gate, checkpoint
from training.evaluation.hybrid_queue import PPO, HASHES, verify_models
from training.failure_curriculum_data import ALL_TIERS, DATA_PATH, RUN, build
from training.train_ppo import ROOT, atomic_json, now
from training.train_transaction import read_state
from training.train_vector_transaction import run, sha256, verify_task

INITIAL_TASKS = 123  # 503,808 actual samples, a complete 4096-sample task boundary.
FINAL_TASKS = 489    # 2,002,944 actual samples; only B can pass the extension gate.
BASE_SEED = 994000


def status(stage, **extra):
    RUN.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN/"status.json", {"stage":stage,"pid":os.getpid(),"updated_at":now(),**extra})
    print(now(),stage,extra,flush=True)


def operation(message,evidence):
    with (ROOT/"OPERATIONS.md").open("a") as file:
        file.write(f"\n- {now()}｜{message}。依据：{evidence}。\n")


def command(args, name):
    with (RUN/name).open("w") as log:
        subprocess.run(args,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)


def wait_for_resource():
    status("waiting_for_death_analysis")
    while True:
        path = ROOT/"runs/ppo30m_death_analysis/pipeline_status.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["stage"] == "completed":
                break
            if row["stage"] == "failed":
                raise RuntimeError("Death-analysis queue failed; curriculum source is unvalidated")
        if subprocess.run(["tmux","has-session","-t","tetris-ppo-death-analysis-pipeline"],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError("Death-analysis pipeline disappeared without completion")
        time.sleep(60)
    while True:
        result = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                                text=True,capture_output=True,check=True)
        used = int(result.stdout.strip().splitlines()[0])
        if used < 500:
            break
        status("waiting_for_gpu",used_mib=used)
        time.sleep(60)


def preset(variant,dataset):
    return {"task_steps":4096,"n_envs":8,"n_steps":512,"target_total_steps":FINAL_TASKS*4096,
            "device":"cuda","base_seed":BASE_SEED,"candidate_policy":True,"max_pieces":100000,
            "final_test_seeds":100,"final_test_max_pieces":50000,
            "initial_model":{"path":str(PPO),"sha256":HASHES["ppo"]},
            "kl_reference":{"path":str(PPO),"sha256":HASHES["ppo"],"beta":.03},
            "overnight_monitor":True,"step0_baseline":False,
            "transaction_retention":3,"milestone_interval":100000000,
            "validation_interval":100000000,
            "failure_curriculum":{"mode":variant,"dataset_path":str(DATA_PATH),
                                  "dataset_sha256":dataset["dataset_sha256"],
                                  "ppo_sha256":HASHES["ppo"],
                                  "snapshot_probability":.7,"window_pieces":5000,
                                  "death_penalty":-20.0,"tiers":list(ALL_TIERS)}}


def freeze(dataset):
    verify_models()
    audit_seeds()
    if file_sha256(DATA_PATH) != dataset["dataset_sha256"]:
        raise RuntimeError("Curriculum dataset changed before training")
    protocol = {"created_at":now(),"git_commit":subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "ppo_sha256":HASHES["ppo"],"bc_sha256":HASHES["bc"],
        "dataset_sha256":dataset["dataset_sha256"],"development_seeds":list(DEV),
        "confirmation_seeds":list(CONFIRM),"normal_and_curriculum_initial_tasks":INITIAL_TASKS,
        "conditional_curriculum_max_tasks":FINAL_TASKS,
        "gate":"four drop <=3pp, common score >=95%, P10 >=95%, mean +>=1000, no extra <10K death",
        "extension":"only B if development gate passes; A fixed 503808-sample control"}
    path = RUN/"protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != value for k,value in protocol.items() if k not in ("created_at","git_commit")):
            raise RuntimeError("Frozen curriculum protocol changed")
    else:
        atomic_json(path,protocol)
    for variant in ("normal","curriculum"):
        output = RUN/f"{variant}_preset.json"
        content = preset(variant,dataset)
        if output.exists() and json.loads(output.read_text()) != content:
            raise RuntimeError("Frozen training preset changed")
        atomic_json(output,content)
    return protocol


def ensure_tasks(variant,target):
    run_dir = RUN/variant
    config_path = RUN/f"{variant}_preset.json"
    pointer = run_dir/"transaction_state.json"
    if not pointer.exists():
        status(f"{variant}_smoke")
        run(run_dir,resume=False,max_tasks=1,config_file=str(config_path))
    state = read_state(run_dir)
    if state["committed_task"] < 1:
        raise RuntimeError("No committed smoke transaction")
    verify_task(run_dir/"committed/task_000001",1,json.loads((run_dir/"config.json").read_text()))
    metrics = json.loads((run_dir/"committed/task_000001/metrics.json").read_text())
    if (metrics["gpu_peak_allocated_mib"] is None or metrics["gpu_peak_allocated_mib"] >= 3900
            or metrics["reference_kl"] is None or metrics["gradient_norm"] is None
            or (variant=="curriculum" and metrics["snapshot_steps"]==0)
            or (variant=="normal" and metrics["snapshot_steps"]!=0)):
        raise RuntimeError(f"{variant} smoke quality gate failed")
    status(f"{variant}_training",committed_tasks=state["committed_task"],target_tasks=target)
    if state["committed_task"] < target:
        run(run_dir,resume=True,max_tasks=target-state["committed_task"],config_file=str(config_path))
    state = read_state(run_dir)
    if state["committed_task"] != target:
        raise RuntimeError("Training stopped before complete task boundary")
    operation(f"{variant} 完成 {target*4096} 个环境交互样本、{target} 个事务任务",
              f"runs/ppo30m_failure_curriculum/{variant}/state.json")
    return state


def training_signal():
    rows = {}
    for variant in ("normal","curriculum"):
        folder = RUN/variant/"metric_history"
        values = [json.loads(path.read_text()) for path in sorted(folder.glob("task_*.json"))[:INITIAL_TASKS]]
        rows[variant] = {"steps":sum(x["end_step"]-x["start_step"] for x in values),
                         "true_deaths":sum(x["true_deaths"] for x in values),
                         "snapshot_steps":sum(x["snapshot_steps"] for x in values),
                         "normal_steps":sum(x["normal_steps"] for x in values),
                         "window_truncations":sum(x["window_truncations"] for x in values),
                         "mean_reference_kl":sum(x["reference_kl"] for x in values)/len(values),
                         "source_seed_counts":{}}
        for row in values:
            for seed,count in json.loads(row["source_seed_counts_json"]).items():
                counts = rows[variant]["source_seed_counts"]
                counts[seed] = counts.get(seed,0)+count
    atomic_json(RUN/"training_signal.json",rows)
    return rows


def write_reports(dataset, signal, development, decision, extended=None, confirmation=None):
    report_dir = ROOT/"reports/experiments"
    design = ["# PPO 30M 危险状态课程学习：冻结设计", "",
              "A/B 均从同一冻结 PPO 30M 初始化，采用相同 PPO 超参、真实死亡 -20、masked KL β=0.03、γ=0.995。",
              "A 从空棋盘起步；B 70% 从完整 Core/RNG 危险快照、30% 正常起步。快照来源只用于训练，Actor 只接收原 237+16 公开输入及合法 Mask。",
              f"训练清单 SHA256：`{dataset['dataset_sha256']}`；训练快照 {dataset['snapshots']} 个，真实死亡来源 {dataset['independent_death_seeds']} seed。",
              f"层数：`{dataset['tier_counts']}`。先各 503808 步；仅 B 通过冻结开发门槛可追加至 2002944 步。",
              f"新开发种子：{list(DEV)}；独立确认种子：{list(CONFIRM)}。所有性能对照从空棋盘开始。", ""]
    (report_dir/"ppo30m_failure_curriculum_design.md").write_text("\n".join(design))
    training = ["# PPO 30M 危险课程训练结果", "",
                "| 方案 | 样本数 | 真正死亡 | 每十万步死亡 | 危险快照步数 | 正常步数 | 窗口截断 | 平均参考 KL |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,row in signal.items():
        training.append(f"| {name} | {row['steps']} | {row['true_deaths']} | {1e5*row['true_deaths']/row['steps']:.1f} | "
                        f"{row['snapshot_steps']} | {row['normal_steps']} | {row['window_truncations']} | {row['mean_reference_kl']:.4f} |")
    training += ["", "逐 Task 损失、熵、梯度、解释方差、奖励和来源 seed 分布位于 `runs/ppo30m_failure_curriculum/{normal,curriculum}/metric_history/`。",
                 "同一快照重复死亡会增加训练信号计数，不代表新增独立失败 episode。", ""]
    (report_dir/"ppo30m_failure_curriculum_training.md").write_text("\n".join(training))
    lines = ["# PPO 30M 危险课程完整对局比较", "",
             "开发评测：相同 16 个全新种子、每局 50K，全部从空棋盘开始。", "",
             "| 方案 | 平均存活 | P10 | 触顶 | 平均总分 | 每千块得分 | 四消比例 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name,row in development["summary"].items():
        lines.append(f"| {name} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | {row['capped']}/16 | "
                     f"{row['mean_score']:.0f} | {row['score_per_1000']:.0f} | {row['four_fraction']:.2%} |")
    lines += ["", f"冻结门槛：`{decision}`。触顶局按右截断；开发组轻微差异不代表独立泛化优势。", ""]
    if extended:
        lines.append(f"B 追加训练后同开发组复核：`{extended['gate']}`。")
    if confirmation:
        lines += ["", "## 独立 100K 确认", "",
                  "| 方案 | 平均存活 | P10 | 触顶 | 平均总分 | 四消比例 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name,row in confirmation["summary"].items():
            lines.append(f"| {name} | {row['mean_pieces']:.0f} | {row['p10_pieces']:.0f} | "
                         f"{row['capped']}/32 | {row['mean_score']:.0f} | {row['four_fraction']:.2%} |")
    else:
        lines.append("开发门槛未通过时，不启动独立确认；不把未评测说成训练失败或胜出。")
    lines += ["", "逐 seed、配对共同得分、策略保持样本和模型 SHA256 在 `runs/ppo30m_failure_curriculum/`，大型模型未提交 Git。", ""]
    (report_dir/"ppo30m_failure_curriculum_comparison.md").write_text("\n".join(lines))


def main():
    try:
        wait_for_resource()
        status("inventory_risk_snapshots")
        dataset = build()
        freeze(dataset)
        operation(f"冻结危险课程数据：{dataset['snapshots']} 状态、{dataset['independent_death_seeds']} 独立死亡 seed",
                  "runs/ppo30m_failure_curriculum/curriculum_manifest.json")
        status("pretraining_regression_tests")
        command([sys.executable,"-m","pytest","-q"],"python_tests.log")
        command([sys.executable,"-m","unittest","discover","-s","tests/parity","-v"],"parity_tests.log")
        command(["node","--test",*[str(p) for p in sorted((ROOT/"tests").glob("*.test.cjs"))]],"js_tests.log")
        ensure_tasks("normal",1)
        ensure_tasks("curriculum",1)
        ensure_tasks("normal",INITIAL_TASKS)
        ensure_tasks("curriculum",INITIAL_TASKS)
        signal = training_signal()
        status("paired_development_50k")
        development = evaluate("development_50k",("ppo","B12","normal","curriculum"),DEV,50000)
        decision = gate(development)
        extended = confirmation = None
        if decision["extend_curriculum"]:
            ensure_tasks("curriculum",FINAL_TASKS)
            status("extended_development_50k")
            extended_result = evaluate("extended_development_50k",("ppo","B12","normal","curriculum"),DEV,50000,FINAL_TASKS)
            extended_gate = gate(extended_result,"extended_development")
            extended = {"result":extended_result,"gate":extended_gate}
            if extended_gate["extend_curriculum"]:
                status("independent_confirmation_100k")
                confirmation = evaluate("confirmation_100k",("ppo","bc","B12","normal","curriculum"),
                                        CONFIRM,100000,FINAL_TASKS)
        status("final_reports")
        write_reports(dataset,signal,development,decision,extended,confirmation)
        operation("危险课程 A/B 训练、冻结门槛和有条件完整对局评测完成",
                  "runs/ppo30m_failure_curriculum/{training_signal.json,development_gate.json,status.json}")
        status("publishing")
        subprocess.run(["git","add","OPERATIONS.md","CHANGELOG.md",
                        "reports/experiments/ppo30m_failure_curriculum_design.md",
                        "reports/experiments/ppo30m_failure_curriculum_training.md",
                        "reports/experiments/ppo30m_failure_curriculum_comparison.md"],cwd=ROOT,check=True)
        if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode==1:
            subprocess.run(["git","commit","-m","Report PPO30M failure curriculum experiment"],cwd=ROOT,check=True)
        subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True)
        commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        status("completed",commit=commit,trained_steps={"normal":INITIAL_TASKS*4096,
                                                          "curriculum":(FINAL_TASKS if decision["extend_curriculum"] else INITIAL_TASKS)*4096})
    except BaseException as error:
        status("failed",error=repr(error),traceback=traceback.format_exc())
        operation(f"危险课程后台队列失败并停止依赖阶段：{repr(error)}",
                  "runs/ppo30m_failure_curriculum/status.json")
        raise


if __name__ == "__main__":
    main()
