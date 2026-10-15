"""Sequential, restartable PPO30M Raw/Risk training and paired long evaluation."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from training.evaluation.hybrid_queue import HASHES, PPO, BC, verify_models
from training.evaluation.ppo30m_overnight_eval import RUN, DEV, CONFIRM, evaluate
from training.evaluation.ppo30m_overnight_audit import audit
from training.imitation_dataset import digest
from training.train_ppo import ROOT, atomic_json, now
from training.train_transaction import task_name
from training.train_vector_transaction import verify_task

VERSION = "ppo30m_overnight_kl_raw_risk_v1"
TRAIN = list(range(880000,880008))
STAGE1_TASK = 123  # 503,808 new samples, nearest Task boundary above 500K.
FINAL_TASK = 489   # 2,002,944 new samples, nearest Task boundary above 2M.
PRESETS = {name: ROOT/"configs"/f"ppo30m_overnight_kl_{name}.json" for name in ("raw","risk")}


def _status(stage, **values):
    atomic_json(RUN/"queue_status.json", {"version": VERSION, "stage": stage,
                "updated_at": now(), **values})
    report = ROOT/"reports/experiments/ppo30m_overnight_finetune_results.md"
    lines = ["# PPO 30M 夜间强化学习结果", "", f"当前阶段：`{stage}`；更新时间：{now()}。", "",
             "方案 A 是 PPO30M 初始化 + PPO30M KL 参考 + 原始奖励；方案 B 只额外加入 0.003 倍非负高度风险增长惩罚。单局训练上限 100,000。", "",
             "每方案第一阶段上限 503,808 新样本；仅通过冻结的开发验证门槛才继续至最多 2,002,944 新样本。所有长局触顶按右截断。", ""]
    for label in ("development_stage1", "development_final", "independent_confirmation"):
        path = RUN/"evaluations"/label/"summary.json"
        if path.exists():
            result = json.loads(path.read_text())
            lines += [f"## {label}", "", f"种子：{result['seeds']}；每局上限：{result['cap']}。", "",
                      "| 模型 | 平均存活 | P10 | 触顶率 | 平均正式得分 | 四消事件占比 | 共同存活得分比 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for name, value in result["variants"].items():
                lines.append(f"| {name} | {value['mean_pieces']:.1f} | {value['p10_pieces']} | {value['cap_rate']:.1%} | {value['mean_score']:,.0f} | {value['four_event_share']:.1%} | {value.get('mean_common_score_ratio',1):.3f} |")
            lines += ["", "逐 seed 配对和完整计分轨迹保存在独立 run 的原子结果文件中。", ""]
    for label in ("raw_stage1","risk_stage1","raw_final","risk_final"):
        path = RUN/"audits"/f"{label}.json"
        if path.exists():
            result = json.loads(path.read_text())
            lines += [f"正常 PPO 状态审计 {label}：Top1 一致率 {result['normal_top1_agreement']:.1%}，Top5 保留率 {result['normal_top5_retention']:.1%}，Masked KL {result['masked_kl_to_ppo']:.4f}。", ""]
    lines += ["训练、验证、确认种子分别隔离；验证未完成前不宣称模型性能改善。原 PPO30M、BC 与 Hybrid B12 权重均保持冻结。", ""]
    report.write_text("\n".join(lines))


def freeze():
    verify_models()
    if digest(PPO) != HASHES["ppo"] or digest(BC) != HASHES["bc"]:
        raise RuntimeError("Frozen model changed")
    seeds = TRAIN+DEV+CONFIRM
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Seed groups overlap")
    for seed in seeds:
        for base in (ROOT/"runs", ROOT/"reports"):
            for path in base.rglob(f"seed_{seed}.*"):
                if not path.is_relative_to(RUN):
                    raise RuntimeError(f"Seed already used: {path}")
    configs = {name: json.loads(path.read_text()) for name,path in PRESETS.items()}
    for name, config in configs.items():
        if (config["base_seed"] != TRAIN[0] or config["max_pieces"] != 100000 or
            config["n_envs"] != 8 or config["task_steps"] != 4096 or
            config["kl_reference"]["sha256"] != HASHES["ppo"] or
            config["initial_model"]["sha256"] != HASHES["ppo"] or
            config["kl_reference"]["beta"] != .03 or
            config.get("height_penalty_coef", 0) != (0 if name=="raw" else .003)):
            raise RuntimeError(f"Invalid frozen preset: {name}")
    protocol = {"version": VERSION, "models": HASHES,
        "config_sha256": {name:digest(path) for name,path in PRESETS.items()},
        "train_seeds": TRAIN, "development_seeds": DEV, "confirmation_seeds": CONFIRM,
        "train_episode_cap": 100000, "dev_cap": 50000, "confirm_cap": 100000,
        "stage1_task": STAGE1_TASK, "final_task": FINAL_TASK,
        "continue_gate": "four share drop <=5pp; mean common score >=90%; mean survival >=90%; P10 >=80%; cap rate drop <=10pp",
        "confirm_gate": "candidate mean survival and P10 both exceed PPO; four share drop <=5pp; mean common score >=90%"}
    path = RUN/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Frozen overnight protocol changed")
    if not path.exists(): atomic_json(path, protocol)
    return protocol


def _run_training(name, target_task):
    run_dir = RUN/name
    config = PRESETS[name]
    state_path = run_dir/"transaction_state.json"
    if not state_path.exists():
        raise RuntimeError(f"Required 4096-sample smoke task absent: {name}")
    state = json.loads(state_path.read_text())
    if state["committed_task"] < 1:
        raise RuntimeError(f"Smoke task did not commit: {name}")
    recorded = json.loads((run_dir/"config.json").read_text())
    if (recorded["max_pieces"] != 100000 or recorded["run_seed"] != TRAIN[0] or
        recorded["kl_reference"]["sha256"] != HASHES["ppo"] or
        recorded.get("height_penalty_coef",0) != (0 if name=="raw" else .003)):
        raise RuntimeError("Run does not match frozen overnight configuration")
    verify_task(run_dir/"committed"/task_name(state["committed_task"]),
                state["committed_task"], recorded)
    if state["committed_task"] >= target_task: return state
    remaining = target_task-state["committed_task"]
    _status(f"{name}_training", committed_steps=state["committed_steps"], target_steps=target_task*4096)
    with (RUN/f"{name}_console.log").open("a",encoding="utf8") as log:
        subprocess.run([sys.executable,"-m","training.train_vector_transaction",
                        "--run-dir",str(run_dir),"--resume","--config-file",str(config),
                        "--max-tasks",str(remaining)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    state = json.loads(state_path.read_text())
    if state["committed_task"] != target_task or state["committed_steps"] != target_task*4096:
        raise RuntimeError(f"Stage did not reach exact Task boundary: {name}")
    return state


def _freeze_checkpoint(name, task, label):
    source = RUN/name/"committed"/task_name(task)/"model.zip"
    destination = RUN/"frozen"/f"{name}_{label}.zip"
    if destination.exists():
        record = json.loads(destination.with_suffix(".json").read_text())
        if record["task"] != task or digest(destination) != record["sha256"]:
            raise RuntimeError("Frozen milestone changed")
        if source.exists() and digest(destination) != digest(source):
            raise RuntimeError("Frozen milestone differs from source Task")
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    shutil.copy2(source,temporary)
    os.replace(temporary,destination)
    atomic_json(destination.with_suffix(".json"), {"variant":name,"task":task,"sha256":digest(destination),
                                                  "source_sha256":digest(source),"frozen_at":now()})
    return destination


def _gate(summary, name, kind):
    base, item = summary["variants"]["ppo"], summary["variants"][name]
    four_ok = item["four_event_share"] >= base["four_event_share"]-.05
    score_ok = item["mean_common_score_ratio"] >= .9
    if kind == "continue":
        return (four_ok and score_ok and item["mean_pieces"] >= .9*base["mean_pieces"] and
                item["p10_pieces"] >= .8*base["p10_pieces"] and
                item["cap_rate"] >= base["cap_rate"]-.1)
    return (four_ok and score_ok and item["mean_pieces"] > base["mean_pieces"] and
            item["p10_pieces"] > base["p10_pieces"])


def main():
    try:
        freeze()
        _status("smoke_verified")
        paths = {"ppo":PPO}
        for name in ("raw","risk"):
            _run_training(name, STAGE1_TASK)
            paths[name] = _freeze_checkpoint(name,STAGE1_TASK,"stage1")
            audit(paths[name],f"{name}_stage1")
        _status("development_stage1")
        paths["hybrid_b12"] = PPO
        stage1 = evaluate(paths,DEV,50000,"development_stage1")
        decisions = {name:_gate(stage1,name,"continue") for name in ("raw","risk")}
        atomic_json(RUN/"stage1_go_no_go.json", {"decisions":decisions,"evaluated_at":now()})
        _status("stage1_decided",decisions=decisions)
        if not any(decisions.values()):
            _status("completed_no_improvement",decisions=decisions)
            return
        final_paths = {"ppo":PPO,"hybrid_b12":PPO}
        for name in ("raw","risk"):
            if decisions[name]:
                _run_training(name, FINAL_TASK)
                final_paths[name] = _freeze_checkpoint(name,FINAL_TASK,"final")
                audit(final_paths[name],f"{name}_final")
            else:
                final_paths[name] = paths[name]
        _status("development_final")
        final = evaluate(final_paths,DEV,100000,"development_final")
        confirm = {name:_gate(final,name,"confirm") for name in ("raw","risk")}
        atomic_json(RUN/"final_go_no_go.json", {"decisions":confirm,"evaluated_at":now()})
        if any(confirm.values()):
            _status("independent_confirmation",decisions=confirm)
            selected = {"ppo":PPO,**{name:final_paths[name] for name in ("raw","risk") if confirm[name]},
                        "hybrid_b12":PPO}
            evaluate(selected,CONFIRM,100000,"independent_confirmation")
        _status("completed",stage1=decisions,confirmation=confirm)
    except Exception as error:
        _status("failed",error=repr(error))
        raise


if __name__ == "__main__": main()
