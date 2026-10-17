"""Bounded, resumable multi-seed PPO follow-up after the 3M paired games."""

import hashlib
import json
import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from training.env.failure_curriculum_env import file_sha256
from training.evaluation.failure_curriculum_eval import (
    _common_score, _summary, atomic_pickle, init_worker, task,
)
from training.evaluation.fast_legal import get_legal_placements as fast_placements
from training.evaluation.hybrid_queue import BC, PPO, digest, verify_models
from training.failure_curriculum_data import DATA_PATH
from training.failure_curriculum_queue import preset
from training.tetris_core import TetrisCore
from training.traditional.v2 import V2, V2Config
from training.train_ppo import ROOT, atomic_json, now
from training.train_transaction import read_state
from training.train_vector_transaction import run, verify_task

RUN = ROOT / "runs/ppo30m_multiseed_followup"
TRAIN_SEEDS = (994101, 994202, 994303)
EVAL_SEEDS = tuple(range(1420000, 1420016))
CAP = 100000
TASKS = 123
WORKERS = 8
CHECKPOINT_INTERVAL = 10000
VARIANTS = ("ppo", "bc", "B12") + tuple(
    f"{mode}_{seed}" for seed in TRAIN_SEEDS for mode in ("normal", "curriculum")
) + ("traditional_v2_hold",)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **extra):
    atomic_json(RUN / "status.json", {"stage": stage, "at": now(), "pid": os.getpid(), **extra})


def operation(message, evidence):
    with (ROOT / "OPERATIONS.md").open("a") as stream:
        stream.write(f"\n- {now()}｜{message}。依据：{evidence}。\n")


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_models()
    manifest = json.loads((ROOT / "runs/ppo30m_failure_curriculum/curriculum_manifest.json").read_text())
    if file_sha256(DATA_PATH) != manifest["dataset_sha256"]:
        raise RuntimeError("Frozen curriculum dataset changed")
    for seed in EVAL_SEEDS:
        for root in (ROOT / "runs", ROOT / "reports"):
            for path in root.rglob(f"seed_{seed}.*"):
                if not path.is_relative_to(RUN):
                    raise RuntimeError(f"Evaluation seed was previously used: {path}")
    protocol = {
        "training_seeds": list(TRAIN_SEEDS), "training_initial_sha256": digest(PPO),
        "bc_sha256": digest(BC),
        "traditional_v2_code_sha256": sha256(ROOT / "training/traditional/v2.py"),
        "training_modes": ["normal", "curriculum"], "training_tasks_per_run": TASKS,
        "training_steps_per_run": TASKS * 4096, "dataset_sha256": manifest["dataset_sha256"],
        "evaluation_seeds": list(EVAL_SEEDS), "evaluation_cap": CAP,
        "evaluation_workers": WORKERS, "variants": list(VARIANTS),
        "score_rule": "official_core_v1", "inference": "candidate_argmax_v1",
        "legal_placements": "bitboard_public_bfs_v1",
        "stop_rule": "Three fixed training seeds per mode; no automatic extension after this pilot",
    }
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Frozen multi-seed protocol changed")
    atomic_json(path, protocol)
    return manifest


def wait_for_long_games():
    while True:
        source = json.loads((ROOT / "runs/post_curriculum_3m/status.json").read_text())
        if source["stage"] == "completed":
            return
        if source["stage"] == "failed":
            raise RuntimeError("The 3M paired queue failed; waiting for resource review")
        status("waiting_for_3m_evaluation", source_stage=source["stage"])
        time.sleep(60)


def wait_for_gpu():
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"], text=True,
                                capture_output=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            used = int(result.stdout.splitlines()[0].strip())
            if used < 500:
                return
            status("waiting_for_gpu", used_mib=used)
        else:
            status("waiting_for_gpu_driver", detail=result.stderr.strip()[-300:])
        time.sleep(60)


def train_all(manifest):
    model_paths = {}
    for seed in TRAIN_SEEDS:
        for mode in ("normal", "curriculum"):
            name = f"{mode}_{seed}"
            config = preset(mode, manifest)
            config["base_seed"] = seed
            config_path = RUN / "presets" / f"{name}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise RuntimeError(f"Frozen training preset changed: {name}")
            atomic_json(config_path, config)
            run_dir = RUN / "training" / name
            pointer = run_dir / "transaction_state.json"
            committed = read_state(run_dir)["committed_task"] if pointer.exists() else 0
            if committed > TASKS:
                raise RuntimeError(f"Training exceeded fixed budget: {name}")
            status("training", variant=name, committed_tasks=committed, target_tasks=TASKS)
            if committed < TASKS:
                wait_for_gpu()
                run(run_dir, resume=pointer.exists(), max_tasks=TASKS - committed,
                    config_file=str(config_path))
            state = read_state(run_dir)
            if state["committed_task"] != TASKS:
                raise RuntimeError(f"Training stopped before fixed budget: {name}")
            model = run_dir / f"committed/task_{TASKS:06d}/model.zip"
            verify_task(model.parent, TASKS, json.loads((run_dir / "config.json").read_text()))
            model_paths[name] = {"path": str(model), "sha256": sha256(model)}
            atomic_json(RUN / "model_manifest.json", model_paths)
            operation(f"多种子训练 {name} 完成 {TASKS * 4096} 样本，事务 checkpoint 校验通过",
                      str(model.relative_to(ROOT)))
    return model_paths


def neural_models(variant, manifest):
    if variant == "ppo":
        return {"ppo": PPO}
    if variant == "bc":
        return {"bc": BC}
    if variant == "B12":
        return {"ppo": PPO, "bc": BC}
    mode = variant.split("_", 1)[0]
    return {"ppo": PPO, mode: Path(manifest[variant]["path"])}


def neural_games(manifest, hashes):
    for variant in VARIANTS[:-1]:
        folder = RUN / "evaluation" / variant
        folder.mkdir(parents=True, exist_ok=True)
        pending = []
        for seed in EVAL_SEEDS:
            path = folder / f"seed_{seed}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if (row["variant"] != (variant.split("_", 1)[0] if variant not in ("ppo", "bc", "B12") else variant)
                        or row["seed"] != seed or row["cap"] != CAP
                        or row["model_sha256"] != hashes):
                    raise RuntimeError(f"Completed game protocol mismatch: {path}")
            else:
                pending.append(seed)
        if not pending:
            continue
        status("neural_evaluation", variant=variant, completed=len(EVAL_SEEDS)-len(pending),
               total=len(EVAL_SEEDS))
        paths = neural_models(variant, manifest)
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn"),
                                 initializer=init_worker, initargs=(paths,)) as pool:
            futures = {}
            for seed in pending:
                name = variant.split("_", 1)[0] if variant not in ("ppo", "bc", "B12") else variant
                args = (name, seed, CAP, hashes, True, True, None)
                futures[pool.submit(task, args)] = seed
            for future in as_completed(futures):
                seed = futures[future]
                atomic_json(folder / f"seed_{seed}.json", future.result())
                status("neural_evaluation", variant=variant,
                       completed=len(EVAL_SEEDS)-sum(not (folder / f"seed_{s}.json").exists()
                                                     for s in EVAL_SEEDS), total=len(EVAL_SEEDS))


def traditional_game(seed, code_hash):
    core = TetrisCore(seed)
    policy = V2(V2Config(mode="hold"))
    folder = RUN / "evaluation/traditional_v2_hold"
    checkpoint = folder / f"seed_{seed}.checkpoint.pkl"
    started = time.monotonic()
    pieces = 0
    clears = [0] * 5
    milestones = []
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["identity"] != (seed, CAP, code_hash):
            raise RuntimeError("Traditional checkpoint protocol mismatch")
        core, pieces = saved["core"], saved["pieces"]
        clears, milestones = saved["clears"], saved["milestones"]
        if not 0 < pieces < CAP or core.phase != "playing":
            raise RuntimeError("Traditional checkpoint invalid")
        started -= saved["elapsed_seconds"]
    while core.phase == "playing" and pieces < CAP:
        public = core.get_public_observation()
        placements = fast_placements(public)
        if pieces < 100 or pieces % 1000 == 0:
            if placements != core.get_legal_placements():
                raise RuntimeError("Traditional legal placements differ from Core")
        action = policy.choose(public, placements)
        if action not in {row["actionId"] for row in placements}:
            raise RuntimeError("Traditional action illegal")
        old_lines = core.lines
        core.get_legal_placements = lambda: placements
        try:
            core.step(action)
        finally:
            del core.get_legal_placements
        pieces += 1
        clears[core.lines - old_lines] += 1
        if pieces % 1000 == 0:
            milestones.append({"pieces": pieces, "score": core.score})
        if pieces % CHECKPOINT_INTERVAL == 0 and pieces < CAP and core.phase == "playing":
            atomic_pickle(checkpoint, {"identity": (seed, CAP, code_hash), "core": core,
                                       "pieces": pieces, "clears": clears,
                                       "milestones": milestones,
                                       "elapsed_seconds": time.monotonic() - started})
    return {"variant": "traditional_v2_hold", "seed": seed, "cap": CAP,
            "model_sha256": {"traditional_v2_code": code_hash}, "pieces": pieces,
            "game_over": core.phase == "over", "capped": core.phase != "over" and pieces == CAP,
            "score": core.score, "clears_0_to_4": clears, "score_milestones": milestones,
            "wall_seconds": time.monotonic() - started}


def traditional_task(seed, code_hash):
    import torch
    torch.set_num_threads(1)
    return traditional_game(seed, code_hash)


def traditional_games(code_hash):
    folder = RUN / "evaluation/traditional_v2_hold"
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed in EVAL_SEEDS:
        path = folder / f"seed_{seed}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if (row["seed"] != seed or row["cap"] != CAP or
                    row["model_sha256"] != {"traditional_v2_code": code_hash}):
                raise RuntimeError(f"Traditional completed game protocol mismatch: {path}")
        else:
            pending.append(seed)
    status("traditional_evaluation", completed=len(EVAL_SEEDS)-len(pending), total=len(EVAL_SEEDS))
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(traditional_task, seed, code_hash): seed for seed in pending}
        for future in as_completed(futures):
            seed = futures[future]
            atomic_json(folder / f"seed_{seed}.json", future.result())
            (folder / f"seed_{seed}.checkpoint.pkl").unlink(missing_ok=True)
            status("traditional_evaluation",
                   completed=len(EVAL_SEEDS)-sum(not (folder / f"seed_{s}.json").exists()
                                                 for s in EVAL_SEEDS), total=len(EVAL_SEEDS))


def summarize(hashes, code_hash):
    rows = {name: [json.loads((RUN / "evaluation" / name / f"seed_{seed}.json").read_text())
                   for seed in EVAL_SEEDS] for name in VARIANTS}
    summaries = {name: _summary(items) for name, items in rows.items()}
    paired = {}
    for name in VARIANTS[1:]:
        scores = [_common_score(a, b) for a, b in zip(rows[name], rows["ppo"])]
        valid = [pair for pair in scores if pair is not None]
        paired[name] = {"common_score_ratio_to_ppo":
                        sum(a for a, _ in valid) / max(1, sum(b for _, b in valid)),
                        "seed_survival_deltas": [a["pieces"]-b["pieces"]
                                                 for a, b in zip(rows[name], rows["ppo"])]}
    result = {"protocol": json.loads((RUN / "protocol.json").read_text()),
              "model_sha256": hashes, "traditional_v2_code_sha256": code_hash,
              "summary": summaries, "paired": paired, "completed_at": now()}
    baseline = summaries["ppo"]
    decisions = {}
    for name in VARIANTS[3:-1]:
        current = summaries[name]
        decisions[name] = {
            "four_drop_pp": 100 * (baseline["four_fraction"] - current["four_fraction"]),
            "common_score_ratio": paired[name]["common_score_ratio_to_ppo"],
            "p10_ratio": current["p10_pieces"] / max(1, baseline["p10_pieces"]),
            "mean_survival_gain": current["mean_pieces"] - baseline["mean_pieces"],
            "early_death_excess": current["early_under_10000"] - baseline["early_under_10000"],
        }
        d = decisions[name]
        d["pass"] = (d["four_drop_pp"] <= 3 and d["common_score_ratio"] >= .95
                     and d["p10_ratio"] >= .95 and d["mean_survival_gain"] >= 1000
                     and d["early_death_excess"] <= 0)
    result["decisions"] = decisions
    atomic_json(RUN / "summary.json", result)
    report = ["# PPO 30M 多训练种子跟进", "",
              "三组独立训练种子均从冻结 PPO 30M 开始；每组普通起点和危险课程各 503808 环境样本。",
              "同一组新的 16 个开发种子，每局上限 100000 块；触顶是右截断。传统 V2 Hold 是固定规则对照，不参与神经网络训练。", "",
              "| 策略 | 平均存活 | P10 | 触顶 | 平均得分（亿） | 四消占比 | 共同存活得分 / PPO | 原门槛 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in VARIANTS:
        s = summaries[name]
        ratio = "—" if name == "ppo" else f"{paired[name]['common_score_ratio_to_ppo']:.2f}"
        decision = ("通过" if decisions[name]["pass"] else "未通过") if name in decisions else "—"
        report.append(f"| {name} | {s['mean_pieces']:.0f} | {s['p10_pieces']:.0f} | "
                      f"{s['capped']}/{len(EVAL_SEEDS)} | {s['mean_score']/1e8:.2f} | "
                      f"{s['four_fraction']:.1%} | {ratio} | {decision} |")
    report += ["", "该开发组用于比较训练种子差异，不能作为独立确认。逐种子正式得分、消行与模型哈希保存在 runs/ppo30m_multiseed_followup/evaluation/。",
               "原开发门槛：四消占比下降不超过 3 个百分点、共同存活得分不少于 PPO 的 95%、P10 不少于 PPO 的 95%、平均存活至少增加 1000、不得新增 1 万块以下死亡。",
               "本轮完成固定三个训练种子后停止；通过开发门槛仍需另一组未使用的种子独立确认，不能在开发组上直接宣称泛化提升。", ""]
    path = ROOT / "reports/experiments/ppo30m_multiseed_followup.md"
    path.write_text("\n".join(report))
    return result


def main():
    try:
        manifest = freeze()
        wait_for_long_games()
        model_manifest = train_all(manifest)
        hashes = {"ppo": digest(PPO), "bc": digest(BC),
                  **{name: item["sha256"] for name, item in model_manifest.items()}}
        code_hash = sha256(ROOT / "training/traditional/v2.py")
        if code_hash != json.loads((RUN / "protocol.json").read_text())["traditional_v2_code_sha256"]:
            raise RuntimeError("Frozen traditional V2 code changed")
        neural_games(model_manifest, hashes)
        traditional_games(code_hash)
        summarize(hashes, code_hash)
        operation("PPO 30M 三训练种子、普通与危险课程、传统 V2 同种子对照完成",
                  "runs/ppo30m_multiseed_followup/summary.json")
        subprocess.run(["git", "add", "OPERATIONS.md",
                        "reports/experiments/ppo30m_multiseed_followup.md"], cwd=ROOT, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 1:
            subprocess.run(["git", "commit", "-m", "Report PPO30M multi-seed follow-up"],
                           cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        status("completed", commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip())
    except BaseException as error:
        status("failed", error=repr(error))
        operation(f"PPO 30M 多种子后台队列失败并停止：{repr(error)}",
                  "runs/ppo30m_multiseed_followup/status.json")
        raise


if __name__ == "__main__":
    main()
