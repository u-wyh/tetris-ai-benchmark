"""Independent recoverable queue for V2 Hold imitation and PPO fine-tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from training.imitation_dataset import digest
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/imitation_v2_hold_20m"
DATA = RUN / "data"
BC = RUN / "bc"
SOURCE20 = ROOT / "runs/ppo_candidate_raw_20m_seed42/milestones/step_020000768/model.zip"
SOURCE30 = ROOT / "runs/ppo_candidate_raw_30m_seed42/final/model.zip"
PURE30 = ROOT / "runs/ppo_candidate_raw_30m_seed42/transaction_state.json"
FINE = ROOT / "runs/imitation_v2_hold_20m/ppo_finetune_2m"
STAGES = ("verify_current_30m", "expert_dataset_smoke", "collect_expert_200k",
          "bc_smoke", "bc_training", "bc_validation", "ppo_finetune_2m",
          "paired_validation", "conditional_ppo_extension", "report")


def command(*args):
    subprocess.run([sys.executable, "-m", *map(str, args)], cwd=ROOT, check=True)


def stage(name, work):
    marker = RUN / f"{name}.done.json"
    if marker.exists():
        return
    atomic_json(RUN / "status.json", {"stage": name, "state": "running", "updated_at": now()})
    with (RUN / "events.log").open("a") as stream:
        stream.write(f"{now()} STARTED {name}\n")
    try:
        work()
        atomic_json(marker, {"stage": name, "completed_at": now()})
        with (RUN / "events.log").open("a") as stream:
            stream.write(f"{now()} COMPLETED {name}\n")
    except Exception:
        error = traceback.format_exc()
        atomic_json(RUN / "status.json", {"stage": name, "state": "failed",
                                          "updated_at": now(), "error": error})
        with (RUN / "events.log").open("a") as stream:
            stream.write(f"{now()} FAILED {name} {error}\n")
        raise


def verify_current_30m():
    while True:
        state = json.loads(PURE30.read_text())
        if state["status"] == "completed":
            if not SOURCE30.exists():
                raise RuntimeError("30M completed without final model")
            atomic_json(RUN / "pure_30m.json", {"steps": state["committed_steps"],
                                                  "model_sha256": digest(SOURCE30),
                                                  "source": str(SOURCE30)})
            return
        if state["status"] in ("failed", "paused"):
            raise RuntimeError(f"Pure PPO 30M stopped: {state['status']}")
        time.sleep(60)


def collect():
    command("training.imitation_dataset", "--output", DATA, "--first-seed", 600000,
            "--seeds", 108, "--limit", 2000)
    total = sum(json.loads(p.read_text())["samples"] for p in DATA.glob("seed_*.json"))
    if total < 200000:
        raise RuntimeError(f"Teacher trajectories contain only {total} samples")


def bc_train(smoke):
    output = RUN / "bc_smoke" if smoke else BC
    train_seeds = list(range(600000, 600004)) if smoke else list(range(600000, 600100))
    val_seeds = [600100] if smoke else list(range(600100, 600108))
    command("training.imitation_bc", "--dataset", DATA, "--output", output,
            "--source", SOURCE20, "--train-seeds", *train_seeds,
            "--val-seeds", *val_seeds, "--epochs", 3, "--device", "cpu" if smoke else "cuda",
            *(["--max-train", "2048", "--max-val", "512"] if smoke else []))
    history = json.loads((output / "history.json").read_text())
    if smoke and not (history[-1]["train"]["loss"] < history[0]["train"]["loss"]
                      and history[-1]["validation"]["top1"] >= history[0]["validation"]["top1"]):
        raise RuntimeError("BC smoke did not improve train loss and validation Top-1")
    if smoke:
        from sb3_contrib import MaskablePPO
        from training.evaluation.champion_pipeline import evaluate_seed
        model = MaskablePPO.load(str(output / "epoch_03.zip"), device="cpu")
        atomic_json(output / "game_smoke.json", evaluate_seed(model, 100000, 20000))


def evaluate_model(name, model_path):
    from sb3_contrib import MaskablePPO
    from training.evaluation.champion_pipeline import evaluate_seed
    folder = RUN / "validation" / name
    folder.mkdir(parents=True, exist_ok=True)
    model = MaskablePPO.load(str(model_path), device="cpu")
    for seed in range(100000, 100032):
        path = folder / f"seed_{seed}.json"
        if not path.exists():
            atomic_json(path, evaluate_seed(model, seed, 20000))


def prepare_finetune():
    source = BC / "epoch_03.zip"
    if not source.exists():
        raise FileNotFoundError(source)
    config = json.loads((ROOT / "configs/ppo_candidate_raw_2m_seed42.json").read_text())
    config.update({"base_seed": 142, "initial_model": {"path": str(source.resolve()),
                   "sha256": digest(source), "actor": "20M PPO + BC epoch 3",
                   "critic": "20M PPO, unchanged by actor-only BC",
                   "optimizer": "fresh", "ppo_steps_before_run": 0},
                   "step0_baseline": False})
    path = RUN / "finetune_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Fine-tune config changed")
    if not path.exists():
        atomic_json(path, config)
    return path


def finetune():
    config = prepare_finetune()
    command("training.train_vector_transaction", "--run-dir", FINE,
            *(["--resume"] if (FINE / "transaction_state.json").exists()
              else ["--config-file", config]))
    state = json.loads((FINE / "transaction_state.json").read_text())
    if state["status"] != "completed":
        raise RuntimeError("Fine-tune stopped before 2M")


def summarize(name):
    from training.evaluation.champion_pipeline import percentile
    rows = [json.loads(p.read_text()) for p in sorted((RUN / "validation" / name).glob("seed_*.json"))]
    if len(rows) != 32:
        raise RuntimeError(f"Incomplete validation {name}")
    pieces = [r["pieces_survived"] for r in rows]
    return {"mean_pieces": sum(pieces)/32, "p10_pieces": percentile(pieces, .1),
            "median_pieces": (sorted(pieces)[15] + sorted(pieces)[16])/2,
            "cap_count": sum(v == 20000 for v in pieces),
            "mean_lines": sum(r["lines"] for r in rows)/32}


def evaluate_teacher():
    from training.evaluation.traditional_v2 import play
    from training.traditional.v2 import V2Config
    folder = RUN / "validation" / "traditional_v2_hold"
    folder.mkdir(parents=True, exist_ok=True)
    for seed in range(100000, 100032):
        path = folder / f"seed_{seed}.json"
        if not path.exists():
            row = play(seed, 20000, V2Config(mode="hold"))
            row.pop("actions")
            row.pop("decision_ms")
            atomic_json(path, row)


def paired():
    evaluate_model("pure_30m", SOURCE30)
    evaluate_model("bc_ppo_2m", FINE / "final/model.zip")
    evaluate_teacher()
    results = {name: summarize(name) for name in
               ("pure_20m", "pure_30m", "bc", "bc_ppo_2m", "traditional_v2_hold")}
    atomic_json(RUN / "validation_summary.json", results)


def extension_decision():
    results = json.loads((RUN / "validation_summary.json").read_text())
    fine, baseline = results["bc_ppo_2m"], results["pure_30m"]
    improve = (fine["mean_pieces"] > baseline["mean_pieces"] and
               fine["p10_pieces"] >= baseline["p10_pieces"] and
               fine["cap_count"] >= baseline["cap_count"])
    atomic_json(RUN / "extension_decision.json", {"extend_to_10m": improve,
                "rule": "higher paired validation mean, nondecreasing P10 and cap count",
                "decided_at": now()})
    if improve:
        from training.extend_candidate import prepare
        destination = RUN / "ppo_finetune_10m"
        if not destination.exists():
            prepare(FINE, destination, 10_000_000, evaluate_baseline=False)
        command("training.train_vector_transaction", "--run-dir", destination, "--resume")


def report():
    summary = json.loads((RUN / "validation_summary.json").read_text())
    counts = [json.loads(p.read_text()) for p in DATA.glob("seed_*.json")]
    bc = json.loads((BC / "history.json").read_text())
    decision = json.loads((RUN / "extension_decision.json").read_text())
    lines = ["# Candidate imitation learning: V2 Hold → BC → PPO", "",
             f"Generated: {now()}. Teacher: V2 Hold-only. Expert samples: {sum(x['samples'] for x in counts)}.",
             f"Teacher config SHA256: {counts[0]['teacher_hash']}. Shards: {len(counts)}.",
             f"Hold labels: {sum(x['hold_actions'] for x in counts)}; danger height ≥12: {sum(x['danger_height_ge_12'] for x in counts)}.",
             "Train/BC validation split by complete episode seed. Official final-test seeds excluded.",
             f"BC validation Top-1: {bc[0]['validation']['top1']:.3f} → {bc[-1]['validation']['top1']:.3f}.",
             f"BC validation teacher probability: {bc[0]['validation']['teacher_probability']:.3f} → {bc[-1]['validation']['teacher_probability']:.3f}.",
             "", "| Model | Mean pieces | P10 | Cap/32 | Mean lines |", "|---|---:|---:|---:|---:|"]
    for name, row in summary.items():
        lines.append(f"| {name} | {row['mean_pieces']:.1f} | {row['p10_pieces']} | {row['cap_count']} | {row['mean_lines']:.1f} |")
    lines += ["", f"Automatic 10M extension: {decision['extend_to_10m']}.",
              "All gameplay comparisons, including V2 Hold, use the same 32 validation seeds and 20,000-piece cap.",
              "50M/100M extension is not authorized by this experiment."]
    path = ROOT / "reports/experiments/candidate_imitation_learning.md"
    path.write_text("\n".join(lines)+"\n")


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    stage(STAGES[0], verify_current_30m)
    stage(STAGES[1], lambda: command("training.imitation_dataset", "--output",
          RUN / "smoke_data", "--first-seed", 500000, "--seeds", 2, "--limit", 128))
    stage(STAGES[2], collect)
    stage(STAGES[3], lambda: bc_train(True))
    stage(STAGES[4], lambda: bc_train(False))
    stage(STAGES[5], lambda: (evaluate_model("pure_20m", SOURCE20),
                             evaluate_model("bc", BC / "epoch_03.zip")))
    stage(STAGES[6], finetune)
    stage(STAGES[7], paired)
    stage(STAGES[8], extension_decision)
    stage(STAGES[9], report)
    atomic_json(RUN / "status.json", {"stage": "completed", "state": "completed",
                                          "updated_at": now()})


if __name__ == "__main__":
    run()
