"""Recoverable overnight evaluation and sequential V3/V4 training queue."""
import json, subprocess, time
from pathlib import Path
from training.train_ppo import ROOT, atomic_json
from training.evaluation.evaluate import evaluate_model

QUEUE = ROOT / "runs/ppo_overnight_height_reward"
STATE = QUEUE / "queue_state.json"
SEEDS = json.loads((ROOT / "training/evaluation/seeds.json").read_text())["validation"]
RUNS = {
    "train_v3_2m": ("ppo_hole_height_v3_2m_seed42", "configs/ppo_hole_height_v3_2m_seed42.json"),
    "train_v4_2m": ("ppo_height_v4_2m_seed42", "configs/ppo_height_v4_2m_seed42.json"),
}

def save(state):
    atomic_json(STATE, state)

def completed(run):
    p = ROOT / "runs" / run / "transaction_state.json"
    return p.exists() and json.loads(p.read_text()).get("status") == "completed"

def model_path(run):
    return ROOT / "runs" / run / "milestones/step_002002944/model.zip"

def evaluate_existing(state):
    out = QUEUE / "evaluation_existing.json"
    if out.exists(): return
    rows = {}
    models = {
        "raw_2m": ROOT / "runs/ppo_raw_10m_seed42/milestones/step_002002944/model.zip",
        "shaped_v1_2m": ROOT / "runs/ppo_hole_v1_2m_seed42_lambda010/milestones/step_002002944/model.zip",
        "shaped_v2_2m": ROOT / "runs/ppo_hole_v2_2m_seed42_lambda002/milestones/step_002002944/model.zip",
    }
    for name, path in models.items():
        rows[name] = evaluate_model(path, SEEDS, 5000, "overnight_validation", 2002944, diagnostics=True)
    atomic_json(out, rows)

def train(stage, run, config):
    if completed(run): return
    run_dir = ROOT / "runs" / run
    args = [str(ROOT / ".venv/bin/python"), "-m", "training.train_vector_transaction",
            "--run-dir", str(run_dir)]
    args += (["--resume"] if (run_dir / "config.json").exists() else ["--config-file", str(ROOT / config)])
    subprocess.run(args, check=True)

def main():
    QUEUE.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"stage":"evaluation_existing","started_at":time.time()}
    save(state)
    try:
        if state["stage"] == "evaluation_existing":
            evaluate_existing(state); state["stage"] = "train_v3_2m"; save(state)
        if state["stage"] == "train_v3_2m":
            train(*RUNS[state["stage"]]); state["stage"] = "evaluate_v3"; save(state)
        if state["stage"] == "evaluate_v3":
            result = evaluate_model(model_path("ppo_hole_height_v3_2m_seed42"), SEEDS, 5000, "overnight_validation", 2002944, diagnostics=True)
            atomic_json(QUEUE / "evaluation_v3.json", result)
            state["stage"] = "train_v4_2m"; save(state)
        if state["stage"] == "train_v4_2m":
            train(*RUNS[state["stage"]]); state["stage"] = "evaluate_v4"; save(state)
        if state["stage"] == "evaluate_v4":
            result = evaluate_model(model_path("ppo_height_v4_2m_seed42"), SEEDS, 5000, "overnight_validation", 2002944, diagnostics=True)
            atomic_json(QUEUE / "evaluation_v4.json", result)
            state["stage"] = "select_candidate"; save(state)
        if state["stage"] == "select_candidate":
            state["stage"] = "completed"; save(state)
    except Exception as error:
        state.update(stage="failed", error=repr(error)); save(state); raise

if __name__ == "__main__": main()
