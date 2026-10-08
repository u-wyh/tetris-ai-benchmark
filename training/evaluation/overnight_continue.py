"""Continue the overnight queue after the already-running legacy worker exits."""
import json, subprocess, time
from pathlib import Path
from training.train_ppo import ROOT, atomic_json
from training.evaluation.evaluate import evaluate_model

QUEUE = ROOT / "runs/ppo_overnight_height_reward"
SEEDS = json.loads((ROOT / "training/evaluation/seeds.json").read_text())["validation"]
MODELS = {
    "raw_2m": ROOT / "runs/ppo_raw_10m_seed42/milestones/step_002002944/model.zip",
    "shaped_v1_2m": ROOT / "runs/ppo_hole_v1_2m_seed42_lambda010/milestones/step_002002944/model.zip",
    "shaped_v2_2m": ROOT / "runs/ppo_hole_v2_2m_seed42_lambda002/milestones/step_002002944/model.zip",
    "v3_2m": ROOT / "runs/ppo_hole_height_v3_2m_seed42/milestones/step_002002944/model.zip",
    "v4_2m": ROOT / "runs/ppo_height_v4_2m_seed42/milestones/step_002002944/model.zip",
}

def evaluate_missing():
    results = {}
    existing = json.loads((QUEUE / "evaluation_existing.json").read_text()) if (QUEUE / "evaluation_existing.json").exists() else {}
    for name, model in MODELS.items():
        if name in existing:
            results[name] = existing[name]
        else:
            output = QUEUE / f"evaluation_{name}.json"
            if output.exists():
                results[name] = json.loads(output.read_text())
                continue
            data = evaluate_model(model, SEEDS, 5000, "overnight_validation", 2002944, diagnostics=True)
            atomic_json(output, data)
            results[name] = data
    return results

def main():
    state_path = QUEUE / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("stage") not in ("completed", "select_candidate"):
        raise RuntimeError(f"legacy queue has not reached selection: {state}")
    results = evaluate_missing()
    summary = {name: data["aggregate"] for name, data in results.items()}
    raw = summary["raw_2m"]["mean_pieces"]
    candidates = {name: data for name, data in summary.items() if name in ("v3_2m", "v4_2m")}
    eligible = {name: data for name, data in candidates.items()
                if data["mean_pieces"] > raw * 1.05 and data["game_over_rate"] < 1.0}
    decision = {"candidate": max(eligible, key=lambda n: eligible[n]["mean_pieces"]) if eligible else None,
                "eligible": eligible, "reason": "mean survival > Raw by 5% and non-total Game Over rate"
                if eligible else "No V3/V4 candidate met the conservative extension gate"}
    atomic_json(QUEUE / "selection.json", {"summary": summary, "decision": decision})
    state.update(stage="extended_training" if decision["candidate"] else "completed",
                 candidate=decision["candidate"], updated_at=time.time())
    atomic_json(state_path, state)
    if not decision["candidate"]:
        return
    coeffs = {"v3_2m": (0.02, 0.02), "v4_2m": (0.0, 0.02)}[decision["candidate"]]
    run = f"ppo_overnight_extended_{decision['candidate']}_10m_seed42"
    config_path = ROOT / "configs" / f"{run}.json"
    if not config_path.exists():
        atomic_json(config_path, {"target_total_steps": 10_000_000, "device": "cuda", "n_envs": 8,
            "n_steps": 512, "task_steps": 4096, "base_seed": 42,
            "hole_penalty_coef": coeffs[0], "height_penalty_coef": coeffs[1],
            "transaction_retention": 3, "milestone_interval": 1_000_000,
            "validation_interval": 250_000, "periodic_validation_seeds": 16,
            "periodic_validation_max_pieces": 5000, "milestone_validation_seeds": 32,
            "milestone_validation_max_pieces": 10000, "final_test_seeds": 100,
            "final_test_max_pieces": 50000, "step0_baseline": True})
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "training.train_vector_transaction",
                    "--run-dir", str(ROOT / "runs" / run), "--config-file", str(config_path)], check=True)
    state.update(stage="completed", updated_at=time.time()); atomic_json(state_path, state)

if __name__ == "__main__": main()
