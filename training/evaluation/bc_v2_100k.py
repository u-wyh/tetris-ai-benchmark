"""Resumable same-seed 100k comparison of frozen BC against V1 and V2 Hold."""
import json
from pathlib import Path
import statistics
import traceback

from sb3_contrib import MaskablePPO
import torch

from training.evaluation.champion_pipeline import evaluate_seed
from training.imitation_dataset import digest
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/imitation_v2_hold_20m/bc_vs_traditional_100k"
MODEL = ROOT / "runs/imitation_v2_hold_20m/bc/epoch_03.zip"
REFERENCE = ROOT / "reports/traditional_longlife/validation10_100000_20261008"
SEEDS = list(range(100016, 100026))
CAP = 100000


def reference(label):
    folder = REFERENCE / label
    config = json.loads((folder / "config.json").read_text())
    if config["seeds"] != SEEDS or config["max_pieces"] != CAP:
        raise RuntimeError(f"{label} seed/cap mismatch")
    if label == "hold" and (config["strategy"] != "v2-1" or config["policy"]["mode"] != "hold"):
        raise RuntimeError("V2 reference is not Hold-only")
    rows = [json.loads((folder / f"seed_{seed}.json").read_text()) for seed in SEEDS]
    if any(r["seed"] != s or r["pieces_survived"] > CAP for r, s in zip(rows, SEEDS)):
        raise RuntimeError(f"{label} results invalid")
    return rows


def summary(rows):
    pieces = sorted(r["pieces_survived"] for r in rows)
    return {"mean_pieces": statistics.mean(pieces), "median_pieces": statistics.median(pieces),
            "minimum_pieces": pieces[0], "p10_pieces": pieces[0],
            "capped_count": sum(v == CAP for v in pieces),
            "mean_lines": statistics.mean(r["lines"] for r in rows)}


def run():
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True)
    v1, v2 = reference("v1"), reference("hold")
    config = {"model": str(MODEL), "model_sha256": digest(MODEL),
              "label": "20M PPO + BC epoch 3", "seeds": SEEDS,
              "max_pieces": CAP, "deterministic": True, "action_mask": True,
              "v1_reference": str(REFERENCE / "v1"), "v2_reference": str(REFERENCE / "hold")}
    path = RUN / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("100k comparison configuration changed")
    if not path.exists():
        atomic_json(path, config)
    model = MaskablePPO.load(str(MODEL), device="cpu")
    try:
        for seed in SEEDS:
            target = RUN / f"seed_{seed}.json"
            if target.exists():
                row = json.loads(target.read_text())
                if row["seed"] != seed or row["pieces_survived"] > CAP:
                    raise RuntimeError(f"Invalid saved BC row for {seed}")
            else:
                atomic_json(RUN / "status.json", {"state": "running", "seed": seed,
                                                      "completed_seeds": seed-SEEDS[0], "updated_at": now()})
                row = evaluate_seed(model, seed, CAP)
                atomic_json(target, row)
            print(f"seed={seed} pieces={row['pieces_survived']} lines={row['lines']}", flush=True)
        bc = [json.loads((RUN / f"seed_{seed}.json").read_text()) for seed in SEEDS]
        result = {"bc": summary(bc), "v1": summary(v1), "v2_hold": summary(v2),
                  "protocol": config, "completed_at": now()}
        atomic_json(RUN / "summary.json", result)
        atomic_json(RUN / "status.json", {"state": "completed", "completed_seeds": len(SEEDS),
                                              "updated_at": now()})
    except Exception:
        atomic_json(RUN / "status.json", {"state": "failed", "error": traceback.format_exc(),
                                              "updated_at": now()})
        raise


if __name__ == "__main__":
    run()
