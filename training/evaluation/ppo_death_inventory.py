"""Inventory historical PPO30M outcomes without relabeling them as new validation."""
import json

from training.evaluation.hybrid_queue import HASHES, digest
from training.evaluation.ppo_death_capture import RUN
from training.train_ppo import ROOT, atomic_json


def build():
    entries = []
    local = ROOT / "runs/ppo30m_local_correction/trajectories"
    for path in sorted(local.glob("seed_*.json")):
        row = json.loads(path.read_text())
        seed = row["seed"]
        trace = local / f"seed_{seed}.jsonl.gz"
        states = local / f"seed_{seed}.states.pkl"
        if row["models"]["ppo"] != HASHES["ppo"] or digest(trace) != row["shard_sha256"] or digest(states) != row["states_sha256"]:
            raise RuntimeError(f"Historical local-correction source mismatch: {seed}")
        entries.append({"source": "prior_local_correction_train_or_validation",
                        "historical_role": "train" if seed < 730100 else "validation",
                        "seed": seed, "model_sha256": HASHES["ppo"], "pieces": row["pieces"],
                        "game_over": row["game_over"], "capped": row["pieces"] == 100000,
                        "score": row["score"], "actions": "sparse_sample_only",
                        "core_rng_snapshots": len(row["branch_steps"]),
                        "files": {str(path.relative_to(ROOT)): digest(path),
                                  str(trace.relative_to(ROOT)): row["shard_sha256"],
                                  str(states.relative_to(ROOT)): row["states_sha256"]}})
    for role in ("tune_100k", "confirm_100k"):
        folder = ROOT / "runs/candidate_ppo30m_bc_hybrid" / role / "ppo"
        for path in sorted(folder.glob("seed_*.json")):
            row = json.loads(path.read_text())
            if row["model_sha256"]["ppo"] != HASHES["ppo"]:
                raise RuntimeError(f"Historical hybrid source model mismatch: {path}")
            entries.append({"source": "prior_hybrid_ppo_baseline", "historical_role": role,
                            "seed": row["seed"], "model_sha256": HASHES["ppo"],
                            "pieces": row["pieces_survived"], "game_over": row["game_over"],
                            "capped": row["survived_cap"], "score": row["score"],
                            "actions": "summary_only", "core_rng_snapshots": 0,
                            "files": {str(path.relative_to(ROOT)): digest(path)}})
    folder = ROOT / "runs/ppo30m_lookahead_v3/development_50k/ppo"
    for path in sorted(folder.glob("seed_*.json.gz")):
        from training.evaluation.ppo_lookahead_v3_queue import read_gzip
        row = read_gzip(path)
        if row["model_sha256"]["ppo"] != HASHES["ppo"]:
            raise RuntimeError(f"Historical V3 source model mismatch: {path}")
        entries.append({"source": "prior_v3_ppo_baseline", "historical_role": "development_50k",
                        "seed": row["seed"], "model_sha256": HASHES["ppo"],
                        "pieces": row["pieces"], "game_over": row["game_over"],
                        "capped": row["capped"], "score": row["score"],
                        "actions": "milestones_only", "core_rng_snapshots": 0,
                        "files": {str(path.relative_to(ROOT)): digest(path)}})
    inventory = {"ppo_sha256": HASHES["ppo"], "entries": entries,
                 "meaning": "Historical exploration inventory; no entry is an independent new validation episode"}
    atomic_json(RUN / "historical_inventory.json", inventory)
    return inventory


if __name__ == "__main__":
    result = build()
    print(len(result["entries"]), sum(x["game_over"] for x in result["entries"]), flush=True)
