"""Immutable training-facing V2 records from verified counterfactual branches."""
import gzip
import hashlib
import json

from training.evaluation.local_correction_active_v2 import V2
from training.imitation_dataset import digest
from training.train_ppo import atomic_bytes, atomic_json


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def normalize(branch):
    row, evidence = branch["row"], branch["evidence"]
    p, b = evidence["branches"]["ppo"], evidence["branches"]["bc"]
    result = {"seed": branch["seed"], "episode_id": f"seed_{branch['seed']}_episode_0",
              "step": branch["step"], "public_observation": row["public"],
              "state_features": row["state"], "legal_action_ids": row["legal_ids"],
              "legal_action_features": row["legal_features"],
              "ppo_action": row["ppo_action"],
              "ppo_action_probability": branch["ppo_action_probability"],
              "bc_action": row["bc_action"], "hold_changed": evidence["hold_changed"],
              "ppo_risk": row["ppo_risk"], "bc_risk": row["bc_risk"],
              "ppo_branch": p, "bc_branch": b,
              "observed_survival_gain": b["pieces"] - p["pieces"],
              "common_score_ratio": evidence["common_score_ratio"],
              "common_four_clears": evidence["common_four_clears"],
              "next_legal": evidence["next_legal"],
              "category": evidence["category"], "confidence_basis": {
                  "trigger": evidence["trigger"], "precondition": evidence["precondition"],
                  "continuation_ok": evidence["continuation_ok"]},
              "death_distance_offline_only": row["death_distance"],
              "source_shard_sha256": branch["source_sha256"],
              "source_state_sha256": row["state_sha256"],
              "source_row_sha256": branch["row_sha256"],
              "model_sha256": branch["models"], "rule_version": branch["version"],
              "selection_sha256": branch["selection_sha256"]}
    result["record_sha256"] = hashlib.sha256(_canonical(result).encode()).hexdigest()
    return result


def main():
    manifest = V2 / "dataset_manifest.json"
    if not manifest.exists():
        raise RuntimeError("Counterfactual manifest must complete first")
    source = json.loads(manifest.read_text())
    path = V2 / "dataset_rows.jsonl.gz"
    meta_path = V2 / "dataset_rows.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta["source_manifest_sha256"] != digest(manifest) or
                meta["dataset_sha256"] != digest(path) or
                meta["records"] != source["branch_count"]):
            raise RuntimeError("Immutable V2 materialization changed")
        return meta
    if path.exists():
        raise RuntimeError("Uncommitted V2 dataset rows already exist")
    rows = []
    for relative, expected in sorted(source["branch_hashes"].items()):
        branch_path = V2 / relative
        if digest(branch_path) != expected:
            raise RuntimeError(f"Branch changed: {relative}")
        rows.append(normalize(json.loads(branch_path.read_text())))
    if len(rows) != source["branch_count"]:
        raise RuntimeError("Branch count mismatch")
    payload = gzip.compress(("\n".join(_canonical(row) for row in rows) + "\n").encode(),
                            compresslevel=4, mtime=0)
    atomic_bytes(path, payload)
    meta = {"version": "local_correction_dataset_v2", "records": len(rows),
            "train_seeds": source["train_seeds"], "validation_seeds": source["validation_seeds"],
            "source_manifest_sha256": digest(manifest), "dataset_sha256": digest(path)}
    atomic_json(meta_path, meta)
    return meta


if __name__ == "__main__":
    main()
