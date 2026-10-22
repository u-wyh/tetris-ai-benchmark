"""Extend one prespecified Top-2 counterfactual per death seed to 10K."""

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import torch

from training.evaluation.hybrid_queue import PPO, digest, load_models, verify_models
from training.evaluation.ppo_fresh_counterfactual import branch
from training.evaluation.ppo_transition_branches import grade
from training.evaluation.ppo_transition_capture import PPO_SHA256, RUN, public_hash
from training.train_ppo import atomic_json, now

OUT = RUN / "terminal_exploratory"
SEEDS = (1530002, 1530003, 1530004)
OFFSET = 500
HORIZON = 10000
VERSION = "terminal_top2_extension_v1"


def main():
    torch.set_num_threads(1)
    verify_models()
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO changed")
    if json.loads((OUT / "status.json").read_text())["stage"] != "complete":
        raise RuntimeError("Terminal study incomplete")
    directory = OUT / "extended_top2"
    directory.mkdir(exist_ok=True)
    protocol = {"version": VERSION, "seeds": list(SEEDS), "offset": OFFSET,
                "horizon": HORIZON, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "selection": "one PPO Top-2 at 500 before death per seed; exploratory post-hoc"}
    frozen = directory / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Extension protocol changed")
    else:
        atomic_json(frozen, protocol)
    models = load_models()
    for seed in SEEDS:
        manifest = json.loads((OUT / "states" / f"seed_{seed}.json").read_text())
        item = next(x for x in manifest["states"] if x["offset_to_death"] == OFFSET)
        state_path = OUT / item["path"]
        if digest(state_path) != item["sha256"]:
            raise RuntimeError("Terminal snapshot changed")
        snapshot = pickle.loads(gzip.decompress(state_path.read_bytes()))
        if public_hash(snapshot["core"].get_public_observation()) != item["public_sha256"]:
            raise RuntimeError("Public state changed")
        comparison = json.loads((OUT / "comparisons" / f"seed_{seed}_step_{item['step']}.json").read_text())
        top2 = next(x for x in comparison["choices"] if "ppo_top2" in x["sources"])
        action = top2["action"]
        baseline_action = comparison["choices"][0]["action"]
        baseline_path = OUT / "branches" / f"seed_{seed}_step_{item['step']}_action_{baseline_action}.json"
        short_path = OUT / "branches" / f"seed_{seed}_step_{item['step']}_action_{action}.json"
        base = json.loads(baseline_path.read_text())["outcome"]
        short = json.loads(short_path.read_text())["outcome"]
        if base["pieces"] != OFFSET or not base["game_over"] or not short["censored"]:
            raise RuntimeError("Prespecified short branches invalid")
        path = directory / f"seed_{seed}.json"
        identity = {"version": VERSION, "seed": seed, "step": item["step"], "action": action,
                    "snapshot_sha256": item["sha256"], "short_sha256": digest(short_path),
                    "baseline_sha256": digest(baseline_path), "horizon": HORIZON}
        if path.exists():
            row = json.loads(path.read_text())
            if any(row.get(k) != v for k, v in identity.items()):
                raise RuntimeError("Extended result identity changed")
        else:
            result = branch(snapshot, action, models, HORIZON)
            if (result["scores"][:short["pieces"]] != short["scores"]
                    or result["fours"][:short["pieces"]] != short["fours"]):
                raise RuntimeError("Extended branch prefix disagrees with frozen short branch")
            row = {**identity, "outcome": result, "comparison": grade(base, result),
                   "completed_at": now()}
            atomic_json(path, row)
        print(f"seed={seed} pieces={row['outcome']['pieces']} dead={row['outcome']['game_over']}", flush=True)
    files = {p.name: digest(p) for p in sorted(directory.glob('seed_*.json'))}
    atomic_json(directory / "integrity.json", {"file_sha256": files, "count": len(files)})
    atomic_json(directory / "status.json", {"stage": "complete", "seeds": len(SEEDS), "at": now()})


if __name__ == "__main__":
    main()
