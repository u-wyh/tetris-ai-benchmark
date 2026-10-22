"""Bounded retrospective terminal-state replay; exploratory labels only."""

import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.evaluation.hybrid_queue import PPO, digest, load_models, verify_models
from training.evaluation.ppo_fresh_counterfactual import branch
from training.evaluation.ppo_transition_branches import grade, options
from training.evaluation.ppo_transition_capture import (CAP, PPO_SHA256, RUN, public_hash,
                                                         snapshot_path)
from training.train_ppo import atomic_bytes, atomic_json, now

VERSION = "terminal_replay_v1"
SEEDS = (1530002, 1530003, 1530004)
OFFSETS = (1000, 500, 100)
HORIZON = 2000
MAX_BRANCHES = 45
MAX_SECONDS = 5400
OUT = RUN / "terminal_exploratory"


def identity():
    return {"version": VERSION, "seeds": list(SEEDS), "offsets": list(OFFSETS),
            "horizon": HORIZON, "max_branches": MAX_BRANCHES,
            "max_seconds": MAX_SECONDS, "ppo_sha256": PPO_SHA256,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "purpose": "post-hoc failure mechanism exploration; not independent validation"}


def capture_terminal(seed, model):
    original = json.loads((RUN / "episodes_corrected" / f"seed_{seed}.json").read_text())
    if not original["game_over"] or original["capped"]:
        raise RuntimeError(f"Expected true PPO death: {seed}")
    path = OUT / "states" / f"seed_{seed}.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row["death_step"] != original["pieces"] or row["death_score"] != original["score"]:
            raise RuntimeError(f"Terminal manifest mismatch: {seed}")
        for item in row["states"]:
            if digest(OUT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"Terminal snapshot changed: {seed}:{item['step']}")
        return row
    earlier = max((x for x in original["states"] if x["step"] < original["pieces"] - max(OFFSETS)),
                  key=lambda x: x["step"])
    source = snapshot_path(seed, earlier["step"])
    if digest(source) != earlier["snapshot_sha256"]:
        raise RuntimeError("Starting snapshot changed")
    snapshot = pickle.loads(gzip.decompress(source.read_bytes()))
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    targets = {original["pieces"] - offset for offset in OFFSETS}
    states = []
    try:
        env.reset(seed=seed)
        env.set_state(snapshot)
        while env.pieces < original["pieces"]:
            step = env.pieces
            public = env.core.get_public_observation()
            if step in targets:
                state = env.get_state()
                raw = gzip.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=3)
                relative = f"states/seed_{seed}_step_{step}.pkl.gz"
                atomic_bytes(OUT / relative, raw)
                states.append({"step": step, "offset_to_death": original["pieces"] - step,
                               "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                               "public_sha256": public_hash(public)})
            mask = env.action_masks()
            action = fast_action(model, env.get_observation(), mask)
            if not mask[action]:
                raise RuntimeError(f"Illegal replay action: {seed}:{step}")
            _, _, dead, capped, _ = env.step(action)
            if dead or capped:
                break
        if (env.pieces != original["pieces"] or not dead or capped
                or env.core.score != original["score"] or len(states) != len(OFFSETS)):
            raise RuntimeError(f"Original death did not replay exactly: {seed}")
        row = {"seed": seed, "death_step": env.pieces, "death_score": env.core.score,
               "source_step": earlier["step"], "source_sha256": earlier["snapshot_sha256"],
               "states": states, "completed_at": now()}
        atomic_json(path, row)
        return row
    finally:
        env.close()


def main():
    torch.set_num_threads(1)
    verify_models()
    if digest(PPO) != PPO_SHA256:
        raise RuntimeError("Frozen PPO changed")
    if json.loads((RUN / "branch_status.json").read_text())["stage"] != "complete":
        raise RuntimeError("Prospective branch study incomplete")
    for name in ("states", "candidates", "branches", "comparisons"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol = identity()
    frozen = OUT / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise RuntimeError("Terminal protocol changed")
    else:
        atomic_json(frozen, protocol)
    model = MaskablePPO.load(PPO, device="cpu")
    models = load_models()
    started = time.monotonic()
    count = 0
    for seed in SEEDS:
        manifest = capture_terminal(seed, model)
        for item in manifest["states"]:
            if time.monotonic() - started > MAX_SECONDS:
                atomic_json(OUT / "status.json", {"stage": "budget_stop", "branches": count, "at": now()})
                return
            state_path = OUT / item["path"]
            if digest(state_path) != item["sha256"]:
                raise RuntimeError("Terminal snapshot changed")
            snapshot = pickle.loads(gzip.decompress(state_path.read_bytes()))
            if public_hash(snapshot["core"].get_public_observation()) != item["public_sha256"]:
                raise RuntimeError("Terminal public state changed")
            choices = options(snapshot, models)
            choices_path = OUT / "candidates" / f"seed_{seed}_step_{item['step']}.json"
            candidate = {"seed": seed, "step": item["step"], "choices": choices,
                         "snapshot_sha256": item["sha256"]}
            if choices_path.exists():
                if json.loads(choices_path.read_text()) != candidate:
                    raise RuntimeError("Terminal candidate changed")
            else:
                atomic_json(choices_path, candidate)
            results = {}
            for choice in choices:
                if count >= MAX_BRANCHES:
                    atomic_json(OUT / "status.json", {"stage": "budget_stop", "branches": count, "at": now()})
                    return
                action = choice["action"]
                result_path = OUT / "branches" / f"seed_{seed}_step_{item['step']}_action_{action}.json"
                branch_id = {"seed": seed, "step": item["step"], "action": action,
                             "snapshot_sha256": item["sha256"], "candidate_sha256": digest(choices_path),
                             "horizon": HORIZON}
                if result_path.exists():
                    saved = json.loads(result_path.read_text())
                    if any(saved.get(k) != v for k, v in branch_id.items()):
                        raise RuntimeError("Terminal branch changed")
                else:
                    saved = {**branch_id, "outcome": branch(snapshot, action, models, HORIZON)}
                    atomic_json(result_path, saved)
                results[action] = saved["outcome"]
                count += 1
            base = results[choices[0]["action"]]
            if base["pieces"] != item["offset_to_death"] or not base["game_over"]:
                raise RuntimeError(f"Terminal PPO branch death mismatch: {seed}:{item['step']}")
            comparison = {"seed": seed, "step": item["step"], "offset_to_death": item["offset_to_death"],
                          "choices": choices, "ppo_pieces": base["pieces"],
                          "comparisons": {str(choice["action"]): grade(base, results[choice["action"]])
                                          for choice in choices[1:]}}
            path = OUT / "comparisons" / f"seed_{seed}_step_{item['step']}.json"
            if path.exists():
                if json.loads(path.read_text()) != comparison:
                    raise RuntimeError("Terminal comparison changed")
            else:
                atomic_json(path, comparison)
            atomic_json(OUT / "status.json", {"stage": "running", "states": len(list((OUT / "comparisons").glob("*.json"))),
                                                  "branches": count, "last_seed": seed, "last_step": item["step"], "at": now()})
            print(f"terminal seed={seed} step={item['step']} branches={count}", flush=True)
    files = {str(path.relative_to(OUT)): digest(path) for folder in ("states", "candidates", "branches", "comparisons")
             for path in sorted((OUT / folder).glob("*")) if path.is_file()}
    atomic_json(OUT / "integrity.json", {"files": files, "count": len(files)})
    atomic_json(OUT / "status.json", {"stage": "complete", "states": len(SEEDS) * len(OFFSETS),
                                      "branches": count, "at": now()})


if __name__ == "__main__":
    main()
