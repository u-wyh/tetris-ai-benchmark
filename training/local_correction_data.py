"""Episode-isolated PPO30M states and bounded, same-state BC counterfactuals."""
import gzip
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, load_models, verify_models
from training.hybrid_policy import F, risk
from training.hybrid_v2_policy import materially_safer, next_visible_legal, obvious_danger
from training.imitation_dataset import digest
from training.train_ppo import ROOT, atomic_bytes, atomic_json

RUN = ROOT / "runs/ppo30m_local_correction"
TRAIN = list(range(730000, 730016))
VALIDATION = list(range(730100, 730104))
DEV = list(range(730200, 730232))
CONFIRM = list(range(730300, 730332))
CAP = 100000
HORIZON = 1000


def audit():
    verify_models()
    groups = {"train": TRAIN, "validation": VALIDATION, "development": DEV,
              "confirmation": CONFIRM}
    seeds = [s for values in groups.values() for s in values]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Local-correction seed groups overlap")
    for seed in seeds:
        for base in (ROOT / "runs", ROOT / "reports"):
            for suffix in ("json", "jsonl.gz"):
                for path in base.rglob(f"seed_{seed}.{suffix}"):
                    if not path.is_relative_to(RUN):
                        raise RuntimeError(f"Previously used seed {seed}: {path}")
    protocol = {"models": HASHES, "seeds": groups, "cap": CAP,
                "branch_horizon": HORIZON, "version": 1}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Frozen local-correction protocol changed")
    if not path.exists():
        atomic_json(path, protocol)
    return protocol


def _paths(seed):
    folder = RUN / "trajectories"
    return folder / f"seed_{seed}.jsonl.gz", folder / f"seed_{seed}.states.pkl", folder / f"seed_{seed}.json"


def capture(seed, models=None):
    shard, states_path, meta_path = _paths(seed)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta["seed"] != seed or meta["models"] != HASHES or
                digest(shard) != meta["shard_sha256"] or
                digest(states_path) != meta["states_sha256"]):
            raise RuntimeError(f"Invalid trajectory shard for {seed}")
        return meta
    models = models or load_models()
    shard.parent.mkdir(parents=True, exist_ok=True)
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    risky, normal, snapshots = [], [], {}
    sample_rng = random.Random(seed)
    risky_seen = normal_seen = 0
    risk_count = disagreement = four_clears = 0
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            step = env.pieces
            mask = env.action_masks()
            public = env.core.get_public_observation()
            placements = {p["actionId"]: p for p in env._placement_cache}
            p = int(models["ppo"].predict(obs, deterministic=True, action_masks=mask)[0])
            b = int(models["bc"].predict(obs, deterministic=True, action_masks=mask)[0])
            if not (mask[p] and mask[b] and p in placements and b in placements):
                raise RuntimeError("Frozen policy selected an illegal placement")
            _, pr = risk(public, obs["candidate"][p], placements[p])
            _, br = risk(public, obs["candidate"][b], placements[b])
            is_risky = (pr["lock_out"] or pr["central_top4"] or pr["height"] >= 12 or
                        pr["new_holes"] >= 2 or pr["height_growth"] >= 3)
            risk_count += bool(is_risky)
            disagreement += p != b
            risk_candidate = is_risky and p != b
            normal_candidate = not is_risky and step % 97 == seed % 97
            if risk_candidate:
                risky_seen += 1
                risk_slot = sample_rng.randrange(risky_seen)
            if normal_candidate:
                normal_seen += 1
                normal_slot = sample_rng.randrange(normal_seen)
            keep_risk = risk_candidate and risk_slot < 1000
            keep_normal = normal_candidate and normal_slot < 1000
            if keep_risk or keep_normal:
                ids = mask.nonzero()[0]
                row = {"seed": seed, "step": step, "public": public,
                       "state": obs["state"].tolist(),
                       "legal_ids": ids.tolist(),
                       "legal_features": obs["candidate"][ids].tolist(),
                       "ppo_action": p, "bc_action": b, "ppo_risk": pr, "bc_risk": br,
                       "category": "candidate" if keep_risk else "normal",
                       "score": env.core.score, "lines": env.core.lines,
                       "four_clears_before": four_clears,
                       "state_sha256": hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}
                if keep_risk:
                    if risk_slot < len(risky):
                        snapshots.pop(risky[risk_slot]["step"], None)
                        risky[risk_slot] = row
                    else:
                        risky.append(row)
                    snapshots[step] = env.get_state()
                else:
                    if normal_slot < len(normal):
                        normal[normal_slot] = row
                    else:
                        normal.append(row)
            obs, _, terminated, truncated, info = env.step(p)
            four_clears += info["cleared_lines"] == 4
            if terminated or truncated:
                break
        for row in risky + normal:
            row["death_distance"] = info["pieces"] - row["step"] if terminated else None
        # Retain diverse high-risk branches. The other candidate rows still document the pool.
        chosen = []
        for row in sorted(risky, key=lambda x: (-int(x["ppo_risk"]["lock_out"]),
                                                -x["ppo_risk"]["height"],
                                                -x["ppo_risk"]["central_top4"], x["step"])):
            if len(chosen) >= 12:
                break
            if all(abs(row["step"] - other) >= 100 for other in chosen):
                chosen.append(row["step"])
        chosen_states = {step: snapshots[step] for step in chosen}
        temporary = shard.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf8", compresslevel=3) as stream:
            for row in sorted(risky + normal, key=lambda x: x["step"]):
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        os.replace(temporary, shard)
        atomic_bytes(states_path, pickle.dumps(chosen_states, protocol=pickle.HIGHEST_PROTOCOL))
        meta = {"seed": seed, "models": HASHES, "pieces": info["pieces"],
                "game_over": bool(terminated), "score": info["score"],
                "risk_states_seen": risk_count, "policy_disagreements": disagreement,
                "risk_disagreements_seen": risky_seen, "normal_candidates_seen": normal_seen,
                "risky_candidates": len(risky), "normal_samples": len(normal),
                "branch_steps": chosen, "four_clears": four_clears,
                "shard_sha256": digest(shard), "states_sha256": digest(states_path)}
        atomic_json(meta_path, meta)
        return meta
    finally:
        env.close()


def _branch(models, snapshot, action):
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    scores, fours = [], []
    try:
        obs, _ = env.reset(seed=snapshot["episode_seed"])
        env.set_state(snapshot)
        if not env.action_masks()[action]:
            raise RuntimeError("Counterfactual action is illegal")
        start = env.pieces
        four = 0
        while env.pieces - start < HORIZON:
            selected = action if env.pieces == start else int(models["ppo"].predict(
                obs, deterministic=True, action_masks=env.action_masks())[0])
            if not env.action_masks()[selected]:
                raise RuntimeError("Counterfactual continuation selected illegal action")
            obs, _, done, capped, info = env.step(selected)
            four += info["cleared_lines"] == 4
            scores.append(info["score"])
            fours.append(four)
            if done or capped:
                break
        return {"pieces": len(scores), "game_over": bool(done),
                "horizon_censored": not (done or capped), "scores": scores, "fours": fours}
    finally:
        env.close()


def classify(row, snapshot, models):
    p, b = row["ppo_risk"], row["bc_risk"]
    precondition = obvious_danger(p) and materially_safer(p, b)
    from training.tetris_core.legal_placements import get_legal_placements
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=row["seed"])
        env.set_state(snapshot)
        public = env.core.get_public_observation()
        by_id = {x["actionId"]: x for x in get_legal_placements(public)}
        pp, pn = next_visible_legal(public, by_id[row["ppo_action"]])
        bp, bn = next_visible_legal(public, by_id[row["bc_action"]])
    finally:
        env.close()
    continuation_ok = bp == "playing" and (pp != "playing" or bn >= pn - 4)
    by_feature = dict(zip(row["legal_ids"], row["legal_features"]))
    hold_changed = (by_feature[row["ppo_action"]][F["hold"]] !=
                    by_feature[row["bc_action"]][F["hold"]])
    branches = {name: _branch(models, snapshot, row[f"{name}_action"])
                for name in ("ppo", "bc")}
    ps, bs = branches["ppo"], branches["bc"]
    common = min(ps["pieces"], bs["pieces"])
    score_ratio = ((bs["scores"][common - 1] - row["score"]) /
                   max(1, ps["scores"][common - 1] - row["score"]))
    four_p = ps["fours"][common - 1]
    four_b = bs["fours"][common - 1]
    reliable = (precondition and continuation_ok and ps["game_over"] and
                bs["pieces"] >= ps["pieces"] + 250 and score_ratio >= .8 and
                four_b >= .75 * four_p)
    for value in branches.values():
        del value["scores"], value["fours"]
    return {"category": "A" if reliable else "B", "precondition": precondition,
            "continuation_ok": continuation_ok, "hold_changed": hold_changed,
            "next_legal": {"ppo": pn, "bc": bn},
            "common_pieces": common, "common_score_ratio": score_ratio,
            "common_four_clears": {"ppo": four_p, "bc": four_b},
            "branches": branches}


def filter_seed(seed, models=None):
    shard, states_path, meta_path = _paths(seed)
    meta = json.loads(meta_path.read_text())
    if digest(shard) != meta["shard_sha256"] or digest(states_path) != meta["states_sha256"]:
        raise RuntimeError("Trajectory integrity check failed")
    output = RUN / "labels" / f"seed_{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["source_sha256"] != meta["shard_sha256"]:
            raise RuntimeError("Label source changed")
        return saved
    models = models or load_models()
    with gzip.open(shard, "rt", encoding="utf8") as stream:
        rows = {row["step"]: row for row in map(json.loads, stream)}
    states = pickle.loads(states_path.read_bytes())
    labels = {}
    for step, snapshot in states.items():
        labels[str(step)] = classify(rows[step], snapshot, models)
    result = {"seed": seed, "source_sha256": meta["shard_sha256"],
              "horizon": HORIZON, "labels": labels,
              "A": sum(v["category"] == "A" for v in labels.values()),
              "B": sum(v["category"] == "B" for v in labels.values()),
              "C": meta["normal_samples"]}
    atomic_json(output, result)
    return result
