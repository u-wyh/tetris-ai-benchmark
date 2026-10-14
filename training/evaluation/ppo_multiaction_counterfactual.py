"""Frozen, resumable PPO30M multi-action counterfactual pilot."""
import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, load_models, verify_models
from training.evaluation.local_correction_active_v2 import V2, _row_hash
from training.evaluation.local_correction_hold_audit import _well
from training.hybrid_policy import F, risk
from training.hybrid_v2_policy import next_visible_legal
from training.imitation_bc import batch
from training.imitation_dataset import digest
from training.local_correction_data import CAP, TRAIN, _paths
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/ppo30m_multiaction_counterfactual"
VERSION = "pilot_v1_50_states_top5_safe_well_256_then_1000"
LIMIT = 1000
SHORT = 256
SEED_LIMIT = 5
STATE_TARGET = 50


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tier(row):
    d = row["death_distance"]
    if d is None or d > 1000:
        return "control"
    if d <= 250:
        return "0-250"
    return "251-500" if d <= 500 else "501-1000"


def freeze_selection():
    """Select by existing trajectory metadata, before any new branch outcome."""
    protocol = {"version": VERSION, "models": HASHES, "training_seeds": TRAIN,
                "target_states": STATE_TARGET, "max_per_seed": SEED_LIMIT,
                "tiers": ["0-250", "251-500", "501-1000", "control"],
                "short_horizon": SHORT, "max_horizon": LIMIT,
                "extension": "all non-baseline candidates alive at 256 with score >= 75pct of PPO at common 256; all PPO baselines to 1000",
                "A": "PPO dies before 1000; candidate survives >= PPO+250 and >=90pct common score and >=90pct common four clears (if PPO four>=2); same public post-action current/hold/next; risk no worse and well not lost >1; >=100 common pieces",
                "B": "survival gain but A evidence incomplete; otherwise C"}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Frozen protocol changed")
    if not path.exists():
        atomic_json(path, protocol)
    pool = []
    for seed in TRAIN:
        shard, _, meta_path = _paths(seed)
        meta = json.loads(meta_path.read_text())
        if meta["models"] != HASHES or digest(shard) != meta["shard_sha256"]:
            raise RuntimeError(f"Source shard mismatch: {seed}")
        snapshot_meta = json.loads((V2 / "snapshots" / f"seed_{seed}.json").read_text())
        snapshot_path = V2 / "snapshots" / f"seed_{seed}.pkl"
        if snapshot_meta["source_sha256"] != meta["shard_sha256"] or digest(snapshot_path) != snapshot_meta["snapshots_sha256"]:
            raise RuntimeError(f"Snapshot mismatch: {seed}")
        for file in (V2 / "branches" / f"seed_{seed}").glob("step_*.json"):
            record = json.loads(file.read_text())
            row = record["row"]
            if record["source_sha256"] != meta["shard_sha256"] or record["row_sha256"] != _row_hash(row):
                raise RuntimeError(f"Existing branch mismatch: {file}")
            pool.append({"seed": seed, "step": row["step"], "tier": _tier(row),
                         "source_sha256": meta["shard_sha256"], "row_sha256": record["row_sha256"],
                         "state_sha256": row["state_sha256"], "snapshot_sha256": snapshot_meta["snapshots_sha256"],
                         "height": row["ppo_risk"]["height"], "growth": row["ppo_risk"]["height_growth"],
                         "central": row["ppo_risk"]["central_top4"]})
    selected, counts, per_seed, seen = [], Counter(), Counter(), set()
    quotas = {"0-250": 16, "251-500": 12, "501-1000": 14, "control": 8}
    for tier, quota in quotas.items():
        tier_pool = [x for x in pool if x["tier"] == tier]
        tier_pool.sort(key=lambda x: (per_seed[x["seed"]], -x["central"], -x["growth"], -x["height"], x["seed"], x["step"]))
        while counts[tier] < quota:
            eligible = [x for x in tier_pool if per_seed[x["seed"]] < SEED_LIMIT and
                        all(x["seed"] != y["seed"] or abs(x["step"]-y["step"]) >= 50 for y in selected) and
                        (x["seed"], x["state_sha256"]) not in seen]
            if not eligible:
                break
            eligible.sort(key=lambda x: (per_seed[x["seed"]], -x["central"], -x["growth"], -x["height"], x["seed"], x["step"]))
            x = eligible[0]
            selected.append(x); per_seed[x["seed"]] += 1; counts[tier] += 1
            seen.add((x["seed"], x["state_sha256"]))
    if len(selected) < 40:
        raise RuntimeError(f"Only {len(selected)} diverse states available")
    selection = {"version": VERSION, "protocol_sha256": digest(path), "states": sorted(selected, key=lambda x:(x["seed"],x["step"]))}
    out = RUN / "selection.json"
    if out.exists() and json.loads(out.read_text()) != selection:
        raise RuntimeError("Frozen pilot selection changed")
    if not out.exists():
        atomic_json(out, selection)
    return selection


def legal_rank(row, model):
    observation, mask, _ = batch([{**row, "action": row["ppo_action"]}], "cpu")
    with torch.no_grad():
        probs = model.policy.get_distribution(observation, action_masks=mask).distribution.probs[0].cpu().numpy()
    ids = np.asarray(row["legal_ids"], dtype=int)
    if not np.all(mask.cpu().numpy()[0, ids]) or np.any(probs[np.setdiff1d(np.arange(len(probs)), ids)] > 1e-7):
        raise RuntimeError("PPO distribution leaks outside official mask")
    ranked = sorted(ids.tolist(), key=lambda a: (-float(probs[a]), a))
    if ranked[0] != row["ppo_action"]:
        raise RuntimeError("PPO Top-1 differs from archived deterministic action")
    return ranked, probs


def candidates(row, model):
    public = row["public"]
    placements = {p["actionId"]: p for p in get_legal_placements(public)}
    if set(placements) != set(row["legal_ids"]):
        raise RuntimeError("Official legal action set changed")
    ranked, probs = legal_rank(row, model)
    features = dict(zip(row["legal_ids"], row["legal_features"]))
    data = {}
    for rank, action in enumerate(ranked, 1):
        placement = placements[action]
        score, danger = risk(public, features[action], placement)
        state, _ = simulate_public(public, placement)
        well = _well(state["board"])
        data[action] = {"action": action, "hold": bool(placement["hold"]),
                        "ppo_probability": float(probs[action]), "ppo_rank": rank,
                        "risk_score": score, "risk": danger, "well_after": well,
                        "sources": []}
    chosen = []
    def add(action, source):
        data[action]["sources"].append(source)
        if action not in chosen and len(chosen) < 8:
            chosen.append(action)
    add(row["ppo_action"], "ppo_top1")
    add(row["bc_action"], "bc")
    for action in ranked[:5]: add(action, "ppo_top5")
    safe = sorted(ranked, key=lambda a:(data[a]["risk_score"], -data[a]["well_after"], data[a]["ppo_rank"]))
    add(safe[0], "public_safe")
    well = sorted(ranked, key=lambda a:(-data[a]["well_after"], data[a]["risk_score"], data[a]["ppo_rank"]))
    add(well[0], "well_preserving")
    for action in safe:
        if len(chosen) >= 8: break
        if action not in chosen and (data[action]["risk_score"] < data[row["ppo_action"]]["risk_score"] or
                                     data[action]["well_after"] >= data[row["ppo_action"]]["well_after"]):
            add(action, "public_diverse")
    for action in chosen:
        phase, count = next_visible_legal(public, placements[action])
        data[action]["next_phase"] = phase
        data[action]["next_legal"] = count
        state, _ = simulate_public(public, placements[action])
        data[action]["after_pieces"] = {"current": state["currentPiece"]["type"] if state["currentPiece"] else None,
                                        "hold": state["hold"], "next": state["next"]}
    return [data[a] for a in chosen]


def branch(snapshot, action, models, horizon=LIMIT):
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    start_time = time.monotonic()
    try:
        obs, _ = env.reset(seed=snapshot["episode_seed"])
        env.set_state(snapshot)
        if not env.action_masks()[action]: raise RuntimeError("Illegal branch action")
        start, initial_score = env.pieces, env.core.score
        scores, fours, risk_trace, clear_events, piece_trace = [], [], [], 0, []
        done = capped = False
        while env.pieces-start < horizon:
            chosen = action if env.pieces == start else int(models["ppo"].predict(obs, deterministic=True, action_masks=env.action_masks())[0])
            if not env.action_masks()[chosen]: raise RuntimeError("Illegal continuation")
            obs, _, done, capped, info = env.step(chosen)
            clear_events += int(info["cleared_lines"] > 0)
            fours.append((fours[-1] if fours else 0) + int(info["cleared_lines"] == 4))
            scores.append(info["score"]-initial_score)
            if len(scores) == 1 or len(scores)%64 == 0 or done or capped:
                public = env.core.get_public_observation()
                board = public["board"]
                heights = [next((20-y for y in range(20) if board[y][x] is not None),0) for x in range(10)]
                risk_trace.append({"piece": len(scores), "height": max(heights),
                                   "central_top4": sum(board[y][x] is not None for y in range(4) for x in range(3,7))})
            if len(scores) <= 4:
                public = env.core.get_public_observation()
                piece_trace.append({"current": public["currentPiece"]["type"] if public["currentPiece"] else None,
                                    "hold": public["hold"], "next": public["next"]})
            if done or capped: break
        return {"action": action, "pieces": len(scores), "game_over": bool(done),
                "censored": not (done or capped), "horizon": horizon,
                "scores": scores, "fours": fours, "clear_events": clear_events,
                "risk_trace": risk_trace, "piece_trace": piece_trace,
                "elapsed_seconds": time.monotonic()-start_time}
    finally:
        env.close()


def compare(base, other):
    common = min(base["pieces"], other["pieces"])
    if not common: raise RuntimeError("Empty branch")
    bp, op = base["scores"][common-1], other["scores"][common-1]
    bf, of = base["fours"][common-1], other["fours"][common-1]
    return {"common_pieces": common, "ppo_score": bp, "candidate_score": op,
            "score_ratio": op/max(1,bp), "ppo_fours": bf, "candidate_fours": of,
            "survival_delta": other["pieces"]-base["pieces"]}


def grade(base, other, p, candidate):
    metric = compare(base, other)
    future_same = candidate["after_pieces"] == p["after_pieces"]
    danger = p["risk"]["lock_out"] or p["risk"]["central_top4"] >= 2 or p["risk"]["height"] >= 16 or p["risk"]["height_growth"] >= 3
    four_ok = metric["candidate_fours"] >= .9*metric["ppo_fours"] if metric["ppo_fours"] >= 2 else metric["candidate_fours"] >= metric["ppo_fours"]
    certified = (danger and base["game_over"] and not other["game_over"] or
                 danger and base["game_over"] and other["pieces"] >= base["pieces"]+250)
    qualified = (certified and metric["survival_delta"] >= 250 and metric["common_pieces"] >= 100 and
                 metric["score_ratio"] >= .9 and four_ok and future_same and
                 candidate["risk_score"] <= p["risk_score"] and candidate["well_after"] >= p["well_after"]-1)
    category = "A" if qualified else "B" if metric["survival_delta"] > 0 else "C"
    return {**metric, "category": category, "danger": bool(danger), "four_ok": bool(four_ok),
            "future_same": future_same, "well_preserved": candidate["well_after"] >= p["well_after"]-1}


def _verified(path, identity):
    if not path.exists(): return None
    record = json.loads(path.read_text())
    if any(record.get(key) != value for key,value in identity.items()):
        raise RuntimeError(f"Resume identity mismatch: {path}")
    return record


def run_state(item, models, snapshots):
    seed, step = item["seed"], item["step"]
    source = V2 / "branches" / f"seed_{seed}" / f"step_{step}.json"
    record = json.loads(source.read_text())
    row = record["row"]
    if _row_hash(row) != item["row_sha256"] or row["state_sha256"] != item["state_sha256"]:
        raise RuntimeError("Source row mismatch")
    snapshot = snapshots[step]
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=seed); env.set_state(snapshot)
        public = env.core.get_public_observation()
        if hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest() != item["state_sha256"]:
            raise RuntimeError("Snapshot public state mismatch")
    finally: env.close()
    options = candidates(row, models["ppo"])
    option_path = RUN / "candidates" / f"seed_{seed}_step_{step}.json"
    option_id = {"version": VERSION, "models": HASHES, "state_sha256": item["state_sha256"],
                 "selection_sha256": digest(RUN/"selection.json"), "options": options}
    if option_path.exists() and json.loads(option_path.read_text()) != option_id:
        raise RuntimeError("Candidate list changed on resume")
    if not option_path.exists(): atomic_json(option_path, option_id)
    branches = {}
    p = options[0]
    for option in options:
        action = option["action"]
        path = RUN / "branches" / f"seed_{seed}_step_{step}_action_{action}.json"
        identity = {"version": VERSION, "models": HASHES, "seed": seed, "step": step,
                    "action": action, "state_sha256": item["state_sha256"],
                    "selection_sha256": digest(RUN/"selection.json"),
                    "candidate_sha256": digest(option_path)}
        saved = _verified(path, identity)
        if saved:
            branches[action] = saved["branch"]
            continue
        if action == p["action"]:
            outcome = branch(snapshot, action, models, LIMIT)
            old = record["evidence"]["branches"]["ppo"]
            if outcome["pieces"] != old["pieces"] or outcome["game_over"] != old["game_over"]:
                raise RuntimeError(f"PPO baseline diverged: {seed}:{step}")
        else:
            outcome = branch(snapshot, action, models, SHORT)
            base = branches[p["action"]]
            common = min(base["pieces"], outcome["pieces"])
            if outcome["censored"] and outcome["scores"][common-1] >= .75*base["scores"][common-1]:
                outcome = branch(snapshot, action, models, LIMIT)
        atomic_json(path, {**identity, "branch": outcome, "completed_at": now()})
        branches[action] = outcome
    result_path = RUN / "states" / f"seed_{seed}_step_{step}.json"
    grades = {str(o["action"]): grade(branches[p["action"]], branches[o["action"]], p, o)
              for o in options if o["action"] != p["action"]}
    result = {"version": VERSION, "models": HASHES, "seed": seed, "step": step,
              "tier": item["tier"], "state_sha256": item["state_sha256"],
              "selection_sha256": digest(RUN/"selection.json"), "candidate_sha256": digest(option_path),
              "grades": grades, "completed_at": now()}
    if result_path.exists() and any(json.loads(result_path.read_text()).get(k)!=v for k,v in result.items() if k!="completed_at"):
        raise RuntimeError("Completed state differs")
    if not result_path.exists(): atomic_json(result_path, result)
    return result


def main():
    started = time.monotonic()
    verify_models()
    selection = freeze_selection()
    models = load_models()
    grouped = {}
    for item in selection["states"]: grouped.setdefault(item["seed"], []).append(item)
    done = 0
    for seed, items in grouped.items():
        cache = V2 / "snapshots" / f"seed_{seed}.pkl"
        if digest(cache) != items[0]["snapshot_sha256"]: raise RuntimeError("Snapshot hash mismatch")
        snapshots = pickle.loads(cache.read_bytes())
        for item in items:
            run_state(item, models, snapshots)
            done += 1
            print(f"pilot {done}/{len(selection['states'])} seed={seed} step={item['step']}", flush=True)
    atomic_json(RUN/"completed.json", {"version": VERSION, "states": done, "elapsed_seconds": time.monotonic()-started, "completed_at": now()})


if __name__ == "__main__":
    main()
