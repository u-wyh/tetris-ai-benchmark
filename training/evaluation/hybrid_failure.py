"""Read-only deterministic B12 failure replay and bounded counterfactual branches."""
import hashlib
import json
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, RUN as OLD_RUN, load_models, verify_models
from training.hybrid_policy import F, HybridConfig, HybridController
from training.candidate_features import board_features
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/hybrid_v2"
FAILURES = (700016, 700017, 700020, 700023, 700026, 700029)
OLD = ROOT / "runs/candidate_ppo30m_bc_hybrid/confirm_100k/B12"
WORKER_MODELS = None


def init_worker():
    global WORKER_MODELS
    WORKER_MODELS = load_models()


def worker(seed):
    return analyze_seed(WORKER_MODELS, seed)


def branch(models, state, action, start, horizon=5000):
    env = FastCandidateTetrisEnv(max_pieces=100000)
    env.reset(seed=state["episode_seed"])
    env.set_state(state)
    if pickle.dumps(env.get_state()["core"]) != pickle.dumps(state["core"]):
        raise RuntimeError("Counterfactual Core snapshot changed during restore")
    control = HybridController(models["ppo"], models["bc"], HybridConfig("B", 12, 8))
    control.step = start + 1
    before = env.core.get_public_observation()
    mask = env.action_masks()
    if not mask[action]:
        raise RuntimeError("Counterfactual first action is illegal")
    initial_score = env.core.score
    clears = [0] * 5
    try:
        obs, _, terminated, truncated, info = env.step(action)
        clears[info["cleared_lines"]] += 1
        first = env.core.get_public_observation()
        while not (terminated or truncated) and env.pieces - start - 1 < horizon:
            mask = env.action_masks()
            selected, _ = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            obs, _, terminated, truncated, info = env.step(selected)
            clears[info["cleared_lines"]] += 1
        return {"pieces_after_branch": env.pieces - start, "end_piece": env.pieces,
                "score_gained": env.core.score - initial_score, "clears_0_to_4": clears,
                "game_over": bool(terminated), "horizon_censored": not (terminated or truncated),
                "first_state": {"hold": first["hold"], "next": first["next"],
                                "current": first["currentPiece"]["type"] if first["currentPiece"] else None,
                                "board_sha256": hashlib.sha256(repr(first["board"]).encode()).hexdigest()},
                "initial_public": {"hold": before["hold"], "next": before["next"]}}
    finally:
        env.close()


def analyze_seed(models, seed):
    output = RUN / "failure_analysis" / f"seed_{seed}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("replay_verified") and "intercepts" in saved:
            return saved
    old = json.loads((OLD / f"seed_{seed}.json").read_text())
    transitions = {s["piece"]: s for s in old["switches"]}
    bc_events = [s for s in old["switches"] if s["source"] == "bc"]
    targets = {"first": bc_events[0]["piece"],
               "height15": next((s["piece"] for s in bc_events if s["ppo_risk"]["height"] >= 15), None),
               "height18": next((s["piece"] for s in bc_events if s["ppo_risk"]["height"] >= 18), None),
               "late": next((s["piece"] for s in reversed(bc_events)
                             if s["piece"] >= old["pieces_survived"] - 500), None)}
    env = FastCandidateTetrisEnv(max_pieces=100000)
    control = HybridController(models["ppo"], models["bc"], HybridConfig("B", 12, 8))
    started = time.monotonic()
    details, last, intercepts = {}, [], []
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            index = env.pieces
            mask = env.action_masks()
            action, decision = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            if index in transitions:
                expected = transitions[index]
                for key in ("source", "reason", "ppo_action", "bc_action"):
                    if decision[key] != expected[key]:
                        raise RuntimeError(f"B12 replay diverged at {seed}:{index}:{key}")
            if index in targets.values():
                snapshot = env.get_state()
                digest = hashlib.sha256(pickle.dumps(snapshot["core"])).hexdigest()
                f = obs["candidate"]
                p, b = decision["ppo_action"], decision["bc_action"]
                label = [name for name, piece in targets.items() if piece == index]
                branches = {name: branch(models, snapshot, selected, index)
                            for name, selected in (("ppo", p), ("bc", b))}
                for name in label:
                    details[name] = {"piece": index, "state_sha256": digest,
                        "ppo_candidate_clears": int(f[p, F["cleared_lines"]]),
                        "bc_candidate_clears": int(f[b, F["cleared_lines"]]),
                        "decision": decision, "branches": branches}
            if decision["source"] == "bc":
                f = obs["candidate"]
                p, b = decision["ppo_action"], decision["bc_action"]
                intercepts.append({"piece": index,
                    "ppo_clears": int(f[p, F["cleared_lines"]]),
                    "bc_clears": int(f[b, F["cleared_lines"]]),
                    "ppo_hold": int(f[p, F["hold"]]), "bc_hold": int(f[b, F["hold"]]),
                    "ppo_risk": decision["ppo_risk"], "bc_risk": decision["bc_risk"]})
            obs, _, terminated, truncated, info = env.step(action)
            board = env.core.board
            bumpiness = board_features(board)[3]
            central = sum(board[y][x] is not None for y in range(4) for x in range(3, 7))
            last.append({"piece": info["pieces"], "height": info["height_after"],
                         "holes": info["holes_after"], "bumpiness": bumpiness,
                         "central_top4": central, "cleared": info["cleared_lines"],
                         "source": decision["source"]})
            if len(last) > 1000:
                last.pop(0)
            if terminated or truncated:
                break
        if info["pieces"] != old["pieces_survived"] or info["score"] != old["score"]:
            raise RuntimeError(f"B12 final replay mismatch: {seed}")
        record = {"seed": seed, "model_sha256": HASHES, "old_pieces": old["pieces_survived"],
                  "replay_score": info["score"], "replay_verified": True,
                  "bc_actions": old["actions_by_source"]["bc"],
                  "bc_action_fraction": old["actions_by_source"]["bc"] / info["pieces"],
                  "intercept_runs": len(bc_events), "first_intercept": bc_events[0]["piece"],
                  "candidate_cases": details, "intercepts": intercepts,
                  "last_100": last[-100:], "last_500": last[-500:], "last_1000": last,
                  "wall_seconds": time.monotonic() - started, "completed_at": now()}
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, record)
        return record
    finally:
        env.close()


def main():
    verify_models()
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=init_worker) as pool:
        futures = {pool.submit(worker, seed): seed for seed in FAILURES}
        for future in as_completed(futures):
            record = future.result()
            print(f"verified B12 seed={record['seed']} pieces={record['old_pieces']} cases={list(record['candidate_cases'])}", flush=True)


if __name__ == "__main__":
    main()
