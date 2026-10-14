"""Independent, resumable evaluation of frozen B12 and safer intercept rules."""
import hashlib
import json
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, load_models, run_game, verify_models
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/hybrid_v2"
DEV = list(range(710000, 710016))
CONFIRM = list(range(710100, 710132))
SMOKE = [710200, 710201]
WORKER_MODELS = None


def rules_digest():
    return hashlib.sha256((ROOT / "training/hybrid_v2_policy.py").read_bytes()).hexdigest()


def audit_seeds():
    for seed in DEV + CONFIRM + SMOKE:
        for base in (ROOT / "runs", ROOT / "reports"):
            for suffix in ("json", "jsonl.gz"):
                if any(not p.is_relative_to(RUN) for p in base.rglob(f"seed_{seed}.{suffix}")):
                    raise RuntimeError(f"Reserved V2 development seed already used: {seed}")


def init_worker():
    global WORKER_MODELS
    WORKER_MODELS = load_models()


def play_v2(models, variant, seed, cap):
    from training.hybrid_v2_policy import V2Controller
    env = FastCandidateTetrisEnv(max_pieces=cap)
    control = V2Controller(models["ppo"], models["bc"], variant)
    started = time.monotonic()
    trace, switches, selected = [], [], {"ppo": 0, "bc": 0}
    clears, milestones = [0] * 5, []
    inference_seconds = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            begin = time.monotonic()
            action, decision = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            inference_seconds += time.monotonic() - begin
            if not mask[action]:
                raise RuntimeError("V2 chose illegal action")
            index, source = env.pieces, decision["source"]
            if not trace or trace[-1]["source"] != source:
                trace.append({"start": index, "source": source})
                switches.append({"piece": index, **decision})
            selected[source] += 1
            obs, _, terminated, truncated, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            if info["pieces"] % 1000 == 0:
                milestones.append({"pieces": info["pieces"], "score": info["score"]})
            if terminated or truncated:
                break
        return {"seed": seed, "variant": variant, "max_pieces": cap,
                "model_sha256": HASHES, "pieces_survived": info["pieces"],
                "score": info["score"], "lines": info["lines"],
                "game_over": bool(terminated), "survived_cap": bool(truncated),
                "clears_0_to_4": clears, "score_milestones": milestones,
                "source_runs": trace, "switches": switches, "actions_by_source": selected,
                "height_at_end": info["height_after"], "holes_at_end": info["holes_after"],
                "inference_seconds": inference_seconds,
                "wall_seconds": time.monotonic() - started}
    finally:
        env.close()


def worker(variant, seed, cap, output):
    row = (run_game(WORKER_MODELS, variant, seed, cap) if variant in ("ppo", "bc", "B12")
           else play_v2(WORKER_MODELS, variant, seed, cap))
    atomic_json(output, row)
    return seed, row["pieces_survived"]


def aggregate(rows):
    pieces = sorted(r["pieces_survived"] for r in rows)
    n = len(pieces)
    rank = .1 * (n - 1)
    p10 = pieces[int(rank)] + (pieces[min(int(rank) + 1, n - 1)] - pieces[int(rank)]) * (rank % 1)
    return {"games": n, "mean_pieces": statistics.mean(pieces),
        "median_pieces": statistics.median(pieces), "p10_pieces": p10,
        "capped": sum(r["survived_cap"] for r in rows),
        "game_over": sum(r["game_over"] for r in rows),
        "mean_score": statistics.mean(r["score"] for r in rows),
        "score_per_1000_pieces": 1000 * sum(r["score"] for r in rows) / sum(pieces),
        "four_clear_count": sum(r["clears_0_to_4"][4] for r in rows),
        "four_clear_fraction": sum(r["clears_0_to_4"][4] for r in rows) /
            max(1, sum(sum(r["clears_0_to_4"][1:]) for r in rows)),
        "ppo_action_fraction": sum(r["actions_by_source"]["ppo"] for r in rows) / sum(pieces),
        "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in rows)}


def evaluate(phase, variants, seeds, cap, rule_hash):
    for variant in variants:
        folder = RUN / phase / variant
        folder.mkdir(parents=True, exist_ok=True)
        config = {"phase": phase, "variant": variant, "seeds": seeds, "cap": cap,
                  "model_sha256": HASHES, "v2_rules_sha256": rule_hash,
                  "deterministic": True, "action_mask": True}
        path = folder / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise RuntimeError(f"V2 protocol changed: {path}")
        if not path.exists():
            atomic_json(path, config)
        pending = []
        for seed in seeds:
            output = folder / f"seed_{seed}.json"
            if output.exists():
                row = json.loads(output.read_text())
                if (row["seed"] != seed or row["variant"] != variant or
                    row["max_pieces"] != cap or row["model_sha256"] != HASHES or
                    row["pieces_survived"] > cap or row["game_over"] == row["survived_cap"] or
                    sum(row["actions_by_source"].values()) != row["pieces_survived"]):
                    raise RuntimeError(f"Invalid saved V2 game: {output}")
            else:
                pending.append((variant, seed, cap, output))
        if pending:
            with ProcessPoolExecutor(max_workers=min(8, len(pending)),
                    mp_context=get_context("spawn"), initializer=init_worker) as pool:
                futures = [pool.submit(worker, *task) for task in pending]
                for future in as_completed(futures):
                    seed, pieces = future.result()
                    atomic_json(RUN / "status.json", {"state": "running", "phase": phase,
                        "variant": variant, "last_seed": seed,
                        "completed": sum((folder / f"seed_{s}.json").exists() for s in seeds),
                        "total": len(seeds), "updated_at": now()})
                    print(f"{phase} {variant} seed={seed} pieces={pieces}", flush=True)
        rows = [json.loads((folder / f"seed_{seed}.json").read_text()) for seed in seeds]
        atomic_json(folder / "summary.json", aggregate(rows))


def main():
    from training.hybrid_v2_policy import VARIANTS
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        verify_models()
        audit_seeds()
        if not (ROOT / "reports/experiments/hybrid_b12_failure_analysis.md").exists():
            raise RuntimeError("Finish failure diagnosis before V2 evaluation")
        if not (ROOT / "reports/experiments/hybrid_v2_design.md").exists():
            raise RuntimeError("Freeze V2 design before evaluation")
        rule_hash = rules_digest()
        variants = ["ppo", "bc", "B12", *VARIANTS]
        protocol = {"models": HASHES, "v2_rules_sha256": rule_hash,
            "dev": DEV, "confirm": CONFIRM, "smoke": SMOKE, "variants": variants,
            "max_pieces": 100000}
        path = RUN / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise RuntimeError("Frozen V2 protocol changed")
        if not path.exists():
            atomic_json(path, protocol)
        evaluate("smoke", variants, SMOKE, 100, rule_hash)
        evaluate("development_20k", variants, DEV, 20000, rule_hash)
        # Fixed variants: no tuning or exclusion based on confirmation outcomes.
        atomic_json(RUN / "freeze.json", {"variants": variants, "model_sha256": HASHES,
                    "v2_rules_sha256": rule_hash, "confirm_seeds": CONFIRM,
                    "frozen_at": now()})
        evaluate("development_100k", variants, DEV, 100000, rule_hash)
        evaluate("confirmation_100k", variants, CONFIRM, 100000, rule_hash)
        from training.evaluation.hybrid_v2_report import main as make_report
        make_report()
        atomic_json(RUN / "status.json", {"state": "completed", "updated_at": now()})
    except Exception:
        atomic_json(RUN / "status.json", {"state": "failed", "error": traceback.format_exc(),
                    "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
