"""Verify frozen PPO action and public hash from sparse Core/RNG checkpoints."""
import gzip
import json
import pickle

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, digest
from training.evaluation.ppo_death_capture import CAP, RUN, VERSION, load_ppo, public_hash, trace_paths
from training.train_ppo import atomic_json


def load_episode(seed):
    trace_path, state_path, meta_path = trace_paths(seed)
    meta = json.loads(meta_path.read_text())
    if (meta["version"] != VERSION or meta["ppo_sha256"] != HASHES["ppo"] or
        digest(trace_path) != meta["trace_sha256"] or
        digest(state_path) != meta["checkpoints_sha256"]):
        raise RuntimeError(f"Episode integrity failed: {seed}")
    with gzip.open(trace_path, "rt") as file:
        rows = [json.loads(line) for line in file]
    with gzip.open(state_path, "rb") as file:
        checkpoints = pickle.load(file)  # Trusted, locally generated experiment artifact.
    if len(rows) != meta["pieces"] or 0 not in checkpoints:
        raise RuntimeError(f"Incomplete episode trace: {seed}")
    return meta, rows, checkpoints


def validate(seed, model=None, all_segments=False):
    meta, rows, checkpoints = load_episode(seed)
    model = model or load_ppo()
    steps = sorted(checkpoints)
    chosen = steps if all_segments else sorted({0, steps[len(steps)//2], steps[-1]})
    checked = 0
    for start in chosen:
        entry = checkpoints[start]
        env = FastCandidateTetrisEnv(max_pieces=CAP)
        try:
            env.reset(seed=seed)
            env.set_state(entry["state"])
            if env.core.rng.state != entry["rng_state"]:
                raise RuntimeError(f"RNG mismatch at {seed}:{start}")
            obs = env.get_observation()
            stop = min(start+1000, len(rows))
            for step in range(start, stop):
                row = rows[step]
                if row["step"] != step or public_hash(env.core.get_public_observation()) != row["public_hash"]:
                    raise RuntimeError(f"Public hash mismatch at {seed}:{step}")
                mask = env.action_masks()
                action = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
                if action != row["action"] or not mask[action]:
                    raise RuntimeError(f"PPO action mismatch at {seed}:{step}")
                obs, _, over, capped, info = env.step(action)
                if bool(over) != row["game_over_after"] or bool(capped) != row["capped_after"]:
                    raise RuntimeError(f"Outcome mismatch at {seed}:{step}")
                checked += 1
            if stop == len(rows):
                if info["pieces"] != meta["pieces"] or info["score"] != meta["score"] or bool(over) != meta["game_over"]:
                    raise RuntimeError(f"Terminal outcome mismatch: {seed}")
            elif stop in checkpoints and public_hash(env.core.get_public_observation()) != checkpoints[stop]["public_hash"]:
                raise RuntimeError(f"Next checkpoint mismatch: {seed}:{stop}")
        finally:
            env.close()
    record = {"seed": seed, "trace_sha256": meta["trace_sha256"],
              "checkpoints_sha256": meta["checkpoints_sha256"],
              "checked_segments": chosen, "checked_actions": checked,
              "terminal_checked": steps[-1] in chosen, "valid": True}
    atomic_json(RUN / "replay_validation" / f"seed_{seed}.json", record)
    return record
