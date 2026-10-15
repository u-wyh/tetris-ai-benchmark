"""Freeze bounded, episode-balanced PPO death and high-risk survivor reset states."""
import gzip
import hashlib
import io
import json
import pickle
from collections import Counter

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, digest, verify_models
from training.evaluation.ppo_death_capture import CAP, RUN as DEATH_RUN, public_hash
from training.evaluation.ppo_death_replay import load_episode
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now

RUN = ROOT / "runs/ppo30m_failure_curriculum"
DATA_PATH = RUN / "curriculum_snapshots.pkl.gz"
LAYERS = {30: "near_1_100", 80: "near_1_100", 150: "near_101_250",
          230: "near_101_250", 350: "mid_251_500", 480: "mid_251_500",
          650: "mid_501_1000", 900: "mid_501_1000"}
ALL_TIERS = tuple(dict.fromkeys(LAYERS.values())) + ("high_risk_survivor",)


def _verify_snapshot(seed, step, state, expected_hash, tier, source):
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=seed)
        env.set_state(state)
        public = env.core.get_public_observation()
        mask = env.action_masks()
        obs = env.get_observation()
        if public_hash(public) != expected_hash or not mask.any():
            raise RuntimeError(f"Curriculum source snapshot changed at {seed}:{step}")
        return {"seed": seed, "step": step, "tier": tier, "source": source,
                "public_hash": expected_hash, "mask": mask,
                "observation_state": obs["state"], "state": state}
    finally:
        env.close()


def _reconstruct(seed, step, rows, checkpoints):
    previous = max(x for x in checkpoints if x <= step)
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=seed)
        env.set_state(checkpoints[previous]["state"])
        for index in range(previous, step):
            if public_hash(env.core.get_public_observation()) != rows[index]["public_hash"]:
                raise RuntimeError(f"PPO source replay diverged at {seed}:{index}")
            action = rows[index]["action"]
            if not env.action_masks()[action]:
                raise RuntimeError("Source PPO action illegal")
            env.step(action)
        return _verify_snapshot(seed,step,env.get_state(),rows[step]["public_hash"],
                                "pending","new_diagnostic_992")
    finally:
        env.close()


def build():
    verify_models()
    manifest_path = RUN / "curriculum_manifest.json"
    if manifest_path.exists():
        record = json.loads(manifest_path.read_text())
        if digest(DATA_PATH) != record["dataset_sha256"] or record["ppo_sha256"] != HASHES["ppo"]:
            raise RuntimeError("Frozen curriculum dataset changed")
        return record
    capture = json.loads((DEATH_RUN / "capture_completed.json").read_text())
    seeds = capture["initial_seeds"] + capture["extra_seeds"]
    snapshots, sources, selected = [], {}, []
    # Previously designated training data; no prior development/confirmation seed enters training.
    old_folder = ROOT / "runs/ppo30m_local_correction/trajectories"
    for seed in range(730000,730016):
        meta_path = old_folder / f"seed_{seed}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        states_path = old_folder / f"seed_{seed}.states.pkl"
        trace_path = old_folder / f"seed_{seed}.jsonl.gz"
        if (digest(states_path) != meta["states_sha256"] or
                digest(trace_path) != meta["shard_sha256"] or meta["models"]["ppo"] != HASHES["ppo"]):
            raise RuntimeError("Historical training snapshot integrity failed")
        with states_path.open("rb") as file:
            states = pickle.load(file)
        with gzip.open(trace_path,"rt") as file:
            rows = {row["step"]:row for row in (json.loads(line) for line in file)}
        chosen = 0
        for step,state in sorted(states.items()):
            row = rows[step]
            distance = meta["pieces"]-step if meta["game_over"] else None
            tier = next((label for limit,label in LAYERS.items() if distance is not None and
                         abs(distance-limit) <= 50), None)
            if tier is None and (not meta["game_over"] or distance >= 5000):
                public = row["public"]
                heights = [20-next((y for y in range(20) if public["board"][y][x] is not None),20)
                           for x in range(10)]
                if max(heights) >= 14:
                    tier = "high_risk_survivor"
            if tier is None or chosen >= 2:
                continue
            if hashlib.sha256(json.dumps(row["public"],sort_keys=True).encode()).hexdigest() != row["state_sha256"]:
                raise RuntimeError("Historical public hash changed")
            snapshots.append(_verify_snapshot(seed,step,state,public_hash(row["public"]),tier,"prior_train_730"))
            chosen += 1
        if chosen:
            sources[str(seed)] = {"source": "prior_train_730", "trace_sha256":meta["shard_sha256"],
                                  "state_sha256":meta["states_sha256"], "selected":chosen}
    for seed in seeds:
        meta, rows, checkpoints = load_episode(seed)
        steps = []
        if meta["game_over"]:
            steps = [(meta["pieces"]-distance,tier) for distance,tier in LAYERS.items()
                     if meta["pieces"] >= distance+1]
        else:
            # High but survivable states are a control distribution, never safety labels.
            pool = [row for row in rows[1000:min(len(rows)-5000,95000):1000]
                    if row["max_height"] >= 14 or row["holes"] >= 3 or row["deepest_well_depth"] >= 5]
            pool.sort(key=lambda row:(-row["max_height"],-row["holes"],row["step"]))
            steps = [(row["step"],"high_risk_survivor") for row in pool[:2]]
        for step,tier in steps:
            item = _reconstruct(seed,step,rows,checkpoints)
            item["tier"] = tier
            snapshots.append(item)
            selected.append({"seed":seed,"step":step,"tier":tier})
        sources[str(seed)] = {"source":"new_diagnostic_992", "trace_sha256":meta["trace_sha256"],
                              "checkpoint_sha256":meta["checkpoints_sha256"],
                              "game_over":meta["game_over"], "selected":len(steps)}
        del rows,checkpoints
    counts = Counter(row["tier"] for row in snapshots)
    deaths = {row["seed"] for row in snapshots if row["tier"].startswith(("near","mid"))}
    if len(deaths) < 8 or len(snapshots) < 100 or any(counts[tier] == 0 for tier in ALL_TIERS):
        raise RuntimeError(f"Curriculum data gate failed: {len(deaths)} death seeds, {len(snapshots)} states, tiers {dict(counts)}")
    payload = {"ppo_sha256":HASHES["ppo"], "snapshots":snapshots,
               "selection":"fixed death distances plus capped high-risk controls; per new episode <=8 death or <=2 control"}
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode="wb",compresslevel=3) as file:
        file.write(pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL))
    RUN.mkdir(parents=True,exist_ok=True)
    atomic_bytes(DATA_PATH,buffer.getvalue())
    record = {"created_at":now(), "ppo_sha256":HASHES["ppo"],
              "dataset_sha256":digest(DATA_PATH), "snapshots":len(snapshots),
              "independent_death_seeds":len(deaths), "tier_counts":dict(counts),
              "source_episodes":sources, "selection":selected,
              "historical_role":"730000..730015 are prior training; 992xxx are diagnostics and now training; none may be independent validation"}
    atomic_json(manifest_path,record)
    return record


if __name__ == "__main__":
    print(build()["tier_counts"],flush=True)
