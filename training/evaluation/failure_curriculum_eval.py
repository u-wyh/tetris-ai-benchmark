"""Normal-start paired evaluation and frozen go/no-go decision for curriculum PPO."""
import json
import os
import pickle
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.candidate_features import candidate_features
from training.evaluation.fast_legal import get_legal_placements as fast_placements
from training.evaluation.hybrid_queue import BC, PPO, HASHES, digest, verify_models
from training.evaluation.ppo_lookahead_v3_audit import policy_probs
from training.hybrid_policy import HybridConfig, HybridController
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/ppo30m_failure_curriculum"
DEV = tuple(range(993000,993016))
CONFIRM = tuple(range(993100,993132))
MODELS = None
CHECKPOINT_INTERVAL = 10000


def atomic_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class LongCandidateTetrisEnv(FastCandidateTetrisEnv):
    """Evaluation-only placement generator; Core and action semantics stay unchanged."""

    def _features_and_mask(self):
        if self.core is None:
            raise RuntimeError("Call reset() before requesting candidates")
        if self._candidate_cache is None:
            public = self.core.get_public_observation()
            self._placement_cache = fast_placements(public)
            self._candidate_cache = candidate_features(public, self._placement_cache)
        return self._candidate_cache


def audit_seeds():
    for seed in DEV+CONFIRM:
        for base in (ROOT/"runs", ROOT/"reports"):
            if any(not path.is_relative_to(RUN) for path in base.rglob(f"seed_{seed}.*")):
                raise RuntimeError(f"Curriculum evaluation seed was previously used: {seed}")


def checkpoint(variant, task=123):
    if variant in ("ppo","B12","bc"):
        return {"ppo":PPO,"B12":PPO,"bc":BC}[variant]
    if variant == "normal":
        task = 123  # Fixed-budget control never extends with B.
    return RUN / variant / "committed" / f"task_{task:06d}" / "model.zip"


def init_worker(paths):
    global MODELS
    import torch
    torch.set_num_threads(1)
    MODELS = {name:MaskablePPO.load(str(path),device="cpu") for name,path in paths.items()}


def fast_action(model, observation, mask):
    """Deterministic candidate argmax without building a 1840-way distribution."""
    legal = np.flatnonzero(mask)
    if not len(legal):
        raise ValueError("No legal evaluation action")
    policy = model.policy
    with torch.inference_mode():
        state = torch.as_tensor(observation["state"], dtype=torch.float32,
                                device=policy.device).unsqueeze(0)
        candidates = torch.as_tensor(observation["candidate"][legal],
                                     dtype=torch.float32, device=policy.device)
        latent = policy.state_encoder(state).expand(len(legal), -1)
        encoded = policy.candidate_encoder(candidates / policy.candidate_scale)
        scores = policy.score_head(torch.cat((latent, encoded), dim=1)).squeeze(-1)
        return int(legal[int(torch.argmax(scores))])


class _DeterministicAdapter:
    def __init__(self, model):
        self.model = model

    def predict(self, observation, deterministic=True, action_masks=None):
        if not deterministic or action_masks is None:
            raise ValueError("Long-run adapter requires deterministic masked actions")
        return np.asarray(fast_action(self.model, observation, action_masks)), None


def play(variant, seed, cap, model_hashes, fast=False, fast_legal=False,
         checkpoint_path=None):
    env = (LongCandidateTetrisEnv if fast_legal else FastCandidateTetrisEnv)(max_pieces=cap)
    started = time.monotonic()
    clears = [0]*5
    milestones = []
    samples = {"count":0,"top1_same":0,"top5_kept":0,"hold_same":0,"masked_kl_sum":0.0}
    if variant == "B12":
        ppo = _DeterministicAdapter(MODELS["ppo"]) if fast else MODELS["ppo"]
        bc = _DeterministicAdapter(MODELS["bc"]) if fast else MODELS["bc"]
        controller = HybridController(ppo, bc, HybridConfig("B",12,8))
    else:
        controller = None
    try:
        if checkpoint_path is not None and checkpoint_path.exists():
            with checkpoint_path.open("rb") as stream:
                saved = pickle.load(stream)
            expected = (variant, seed, cap, model_hashes, fast, fast_legal)
            if saved["identity"] != expected:
                raise RuntimeError("Long game checkpoint protocol mismatch")
            env.reset(seed=seed)
            env.set_state(saved["env_state"])
            if env.pieces != saved["pieces"] or env.pieces >= cap:
                raise RuntimeError("Long game checkpoint step mismatch")
            clears = saved["clears"]
            milestones = saved["milestones"]
            samples = saved["samples"]
            if controller:
                controller.active, controller.last_switch, controller.step = saved["controller"]
            started -= saved["elapsed_seconds"]
            obs = env.get_observation()
        else:
            obs,_ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            if fast_legal and (env.pieces < 100 or env.pieces % 1000 == 0):
                if env._placement_cache != env.core.get_legal_placements():
                    raise RuntimeError("Accelerated legal placements differ from official Core")
            if controller:
                action,_ = controller.decide(obs,mask,env.core.get_public_observation(),env._placement_cache)
            else:
                action = (fast_action(MODELS[variant], obs, mask) if fast
                          else int(MODELS[variant].predict(obs,deterministic=True,action_masks=mask)[0]))
            if not mask[action]:
                raise RuntimeError("Formal evaluation action illegal")
            if variant in ("normal","curriculum") and env.pieces%500==0:
                reference = policy_probs(MODELS["ppo"],obs,mask)
                current = policy_probs(MODELS[variant],obs,mask)
                ranking = sorted(np.flatnonzero(mask),key=lambda a:(-reference[a],a))
                original = ranking[0]
                by_id = {p["actionId"]:p for p in env._placement_cache}
                samples["count"] += 1
                samples["top1_same"] += int(action==original)
                samples["top5_kept"] += int(action in ranking[:5])
                samples["hold_same"] += int(bool(by_id[action]["hold"])==bool(by_id[original]["hold"]))
                legal = np.flatnonzero(mask)
                samples["masked_kl_sum"] += float(np.sum(current[legal]*(np.log(np.maximum(current[legal],1e-12))-
                                                                      np.log(np.maximum(reference[legal],1e-12)))))
            obs,_,over,capped,info = env.step(action)
            clears[info["cleared_lines"]] += 1
            if info["pieces"]%1000==0:
                milestones.append({"pieces":info["pieces"],"score":info["score"]})
            if (checkpoint_path is not None and info["pieces"] % CHECKPOINT_INTERVAL == 0
                    and not (over or capped)):
                atomic_pickle(checkpoint_path, {
                    "identity": (variant, seed, cap, model_hashes, fast, fast_legal),
                    "pieces": env.pieces, "env_state": env.get_state(),
                    "clears": clears, "milestones": milestones, "samples": samples,
                    "controller": ((controller.active, controller.last_switch, controller.step)
                                   if controller else None),
                    "elapsed_seconds": time.monotonic() - started})
            if over or capped:
                break
        return {"variant":variant,"seed":seed,"cap":cap,"model_sha256":model_hashes,
                "pieces":info["pieces"],"game_over":bool(over),"capped":bool(capped),
                "score":info["score"],"clears_0_to_4":clears,"score_milestones":milestones,
                "policy_samples":samples,"wall_seconds":time.monotonic()-started}
    finally:
        env.close()


def task(args):
    variant,seed,cap,hashes,fast,fast_legal,checkpoint_path = args
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    return play(variant,seed,cap,hashes,fast,fast_legal,
                None if checkpoint_path is None else Path(checkpoint_path))


def _summary(rows):
    pieces = [r["pieces"] for r in rows]
    return {"games":len(rows),"mean_pieces":statistics.mean(pieces),
            "p10_pieces":float(np.percentile(pieces,10)),
            "capped":sum(r["capped"] for r in rows),
            "early_under_10000":sum(r["pieces"]<10000 for r in rows),
            "mean_score":statistics.mean(r["score"] for r in rows),
            "score_per_1000":1000*sum(r["score"] for r in rows)/sum(pieces),
            "four_fraction":sum(r["clears_0_to_4"][4] for r in rows)/
                            max(1,sum(sum(r["clears_0_to_4"][1:]) for r in rows))}


def _common_score(a,b):
    common = min(a["pieces"],b["pieces"])
    index = common//1000
    if index == 0:
        return None
    return (a["score_milestones"][index-1]["score"], b["score_milestones"][index-1]["score"])


def evaluate(phase, variants, seeds, cap, task_number=123, workers=4,
             fast=False, fast_legal=False):
    verify_models()
    audit_seeds()
    paths = {"ppo":PPO,"bc":BC}
    for variant in variants:
        if variant in ("normal","curriculum"):
            paths[variant] = checkpoint(variant,task_number)
    hashes = {variant:digest(checkpoint(variant,task_number)) for variant in variants}
    folder = RUN/phase
    folder.mkdir(parents=True,exist_ok=True)
    protocol = {"phase":phase,"variants":list(variants),"seeds":list(seeds),
                "cap":cap,"checkpoint_task":task_number,"hashes":hashes}
    if fast:
        protocol["deterministic_inference"] = "candidate_argmax_v1"
    if fast_legal:
        protocol["legal_placements"] = "bitboard_public_bfs_v1"
    config_path = folder/"protocol.json"
    if config_path.exists() and json.loads(config_path.read_text()) != protocol:
        raise RuntimeError("Frozen curriculum evaluation protocol changed")
    atomic_json(config_path,protocol)
    rows = {}
    if not 1 <= workers <= 8:
        raise ValueError("Evaluation workers must be between 1 and 8")
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn"),
                             initializer=init_worker,initargs=(paths,)) as pool:
        futures = {}
        for variant in variants:
            for seed in seeds:
                path = folder/variant/f"seed_{seed}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    if row["model_sha256"] != hashes or row["seed"] != seed or row["cap"] != cap:
                        raise RuntimeError("Existing formal game mismatches protocol")
                    rows[(variant,seed)] = row
                else:
                    checkpoint_file = (str(path.with_suffix(".checkpoint.pkl"))
                                       if cap >= 1000000 else None)
                    futures[pool.submit(task,(variant,seed,cap,hashes,fast,
                                              fast_legal,checkpoint_file))] = (variant,seed,path)
        for future in as_completed(futures):
            variant,seed,path = futures[future]
            row = future.result()
            atomic_json(path,row)
            path.with_suffix(".checkpoint.pkl").unlink(missing_ok=True)
            rows[(variant,seed)] = row
            atomic_json(folder/"status.json",{"completed":len(rows),"total":len(variants)*len(seeds),
                                               "last_variant":variant,"last_seed":seed,"at":now()})
    summary = {variant:_summary([rows[(variant,seed)] for seed in seeds]) for variant in variants}
    paired = {}
    for variant in variants:
        if variant=="ppo": continue
        pairs = [_common_score(rows[(variant,seed)],rows[("ppo",seed)]) for seed in seeds]
        pairs = [x for x in pairs if x is not None]
        paired[variant] = {"common_score_ratio":sum(a for a,b in pairs)/max(1,sum(b for a,b in pairs)),
                           "survival_deltas":[rows[(variant,seed)]["pieces"]-rows[("ppo",seed)]["pieces"]
                                              for seed in seeds]}
    report = {"protocol":protocol,"summary":summary,"paired":paired}
    atomic_json(folder/"summary.json",report)
    return report


def gate(report, stage="development"):
    baseline = report["summary"]["ppo"]
    decisions = {}
    for variant in ("normal","curriculum"):
        if variant not in report["summary"]:
            continue
        row = report["summary"][variant]
        p = report["paired"][variant]
        decisions[variant] = {"four_drop_pp":100*(baseline["four_fraction"]-row["four_fraction"]),
                              "common_score_ratio":p["common_score_ratio"],
                              "p10_ratio":row["p10_pieces"]/baseline["p10_pieces"],
                              "mean_survival_gain":row["mean_pieces"]-baseline["mean_pieces"],
                              "early_death_excess":row["early_under_10000"]-baseline["early_under_10000"]}
        d = decisions[variant]
        d["pass"] = (d["four_drop_pp"]<=3 and d["common_score_ratio"]>=.95 and
                     d["p10_ratio"]>=.95 and d["mean_survival_gain"]>=1000 and
                     d["early_death_excess"]<=0)
    # Only curriculum may extend; A is a fixed-budget control.
    result = {"rules":"four drop <=3pp, common score >=95%, P10 >=95%, mean +>=1000, no extra <10K death",
              "decisions":decisions,"extend_curriculum":decisions["curriculum"]["pass"],
              "extend_normal":False}
    atomic_json(RUN/f"{stage}_gate.json",result)
    return result
