"""Frozen, staged and resumable B12 Rescue-A evaluation; no model training."""

import json
import os
import pickle
import re
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, _DeterministicAdapter, atomic_pickle, fast_action
from training.evaluation.hybrid_queue import BC, PPO, HASHES, digest, verify_models
from training.hybrid_b12_v2_rescue_policy import RescueController, VERSION
from training.hybrid_policy import HybridConfig, HybridController
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/hybrid_b12_v2_rescue"
REPORT = ROOT / "reports/experiments/hybrid_b12_v2_rescue.md"
DEV = tuple(range(1440000,1440032))
CONFIRM = tuple(range(1441000,1441032))
SMOKE = (1440040,)
VARIANTS = ("ppo","bc","B12","Rescue")
CAP = 100000
WORKERS = 4
CHECKPOINT = 10000
MODELS = None


def seed_audit():
    targets = set(DEV+CONFIRM+SMOKE)
    numeric = re.compile(r"(?<!\d)(?:14400(?:0[0-9]|1[0-9]|2[0-9]|3[01]|40)|14410(?:0[0-9]|1[0-9]|2[0-9]|3[01]))(?!\d)")
    field = re.compile(r"(?:\"seed\"\s*:\s*|seed[_=]|\bseed\s+)(\d+)\b",re.I)
    list_field = re.compile(r'"(?:seeds|dev|confirm|evaluation_seeds|validation_seed_set)"\s*:\s*\[([^\]]*)\]')
    hits = []
    for root in (ROOT/"runs", ROOT/"reports", ROOT/"training", ROOT/"tests"):
        for path in root.rglob("*"):
            if not path.is_file() or path.is_relative_to(RUN) or path == REPORT or path == Path(__file__):
                continue
            if numeric.search(path.name):
                hits.append(str(path.relative_to(ROOT)))
                continue
            if path.suffix not in (".json",".md",".py",".csv",".txt") or path.stat().st_size > 2_000_000:
                continue
            content = path.read_text(errors="replace")
            if any(int(m.group(1)) in targets for m in field.finditer(content)):
                hits.append(str(path.relative_to(ROOT)))
                continue
            if any(int(n) in targets for m in list_field.finditer(content)
                   for n in re.findall(r"\b\d+\b",m.group(1))):
                hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise RuntimeError(f"Rescue evaluation seed collision: {hits[:20]}")
    return {"dev":list(DEV),"confirm":list(CONFIRM),"smoke":list(SMOKE),
            "hits":[],"audited_at":now()}


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    verify_models()
    path = RUN/"protocol.json"
    if not path.exists():
        atomic_json(RUN/"seed_audit.json",seed_audit())
    spec = {"models":HASHES, "b12_rule_sha256":digest(ROOT/"training/hybrid_policy.py"),
            "rescue_rule_sha256":digest(ROOT/"training/hybrid_b12_v2_rescue_policy.py"),
            "rescue_version":VERSION,"core_sha256":digest(ROOT/"training/tetris_core/core.py"),
            "dev":list(DEV),"confirm":list(CONFIRM),"smoke":list(SMOKE),
            "variants":list(VARIANTS),"workers":WORKERS,"cap":CAP,
            "checkpoint_interval":CHECKPOINT,
            "development_20k_screen":{"min_score_ratio":.95,
                                      "min_four_ratio":.9,"max_early_death_excess":0},
            "development_100k_gate":{"p10_strictly_higher":True,"min_mean_survival_ratio":1.0,
                                      "min_score_ratio":.95,"min_four_ratio":.9,
                                      "max_early_death_excess":0},
            "confirmation_success":{"p10_strictly_higher":True,"mean_survival_strictly_higher":True,
                                    "paired_mean_survival_ci95_lower_positive":True,
                                    "min_score_ratio":.95,"min_four_ratio":.9,
                                    "max_early_death_excess":0},
            "selection":"Rescue only; no parameter tuning on development results",
            "confirmation_use":"one_shot_no_retuning",
            "inference":"deterministic_masked_cpu_fast_candidate_argmax",
            "score_rule":"official_core_v1","right_censoring":"cap_is_not_death"}
    if path.exists():
        if json.loads(path.read_text()) != spec:
            raise RuntimeError("Frozen Rescue protocol changed")
    else:
        atomic_json(path,spec)
    return spec


def status(stage,**kw):
    atomic_json(RUN/"status.json",{"stage":stage,"at":now(),"pid":os.getpid(),**kw})


def init_worker():
    global MODELS
    import torch
    torch.set_num_threads(1)
    MODELS = {"ppo":MaskablePPO.load(str(PPO),device="cpu"),
              "bc":MaskablePPO.load(str(BC),device="cpu")}
    for model in MODELS.values():
        if model.policy.__class__.__name__ != "CandidateScoringPolicy" or model.action_space.n != 1840:
            raise RuntimeError("Frozen model policy/action space changed")


def make_controller(variant):
    p,b = _DeterministicAdapter(MODELS["ppo"]),_DeterministicAdapter(MODELS["bc"])
    if variant == "B12":
        return HybridController(p,b,HybridConfig("B",12,8))
    if variant == "Rescue":
        return RescueController(p,b)
    return None


def checkpoint_file(phase,variant,seed):
    return RUN/phase/variant/f"seed_{seed}.checkpoint.pkl"


def game(phase,variant,seed,cap,protocol):
    checkpoint = checkpoint_file(phase,variant,seed)
    env = LongCandidateTetrisEnv(max_pieces=cap)
    controller = make_controller(variant)
    started = time.monotonic()
    clears = [0]*5; milestones=[]; trace=[]; events=[]
    selected={"ppo":0,"bc":0,"v2":0}; proposed=0; rejected=0; reject_reasons={}
    inference_seconds=0.0
    try:
        obs,_=env.reset(seed=seed)
        if checkpoint.exists():
            seal=checkpoint.with_suffix(".sha256.json")
            if not seal.exists() or json.loads(seal.read_text())["sha256"]!=digest(checkpoint):
                raise RuntimeError("Rescue checkpoint checksum mismatch")
            saved=pickle.loads(checkpoint.read_bytes())
            if saved["identity"] != (phase,variant,seed,cap,protocol["models"],protocol["rescue_rule_sha256"]):
                raise RuntimeError("Rescue checkpoint protocol mismatch")
            env.set_state(saved["env_state"])
            obs=env.get_observation()
            clears,milestones,trace,events=(saved[k] for k in ("clears","milestones","trace","events"))
            selected,proposed,rejected,reject_reasons=(saved[k] for k in
                ("selected","proposed","rejected","reject_reasons"))
            inference_seconds=saved["inference_seconds"]
            started-=saved["elapsed_seconds"]
            if controller:
                base=controller.base if variant=="Rescue" else controller
                base.active,base.last_switch,base.step=saved["control"]
        while True:
            mask=env.action_masks()
            public=env.core.get_public_observation()
            if env.pieces<100 or env.pieces%1000==0:
                if env._placement_cache != env.core.get_legal_placements():
                    raise RuntimeError("Fast legal placements differ from Core")
            t0=time.monotonic()
            if controller:
                action,decision=controller.decide(obs,mask,public,env._placement_cache)
                source=decision["source"]
                if variant=="Rescue" and decision["v2_called"]:
                    proposed+=1
                    if decision["v2_changed"]:
                        rejected+=1
                    reason=decision["reason"]
                    reject_reasons[reason]=reject_reasons.get(reason,0)+1
                    events.append({"piece":env.pieces,"source":source,
                        "base_source":decision["base_source"],"reason":reason,
                        "ppo_action":decision["ppo_action"],"bc_action":decision["bc_action"],
                        "v2_action":decision["v2_action"],"v2_changed":decision["v2_changed"],
                        "v2_seconds":decision["v2_seconds"],"rescue_features":decision["rescue_features"]})
            else:
                source=variant
                action=fast_action(MODELS[variant],obs,mask)
            inference_seconds+=time.monotonic()-t0
            if not mask[action]:
                raise RuntimeError("Rescue evaluation action illegal")
            if not trace or trace[-1]["source"]!=source:
                trace.append({"piece":env.pieces,"source":source})
            selected[source]+=1
            obs,_,over,capped,info=env.step(action)
            clears[info["cleared_lines"]]+=1
            if env.pieces%1000==0:
                milestones.append({"pieces":env.pieces,"score":info["score"]})
            if env.pieces%CHECKPOINT==0 and not (over or capped):
                base=(controller.base if variant=="Rescue" else controller) if controller else None
                payload={"identity":(phase,variant,seed,cap,protocol["models"],protocol["rescue_rule_sha256"]),
                         "env_state":env.get_state(),"control":None if base is None else
                         (base.active,base.last_switch,base.step),
                         "clears":clears,"milestones":milestones,"trace":trace,"events":events,
                         "selected":selected,"proposed":proposed,"rejected":rejected,
                         "reject_reasons":reject_reasons,"inference_seconds":inference_seconds,
                         "elapsed_seconds":time.monotonic()-started}
                atomic_pickle(checkpoint,payload)
                atomic_json(checkpoint.with_suffix(".sha256.json"),{"sha256":digest(checkpoint)})
            if over or capped: break
        return {"phase":phase,"variant":variant,"seed":seed,"cap":cap,
                "model_sha256":protocol["models"],"rule_sha256":protocol["rescue_rule_sha256"],
                "pieces":env.pieces,"score":info["score"],"game_over":bool(over),
                "capped":bool(capped),"clears_0_to_4":clears,"score_milestones":milestones,
                "source_runs":trace,"actions_by_source":selected,
                "v2_calls":proposed,"v2_changed":rejected,"v2_reasons":reject_reasons,
                "intercepts":events,"inference_seconds":inference_seconds,
                "wall_seconds":time.monotonic()-started}
    finally:
        env.close()


def task(args):
    return game(*args)


def validated_row(path,phase,variant,seed,cap,protocol):
    row=json.loads(path.read_text())
    if (row["phase"]!=phase or row["variant"]!=variant or row["seed"]!=seed
            or row["cap"]!=cap or row["model_sha256"]!=protocol["models"]
            or row["rule_sha256"]!=protocol["rescue_rule_sha256"]
            or row["game_over"]==row["capped"] or not 0<row["pieces"]<=cap
            or row["capped"]!=(row["pieces"]==cap and not row["game_over"])
            or sum(row["actions_by_source"].values())!=row["pieces"]):
        raise RuntimeError(f"Completed Rescue game mismatch: {path}")
    seal=path.with_suffix(".sha256.json")
    if seal.exists() and json.loads(seal.read_text())["sha256"]!=digest(path):
        raise RuntimeError(f"Completed Rescue game checksum mismatch: {path}")
    if not seal.exists():
        atomic_json(seal,{"sha256":digest(path),"seed":seed})
    return row


def evaluate(phase,variants,seeds,cap,protocol):
    for variant in variants:
        folder=RUN/phase/variant
        folder.mkdir(parents=True,exist_ok=True)
        pending=[]
        for seed in seeds:
            path=folder/f"seed_{seed}.json"
            if path.exists():
                validated_row(path,phase,variant,seed,cap,protocol)
                checkpoint_file(phase,variant,seed).unlink(missing_ok=True)
                checkpoint_file(phase,variant,seed).with_suffix(".sha256.json").unlink(missing_ok=True)
            else: pending.append(seed)
        status("evaluating",phase=phase,variant=variant,completed=len(seeds)-len(pending),total=len(seeds))
        if pending:
            with ProcessPoolExecutor(max_workers=min(WORKERS,len(pending)),mp_context=get_context("spawn"),
                                     initializer=init_worker) as pool:
                futures={pool.submit(task,(phase,variant,seed,cap,protocol)):seed for seed in pending}
                for future in as_completed(futures):
                    seed=futures[future]
                    path=folder/f"seed_{seed}.json"
                    atomic_json(path,future.result())
                    validated_row(path,phase,variant,seed,cap,protocol)
                    checkpoint_file(phase,variant,seed).unlink(missing_ok=True)
                    checkpoint_file(phase,variant,seed).with_suffix(".sha256.json").unlink(missing_ok=True)
                    status("evaluating",phase=phase,variant=variant,
                           completed=sum((folder/f"seed_{s}.json").exists() for s in seeds),total=len(seeds))
        aggregate_phase(phase,variant,seeds,protocol)


def aggregate_phase(phase,variant,seeds,protocol):
    rows=[validated_row(RUN/phase/variant/f"seed_{s}.json",phase,variant,s,
                        20000 if phase=="development_20k" else 100 if phase=="smoke" else CAP,protocol)
          for s in seeds]
    pieces=[r["pieces"] for r in rows]
    summary={"games":len(rows),"mean_pieces":statistics.mean(pieces),
             "median_pieces":statistics.median(pieces),"p10_pieces":float(np.percentile(pieces,10)),
             "capped":sum(r["capped"] for r in rows),"game_over":sum(r["game_over"] for r in rows),
             "early_under_10000":sum(r["pieces"]<10000 for r in rows),
             "mean_score":statistics.mean(r["score"] for r in rows),
             "score_per_1000":1000*sum(r["score"] for r in rows)/sum(pieces),
             "four_count":sum(r["clears_0_to_4"][4] for r in rows),
             "four_fraction":sum(r["clears_0_to_4"][4] for r in rows)/max(1,sum(sum(r["clears_0_to_4"][1:]) for r in rows)),
             "v2_calls":sum(r["v2_calls"] for r in rows),
             "v2_changed":sum(r["v2_changed"] for r in rows),
             "mean_inference_seconds":statistics.mean(r["inference_seconds"] for r in rows),
             "mean_wall_seconds":statistics.mean(r["wall_seconds"] for r in rows)}
    atomic_json(RUN/phase/variant/"summary.json",summary)
    return summary


def gate(phase,protocol):
    a=json.loads((RUN/phase/"Rescue"/"summary.json").read_text())
    b=json.loads((RUN/phase/"B12"/"summary.json").read_text())
    g=protocol["development_20k_screen" if phase=="development_20k" else "development_100k_gate"]
    checks={"score":a["mean_score"]>=g["min_score_ratio"]*b["mean_score"],
            "four":a["four_fraction"]>=g["min_four_ratio"]*b["four_fraction"],
            "early_death":a["early_under_10000"]<=b["early_under_10000"]+g["max_early_death_excess"]}
    if phase=="development_100k":
        checks["mean_survival"]=a["mean_pieces"]>=g["min_mean_survival_ratio"]*b["mean_pieces"]
        checks["p10"]=a["p10_pieces"]>b["p10_pieces"]
    result={"phase":phase,"checks":checks,"pass":all(checks.values()),"decided_at":now(),
            "rescue_summary":a,"b12_summary":b}
    atomic_json(RUN/f"{phase}_gate.json",result)
    return result


def main():
    try:
        protocol=freeze()
        evaluate("smoke",VARIANTS,SMOKE,100,protocol)
        evaluate("development_20k",VARIANTS,DEV[:16],20000,protocol)
        first=gate("development_20k",protocol)
        if not first["pass"]:
            from training.evaluation.hybrid_b12_v2_rescue_report import write_report
            write_report(protocol,"stopped_after_20k")
            status("completed",outcome="stopped_after_20k")
            return
        evaluate("development_100k",VARIANTS,DEV,CAP,protocol)
        second=gate("development_100k",protocol)
        if not second["pass"]:
            from training.evaluation.hybrid_b12_v2_rescue_report import write_report
            write_report(protocol,"stopped_after_development")
            status("completed",outcome="stopped_after_development")
            return
        seed_audit()
        atomic_json(RUN/"confirmation_freeze.json",{"candidate":"Rescue","models":HASHES,
             "rule_sha256":protocol["rescue_rule_sha256"],"seeds":list(CONFIRM),"frozen_at":now()})
        evaluate("confirmation_100k",VARIANTS,CONFIRM,CAP,protocol)
        from training.evaluation.hybrid_b12_v2_rescue_report import write_report
        write_report(protocol,"confirmation_completed")
        status("completed",outcome="confirmation_completed")
    except Exception as exc:
        status("failed",error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke-only",action="store_true")
    args=parser.parse_args()
    if args.smoke_only:
        protocol=freeze()
        evaluate("smoke",VARIANTS,SMOKE,100,protocol)
    else:
        main()
