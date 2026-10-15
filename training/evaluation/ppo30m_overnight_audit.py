"""Public-state policy-retention audit; counterfactual labels are never training input."""
import gzip
import json
import statistics

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.evaluation.hybrid_queue import PPO
from training.evaluation.ppo30m_overnight_eval import RUN
from training.imitation_bc import batch
from training.hybrid_policy import F
from training.local_correction_data import TRAIN, _paths
from training.train_ppo import atomic_json, now


def _normal_rows():
    rows = []
    for seed in TRAIN:
        shard, _, _ = _paths(seed)
        with gzip.open(shard,"rt",encoding="utf8") as stream:
            rows.extend(row for row in map(json.loads,stream) if row["category"] == "normal")
    rows.sort(key=lambda r:(r["seed"],r["step"]))
    by_seed = {}
    for row in rows:
        selected = by_seed.setdefault(row["seed"],[])
        if len(selected)<4: selected.append(row)
    return [row for rows in by_seed.values() for row in rows]


def _distribution(model, row):
    obs, mask, _ = batch([{**row,"action":row["ppo_action"]}],"cpu")
    with torch.no_grad():
        probs = model.policy.get_distribution(obs,action_masks=mask).distribution.probs[0].cpu().numpy()
    return probs


def audit(model_path, label):
    torch.set_num_threads(1)
    original = MaskablePPO.load(str(PPO),device="cpu")
    candidate = MaskablePPO.load(str(model_path),device="cpu")
    rows = _normal_rows()
    same, top5, hold, kls = [],[],[],[]
    for row in rows:
        legal = np.asarray(row["legal_ids"],dtype=int)
        p = _distribution(original,row)
        q = _distribution(candidate,row)
        p_rank = legal[np.argsort(-p[legal])]
        q_action = int(legal[np.argmax(q[legal])])
        same.append(q_action == row["ppo_action"])
        top5.append(q_action in set(map(int,p_rank[:5])))
        feature = dict(zip(row["legal_ids"],row["legal_features"]))
        hold.append(feature[q_action][F["hold"]] != feature[row["ppo_action"]][F["hold"]])
        mask = p[legal] > 0
        kls.append(float(np.sum(q[legal][mask]*(np.log(np.maximum(q[legal][mask],1e-12))-
                                                     np.log(np.maximum(p[legal][mask],1e-12))))))
    outcome = {"label":label,"states":len(rows),"normal_top1_agreement":statistics.mean(same),
               "normal_top5_retention":statistics.mean(top5),"hold_difference":statistics.mean(hold),
               "masked_kl_to_ppo":statistics.mean(kls),"completed_at":now()}
    atomic_json(RUN/"audits"/f"{label}.json",outcome)
    return outcome
