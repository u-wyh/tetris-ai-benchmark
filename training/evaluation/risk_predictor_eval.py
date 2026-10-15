"""Held-out episode warning audit; no online control is enabled here."""
import json

import numpy as np
import torch

from training.imitation_dataset import digest
from training.risk_predictor import RiskPredictor
from training.train_risk_predictor import RUN, VALIDATION, dataset, metrics
from training.train_ppo import atomic_json, now


def run():
    state, action, target, observed, rows = dataset(VALIDATION)
    data = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    model = RiskPredictor(); model.load_state_dict(data["model"]); model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(state), torch.from_numpy(action)).numpy()
    network = metrics(logits, target, observed)
    baseline = {}
    for name, values in {"height": action[:, 8],
                         "height_holes": action[:, 8] + 0.2*action[:, 6],
                         "hand_risk": action[:, 8] + 0.2*action[:, 6] +
                                      2*action[:, 11] + 5*action[:, 15]}.items():
        baseline[name] = metrics(np.repeat(values[:, None], 2, axis=1), target, observed)
        for horizon in baseline[name].values():
            horizon.pop("brier")  # Baseline scores are not calibrated probabilities.
    warnings = {}
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()[:, 0]
    calibration = []
    valid_indices = np.flatnonzero(observed[:, 0])
    ordered = valid_indices[np.argsort(probabilities[valid_indices], kind="stable")]
    for group in np.array_split(ordered, 5):
        calibration.append({"n": len(group), "mean_predicted": float(np.mean(probabilities[group])),
                            "observed_death_rate": float(np.mean(target[group, 0]))})
    for lead in (1000, 500, 250):
        indices = [i for i, row in enumerate(rows) if row["death_distance"] is not None
                   and lead <= row["death_distance"] <= lead+100]
        warnings[str(lead)] = {"samples": len(indices), "episodes": len({rows[i]["seed"] for i in indices}),
                               "mean_probability": float(np.mean(probabilities[indices])) if indices else None,
                               "fraction_above_0_1": float(np.mean(probabilities[indices] >= 0.1)) if indices else None}
    safe = [i for i, row in enumerate(rows) if row["death_distance"] is None or row["death_distance"] > 5000]
    result = {"version": "heldout_archived_episodes_v1", "model_epoch": data["epoch"],
              "episodes": VALIDATION, "network": network, "baselines": baseline,
              "warning": warnings, "safe_samples": len(safe),
              "calibration_quintiles_1000": calibration,
              "safe_false_alarm_fraction_at_0_1": float(np.mean(probabilities[safe] >= 0.1)) if safe else None,
              "note": "Archived rows are selected samples, not continuous trajectories; these warning rates are sample-level only.",
              "updated_at": now()}
    atomic_json(RUN / "heldout_evaluation.json", result)
    gate = (network["1000"]["pr_auc"] is not None and
            network["1000"]["pr_auc"] > baseline["height_holes"]["1000"]["pr_auc"] and
            sum(row["death_distance"] is not None for row in rows) > 0)
    atomic_json(RUN / "status.json", {"stage": "risk_model_passed_pending_action_ranking" if gate else "completed_no_go",
                                     "reason": "network_below_height_holes_baseline" if not gate else None,
                                     "best_sha256": digest(RUN / "best.pt"),
                                     "updated_at": now()})
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
