"""Train an episode-isolated auxiliary risk model from frozen PPO trajectories."""
import argparse
import gzip
import io
import json
import random

import numpy as np
import torch

from training.evaluation.hybrid_queue import HASHES, PPO, verify_models
from training.imitation_dataset import digest
from training.local_correction_data import TRAIN, VALIDATION, _paths
from training.risk_predictor import RiskPredictor, labels
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now

RUN = ROOT / "runs/ppo30m_risk_predictor"
VERSION = "observed_ppo_action_h1000_h5000_v1"
HORIZONS = (1000, 5000)


def audit():
    verify_models()
    if digest(PPO) != HASHES["ppo"]:
        raise RuntimeError("PPO checkpoint changed")
    if set(TRAIN) & set(VALIDATION):
        raise RuntimeError("Episode split overlaps")
    sources = {}
    for seed in TRAIN + VALIDATION:
        shard, _, metadata = _paths(seed)
        meta = json.loads(metadata.read_text())
        if meta["seed"] != seed or digest(shard) != meta["shard_sha256"]:
            raise RuntimeError(f"Source mismatch: {seed}")
        sources[str(seed)] = meta["shard_sha256"]
    protocol = {"version": VERSION, "ppo_sha256": HASHES["ppo"],
                "train_seeds": TRAIN, "validation_seeds": VALIDATION,
                "source_shards": sources, "horizons": HORIZONS, "seed": 29017,
                "epochs": 20, "patience": 4, "batch_size": 512,
                "learning_rate": 0.001, "selection": "mean validation BCE over observed heads"}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Frozen protocol changed")
    if not path.exists():
        atomic_json(path, protocol)
    return protocol


def dataset(seeds):
    state, action, target, seen, details = [], [], [], [], []
    for seed in seeds:
        shard, _, metadata = _paths(seed)
        meta = json.loads(metadata.read_text())
        with gzip.open(shard, "rt", encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["seed"] != seed or row["ppo_action"] not in row["legal_ids"]:
                    raise RuntimeError("Illegal or mismatched archived action")
                index = row["legal_ids"].index(row["ppo_action"])
                candidate = row["legal_features"][index]
                if len(row["state"]) != 237 or len(candidate) != 16:
                    raise RuntimeError("Feature dimensions changed")
                targets = [labels(row["step"], meta["pieces"], meta["game_over"], h)
                           for h in HORIZONS]
                state.append(row["state"]); action.append(candidate)
                target.append([0 if x is None else x for x in targets])
                seen.append([x is not None for x in targets])
                details.append({"seed": seed, "step": row["step"],
                                "death_distance": meta["pieces"]-row["step"] if meta["game_over"] else None,
                                "height": candidate[8], "holes": candidate[6]})
    return (np.asarray(state, np.float32), np.asarray(action, np.float32),
            np.asarray(target, np.float32), np.asarray(seen, bool), details)


def metrics(logits, targets, mask):
    probabilities = torch.sigmoid(torch.as_tensor(logits)).numpy()
    result = {}
    for j, horizon in enumerate(HORIZONS):
        y, p = targets[mask[:, j], j], probabilities[mask[:, j], j]
        order = np.argsort(-p, kind="stable")
        sorted_y = y[order]
        positive = int(y.sum()); negative = len(y) - positive
        ap = None
        if positive and negative:
            ordered_p = p[order]
            cumulative = np.cumsum(sorted_y)
            ends = np.r_[np.flatnonzero(np.diff(ordered_p)), len(y)-1]
            true_at_ends = cumulative[ends]
            gained = np.diff(np.r_[0, true_at_ends])
            ap = float(np.sum(gained * true_at_ends / (ends+1)) / positive)
        # Mann–Whitney AUC with average ranks for ties.
        auc = None
        if positive and negative:
            ascending = np.argsort(p, kind="stable")
            ranks = np.empty(len(y), dtype=float)
            ordered = p[ascending]
            i = 0
            while i < len(y):
                k = i + 1
                while k < len(y) and ordered[k] == ordered[i]: k += 1
                ranks[ascending[i:k]] = (i + 1 + k) / 2
                i = k
            auc = float((ranks[y == 1].sum() - positive*(positive+1)/2) / (positive*negative))
        result[str(horizon)] = {"n": len(y), "positives": positive,
            "prevalence": float(y.mean()) if len(y) else None,
            "brier": float(np.mean((y-p)**2)) if len(y) else None,
            "pr_auc": ap, "roc_auc": auc}
    return result


def checkpoint(path, model, optimizer, epoch, best, stale):
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best": best, "stale": stale,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(), "protocol_sha256": digest(RUN / "protocol.json")}, stream)
    atomic_bytes(path, stream.getvalue())


def run():
    config = audit()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    train, valid = dataset(TRAIN), dataset(VALIDATION)
    counts = {name: {str(h): {"observed": int(data[3][:, j].sum()),
                              "positive": int(data[2][data[3][:, j], j].sum()),
                              "censored": int((~data[3][:, j]).sum())}
                     for j, h in enumerate(HORIZONS)}
              for name, data in (("train", train), ("validation", valid))}
    atomic_json(RUN / "data_manifest.json", {"version": VERSION, "counts": counts,
                "train_episodes": TRAIN, "validation_episodes": VALIDATION,
                "source_shards": config["source_shards"]})
    model = RiskPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    latest = RUN / "latest.pt"; best_path = RUN / "best.pt"
    start, best, stale = 0, float("inf"), 0
    if latest.exists():
        data = torch.load(latest, map_location="cpu", weights_only=False)
        if data["protocol_sha256"] != digest(RUN / "protocol.json"):
            raise RuntimeError("Checkpoint protocol mismatch")
        model.load_state_dict(data["model"]); optimizer.load_state_dict(data["optimizer"])
        torch.set_rng_state(data["torch_rng"]); np.random.set_state(data["numpy_rng"])
        random.setstate(data["python_rng"])
        start, best, stale = data["epoch"], data["best"], data["stale"]
    def tensors(data):
        return tuple(torch.from_numpy(x).to(device) for x in data[:4])
    tr, va = tensors(train), tensors(valid)
    history = json.loads((RUN / "history.json").read_text()) if (RUN / "history.json").exists() else []
    for epoch in range(start, config["epochs"]):
        if stale >= config["patience"]: break
        model.train(); order = torch.randperm(len(tr[0]), device=device)
        losses = []
        for batch in order.split(config["batch_size"]):
            logits = model(tr[0][batch], tr[1][batch])
            raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, tr[2][batch], reduction="none")
            loss = (raw * tr[3][batch]).sum() / tr[3][batch].sum().clamp(min=1)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            output = model(va[0], va[1]); raw = torch.nn.functional.binary_cross_entropy_with_logits(output, va[2], reduction="none")
            val_loss = float((raw * va[3]).sum() / va[3].sum().clamp(min=1))
            scores = metrics(output.cpu(), valid[2], valid[3])
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            checkpoint(best_path, model, optimizer, epoch + 1, best, stale)
        else: stale += 1
        row = {"epoch": epoch+1, "train_loss": float(np.mean(losses)),
               "validation_loss": val_loss, "metrics": scores, "timestamp": now()}
        history.append(row)
        checkpoint(latest, model, optimizer, epoch+1, best, stale)
        atomic_json(RUN / "history.json", history)
        atomic_json(RUN / "status.json", {"stage": "training", "epoch": epoch+1,
                                         "best_validation_loss": best, "device": device, "updated_at": now()})
    atomic_json(RUN / "status.json", {"stage": "trained_pending_independent_validation",
                "best_validation_loss": best, "best_sha256": digest(best_path),
                "epochs_completed": len(history), "device": device, "updated_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.parse_args(); run()
