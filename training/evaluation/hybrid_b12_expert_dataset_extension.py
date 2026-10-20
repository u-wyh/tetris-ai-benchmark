"""Prospective B12 trajectory sampling and bounded V2 counterfactual dataset."""

import hashlib
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

from training.candidate_features import board_features, candidate_board
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, _DeterministicAdapter, atomic_pickle
from training.evaluation.hybrid_queue import PPO, BC, HASHES, digest, verify_models
from training.evaluation.ppo_death_capture import public_hash
from training.hybrid_policy import HybridConfig, HybridController
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import V2, V2Config, features as v2_features, simulate_public
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/hybrid_b12_expert_dataset_extension"
REPORT = ROOT / "reports/experiments/hybrid_b12_expert_dataset_extension.md"
TRAIN = tuple(range(1460000, 1460032))
VALID = tuple(range(1461000, 1461016))
RESERVED = tuple(range(1452000, 1452032))
CAP = 100000
HORIZON = 5000
WORKERS = 4
SPACING = 500
SAMPLER = {"A": {"probability": 1., "quota": 4},
           "B": {"probability": .25, "quota": 2},
           "C": {"probability": .2, "quota": 1},
           "D": {"probability": .001, "quota": 1}}
MODELS = None


def audit_seeds():
    targets = set(TRAIN + VALID)
    name_pattern = re.compile(r"(?<!\d)146\d{4}(?!\d)")
    field = re.compile(r'(?:"seed"\s*:\s*|seed[_=]|\bseed\s+)(\d+)\b', re.I)
    list_field = re.compile(r'"(?:seeds|dev|confirm|train|validation|reserved)"\s*:\s*\[([^\]]*)\]')
    hits = []
    for root in (ROOT / "runs", ROOT / "reports", ROOT / "training", ROOT / "tests"):
        for path in root.rglob("*"):
            if not path.is_file() or path.is_relative_to(RUN) or path in (REPORT, Path(__file__)):
                continue
            if any(int(m.group()) in targets for m in name_pattern.finditer(path.name)):
                hits.append(str(path.relative_to(ROOT)))
                continue
            if path.suffix not in (".json", ".md", ".py", ".csv", ".txt") or path.stat().st_size > 2_000_000:
                continue
            content = path.read_text(errors="replace")
            if any(int(m.group(1)) in targets for m in field.finditer(content)) or any(
                    int(n) in targets for m in list_field.finditer(content)
                    for n in re.findall(r"\b\d+\b", m.group(1))):
                hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise RuntimeError(f"Dataset seed conflict: {hits[:20]}")
    return {"train": list(TRAIN), "validation": list(VALID), "reserved_not_run": list(RESERVED),
            "hits": [], "audited_at": now()}


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_models()
    path = RUN / "protocol.json"
    if not path.exists():
        atomic_json(RUN / "seed_audit.json", audit_seeds())
    spec = {"train": list(TRAIN), "validation": list(VALID), "reserved_not_run": list(RESERVED),
            "models": HASHES, "b12_sha256": digest(ROOT / "training/hybrid_policy.py"),
            "v2_sha256": digest(ROOT / "training/traditional/v2.py"),
            "core_sha256": digest(ROOT / "training/tetris_core/core.py"),
            "features_sha256": digest(ROOT / "training/candidate_features.py"),
            "dataset_code_sha256": digest(Path(__file__)),
            "cap": CAP, "branch_horizon": HORIZON, "workers": WORKERS,
            "sampling": {"broader_pool": "height>=15 and bumpiness>=20",
                         "A": "height>=17 and bumpiness>=25, V2 distinct from PPO and BC",
                         "B": "broader pool outside A trigger, V2 distinct from PPO and BC",
                         "C": "A trigger, V2 equals PPO or BC",
                         "D": "height<12 and bumpiness<20, random low-risk distribution control",
                         "minimum_piece_gap": SPACING, "per_seed_quota": SAMPLER,
                         "hash": "sha256(seed:piece:category:expert-dataset-extension-v1), uniform first 64 bits",
                         "selection": "online before executing B12 action, not based on future outcome"},
            "next_safe": "visible next only; no Lock Out, resulting max height <=18, new holes <=1",
            "label": "beneficial if B12 truly dies and V2 survives >=500 more; harmful vice versa; otherwise uncertain",
            "censoring": "5000-piece branch cap is right-censored; episode 100K cap is right-censored"}
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Frozen expert dataset protocol changed")
    if not path.exists():
        atomic_json(path, spec)
    return spec


def status(stage, **extra):
    atomic_json(RUN / "status.json", {"stage": stage, "pid": os.getpid(), "at": now(), **extra})


def sealed_json(path, value=None):
    seal = path.with_suffix(".sha256.json")
    if path.exists():
        if not seal.exists() or json.loads(seal.read_text())["sha256"] != digest(path):
            raise RuntimeError(f"Completed result checksum mismatch: {path}")
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    atomic_json(seal, {"sha256": digest(path)})
    return value


def sample_uniform(seed, piece, category):
    raw = hashlib.sha256(f"{seed}:{piece}:{category}:expert-dataset-extension-v1".encode()).digest()
    return int.from_bytes(raw[:8], "big") / 2**64


def episode_path(seed):
    group = "train" if seed in TRAIN else "validation"
    return RUN / "episodes" / group / f"seed_{seed}.json"


def state_path(seed, piece):
    return RUN / "states" / f"seed_{seed}_step_{piece}.pkl"


def save_state(seed, piece, env, control_state, candidate):
    path = state_path(seed, piece)
    if path.exists():
        seal = json.loads(path.with_suffix(".sha256.json").read_text())
        if seal["sha256"] != digest(path):
            raise RuntimeError("Selected snapshot checksum mismatch")
        return digest(path)
    payload = {"seed": seed, "piece": piece, "protocol_sha256": digest(RUN / "protocol.json"),
               "core_state": env.get_state(), "control": control_state,
               "public_hash": public_hash(env.core.get_public_observation()),
               "actions": candidate["actions"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle(path, payload)
    atomic_json(path.with_suffix(".sha256.json"), {"sha256": digest(path)})
    return digest(path)


def load_state(seed, piece):
    path = state_path(seed, piece)
    seal = json.loads(path.with_suffix(".sha256.json").read_text())
    if seal["sha256"] != digest(path):
        raise RuntimeError("Selected snapshot checksum mismatch")
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if (saved["seed"] != seed or saved["piece"] != piece or
            saved["protocol_sha256"] != digest(RUN / "protocol.json")):
        raise RuntimeError("Selected snapshot protocol mismatch")
    return saved


def init_worker():
    global MODELS
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    MODELS = {"ppo": MaskablePPO.load(str(PPO), device="cpu"),
              "bc": MaskablePPO.load(str(BC), device="cpu")}
    for model in MODELS.values():
        if (model.policy.__class__.__name__ != "CandidateScoringPolicy" or model.action_space.n != 1840
                or model.observation_space["state"].shape != (237,)
                or model.observation_space["candidate"].shape != (1840, 16)):
            raise RuntimeError("Frozen expert model spaces changed")


def make_b12():
    return HybridController(_DeterministicAdapter(MODELS["ppo"]),
                            _DeterministicAdapter(MODELS["bc"]), HybridConfig("B", 12, 8))


def classify_state(seed, piece, public, placements, decision, v2, metrics, selected_d):
    """Return a pre-outcome sampling record or None; mutate only selection counters."""
    _, _, height, bump, _ = metrics
    trigger = height >= 17 and bump >= 25
    broad = height >= 15 and bump >= 20
    low = height < 12 and bump < 20
    if not broad and not low:
        return None
    if low:
        category = "D"
        selected_d["eligible"][category] += 1
        u = sample_uniform(seed, piece, category)
        if u >= SAMPLER[category]["probability"]:
            return None  # Do not run V2 in every ordinary state.
        v2_action = int(v2.choose(public, placements))
    else:
        v2_action = int(v2.choose(public, placements))
        independent = v2_action not in (decision["ppo_action"], decision["bc_action"])
        category = ("A" if trigger else "B") if independent else ("C" if trigger else None)
        if category is None:
            return None
        u = sample_uniform(seed, piece, category)
    same = v2_action == decision["ppo_action"] or v2_action == decision["bc_action"]
    record = {"category": category, "piece": piece,
              "uniform": u, "base_probability": SAMPLER[category]["probability"],
              "independent_v2": not same,
              "actions": {"ppo": decision["ppo_action"], "bc": decision["bc_action"],
                          "b12": decision["ppo_action"] if decision["source"] == "ppo" else decision["bc_action"],
                          "v2": v2_action}, "b12_source": decision["source"],
              "b12_reason": decision["reason"]}
    if category != "D":
        selected_d["eligible"][category] += 1
    if selected_d["counts"][category] >= SAMPLER[category]["quota"] or (
            selected_d["last_step"] >= 0 and piece - selected_d["last_step"] < SPACING):
        return None
    if u >= SAMPLER[category]["probability"]:
        return None
    selected_d["counts"][category] += 1
    selected_d["last_step"] = piece
    return record


def candidate(public, placements, row):
    by_id = {p["actionId"]: p for p in placements}
    holes, total, height, bump, wells = board_features(public["board"])
    surface = v2_features(public["board"])
    top = sum(public["board"][y][x] is not None for y in range(4) for x in range(3, 7))
    actions = {}
    for name, action in row["actions"].items():
        if action not in by_id:
            raise RuntimeError("Selected expert action not legal")
        placement = by_id[action]
        board, cleared, lockout = candidate_board(public["board"], placement["occupiedCells"])
        after_holes, after_total, after_height, after_bump, after_wells = board_features(board)
        visible, _ = simulate_public(public, placement)
        next_legal = next_safe = 0
        if visible["phase"] == "playing":
            following = get_legal_placements(visible)
            next_legal = len(following)
            for move in following:
                child, _, dead = candidate_board(visible["board"], move["occupiedCells"])
                child_holes, _, child_height, _, _ = board_features(child)
                next_safe += not dead and child_height <= 18 and child_holes <= after_holes + 1
        actions[name] = {"action": action, "uses_hold": bool(placement["hold"]),
                         "height_after": after_height, "aggregate_after": after_total,
                         "holes_after": after_holes, "new_holes": after_holes - holes,
                         "bump_after": after_bump, "wells_after": after_wells,
                         "cleared": cleared, "lock_out": bool(lockout),
                         "next_visible_legal": next_legal, "next_visible_safe": next_safe,
                         "visible_spawn_failed": visible["phase"] == "over",
                         "visible_phase": visible["phase"]}
    return {"before": {"height": height, "aggregate_height": total, "holes": holes,
                       "bumpiness": bump, "wells": wells,
                       "row_transitions": surface["row_transitions"],
                       "column_transitions": surface["column_transitions"],
                       "central_top4": top, "hold": public["hold"], "next3": public["next"],
                       "legal_actions": len(placements)},
            "actions": actions,
            "v2_minus_b12_safe": actions["v2"]["next_visible_safe"] - actions["b12"]["next_visible_safe"]}


def collect(seed):
    path = episode_path(seed)
    if path.exists():
        row = sealed_json(path)
        if row["protocol_sha256"] != digest(RUN / "protocol.json"):
            raise RuntimeError("Completed episode protocol mismatch")
        for item in row["selected"]:
            if digest(state_path(seed, item["piece"])) != item["snapshot_sha256"]:
                raise RuntimeError("Completed episode snapshot differs")
        return {k: row[k] for k in ("seed", "pieces", "game_over", "capped")}
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    control = make_b12()
    v2 = V2(V2Config(mode="hold"))
    selected_d = {"counts": {c: 0 for c in SAMPLER},
                  "eligible": {c: 0 for c in SAMPLER}, "last_step": -1}
    selected = []
    started = time.monotonic()
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            piece = env.pieces
            public = env.core.get_public_observation()
            mask = env.action_masks()
            placements = env._placement_cache
            before_control = (control.active, control.last_switch, control.step)
            b12_action, decision = control.decide(obs, mask, public, placements)
            if not mask[b12_action]:
                raise RuntimeError("B12 proposed illegal action")
            metrics = board_features(public["board"])
            sampled = classify_state(seed, piece, public, placements, decision, v2, metrics, selected_d)
            if sampled is not None:
                if not mask[sampled["actions"]["v2"]]:
                    raise RuntimeError("V2 proposed illegal action")
                item = {"seed": seed, **sampled,
                        "public_hash": public_hash(public),
                        "features": candidate(public, placements, sampled)}
                item["snapshot_sha256"] = save_state(seed, piece, env, before_control, sampled)
                selected.append(item)
            obs, _, over, capped, info = env.step(b12_action)
            if over or capped:
                break
        row = {"seed": seed, "group": "train" if seed in TRAIN else "validation",
               "protocol_sha256": digest(RUN / "protocol.json"),
               "pieces": info["pieces"], "score": info["score"],
               "game_over": bool(over), "capped": bool(capped),
               "selected": selected, "eligible_counts": selected_d["eligible"],
               "selected_counts": selected_d["counts"],
               "wall_seconds": time.monotonic()-started}
        sealed_json(path, row)
        return {k: row[k] for k in ("seed", "pieces", "game_over", "capped")}
    finally:
        env.close()


def branch(task):
    seed, piece, action_name = task
    path = RUN / "branches" / f"seed_{seed}_step_{piece}_{action_name}.json"
    if path.exists():
        row = sealed_json(path)
        if row["snapshot_sha256"] != digest(state_path(seed, piece)):
            raise RuntimeError("Completed branch snapshot mismatch")
        return {"seed": seed, "piece": piece, "role": action_name}
    saved = load_state(seed, piece)
    ep = sealed_json(episode_path(seed))
    item = next(x for x in ep["selected"] if x["piece"] == piece)
    env = LongCandidateTetrisEnv(max_pieces=max(CAP, piece + HORIZON + 1))
    env.reset(seed=seed)
    state = saved["core_state"].copy()
    state["max_pieces"] = max(CAP, piece + HORIZON + 1)
    env.set_state(state)
    obs = env.get_observation()
    control = make_b12()
    control.active, control.last_switch, control.step = saved["control"]
    score0 = env.core.score
    clears = [0] * 5
    scores = []
    started = time.monotonic()
    try:
        if public_hash(env.core.get_public_observation()) != saved["public_hash"]:
            raise RuntimeError("Counterfactual start state mismatch")
        selected_first = item["actions"][action_name]
        for offset in range(HORIZON):
            mask = env.action_masks()
            proposed, _ = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            action = selected_first if offset == 0 else proposed
            if offset == 0 and action_name == "b12" and proposed != action:
                raise RuntimeError("Frozen B12 first action changed")
            if not mask[action]:
                raise RuntimeError("Counterfactual action illegal")
            obs, _, over, capped, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            scores.append(info["score"] - score0)
            if over or capped:
                break
        public = env.core.get_public_observation()
        holes, _, height, bump, _ = board_features(public["board"])
        row = {"seed": seed, "piece": piece, "role": action_name,
               "snapshot_sha256": digest(state_path(seed, piece)),
               "first_action": selected_first, "pieces_after": env.pieces - piece,
               "game_over": bool(over), "censored": not over,
               "score_gain": env.core.score - score0, "score_trace": scores,
               "score_per_1000": 1000 * (env.core.score - score0) / (env.pieces - piece),
               "lines_cleared": sum(i * n for i, n in enumerate(clears)),
               "clears_0_to_4": clears, "final_height": height, "final_holes": holes,
               "final_bumpiness": bump, "wall_seconds": time.monotonic()-started}
        sealed_json(path, row)
        return {"seed": seed, "piece": piece, "role": action_name}
    finally:
        env.close()


def classify_pair(a, b):
    difference = b["pieces_after"] - a["pieces_after"]
    if a["game_over"] and difference >= 500:
        return "beneficial"
    if b["game_over"] and difference <= -500:
        return "harmful"
    return "uncertain"


def dataset_row(seed, item):
    piece = item["piece"]
    a = sealed_json(RUN / "branches" / f"seed_{seed}_step_{piece}_b12.json")
    b = sealed_json(RUN / "branches" / f"seed_{seed}_step_{piece}_v2.json")
    shared = min(a["pieces_after"], b["pieces_after"])
    common_a = a["score_trace"][shared - 1]
    common_b = b["score_trace"][shared - 1]
    return {"seed": seed, "group": "train" if seed in TRAIN else "validation",
            "piece": piece, "category": item["category"],
            "features": item["features"], "sampling_probability": item["base_probability"],
            "hash_uniform": item["uniform"], "snapshot_sha256": item["snapshot_sha256"],
            "b12_source": item["b12_source"], "b12_reason": item["b12_reason"],
            "label": classify_pair(a, b),
            "survival_difference": b["pieces_after"] - a["pieces_after"],
            "common_window_pieces": shared, "common_score_b12": common_a,
            "common_score_v2": common_b,
            "common_score_difference": common_b - common_a,
            "branches": {"b12": {k: v for k, v in a.items() if k != "score_trace"},
                         "v2": {k: v for k, v in b.items() if k != "score_trace"}}}


def paired_interval_by_seed(rows, values, samples=4000):
    seeds = sorted({r["seed"] for r in rows})
    if not seeds:
        return None
    by_seed = {s: [values(r) for r in rows if r["seed"] == s] for s in seeds}
    means = np.asarray([statistics.mean(by_seed[s]) for s in seeds], dtype=float)
    rng = np.random.default_rng(1463001)
    indices = rng.integers(0, len(seeds), size=(samples, len(seeds)))
    return [float(x) for x in np.quantile(means[indices].mean(axis=1), [.025, .975])]


def summarize(group, rows, all_selected):
    selected = [r for r in rows if r["group"] == group]
    sampled = [r for r in all_selected if r["group"] == group]
    independent = [r for r in selected if r["category"] in ("A", "B")]
    labels = {name: [r for r in independent if r["label"] == name]
              for name in ("beneficial", "harmful", "uncertain")}
    mobility = {"positive": [r for r in independent if r["features"]["v2_minus_b12_safe"] > 0],
                "zero_or_negative": [r for r in independent if r["features"]["v2_minus_b12_safe"] <= 0]}
    return {"episodes": 32 if group == "train" else 16,
            "selected_total": len(sampled), "different_action_states": len(selected),
            "independent_states": len(independent),
            "independent_seeds": len({r["seed"] for r in independent}),
            "category_counts": {c: sum(r["category"] == c for r in sampled) for c in SAMPLER},
            "labels": {k: {"states": len(v), "seeds": len({r["seed"] for r in v})}
                       for k, v in labels.items()},
            "mobility_groups": {k: {"states": len(v),
                                    "beneficial": sum(r["label"] == "beneficial" for r in v),
                                    "harmful": sum(r["label"] == "harmful" for r in v),
                                    "uncertain": sum(r["label"] == "uncertain" for r in v)}
                                for k, v in mobility.items()},
            "survival_difference_seed_mean_ci95": paired_interval_by_seed(
                independent, lambda r: r["survival_difference"]),
            "positive_safe_benefit_rate_seed_ci95": paired_interval_by_seed(
                mobility["positive"], lambda r: float(r["label"] == "beneficial")),
            "near_death_b12_100": sum(r["branches"]["b12"]["game_over"] and
                                      r["branches"]["b12"]["pieces_after"] <= 100 for r in independent)}


def analyze():
    rows = []
    all_selected = []
    for seed in TRAIN + VALID:
        ep = sealed_json(episode_path(seed))
        for item in ep["selected"]:
            all_selected.append({"seed": seed, "group": ep["group"],
                                 "piece": item["piece"], "category": item["category"],
                                 "same_b12": item["actions"]["b12"] == item["actions"]["v2"]})
            if item["actions"]["b12"] != item["actions"]["v2"]:
                rows.append(dataset_row(seed, item))
    record = {"rows": rows, "sampled_index": all_selected,
              "train": summarize("train", rows, all_selected),
              "validation": summarize("validation", rows, all_selected), "created_at": now()}
    sealed_json(RUN / "analysis.json", record)
    write_report(record)


def write_report(data):
    lines = ["# Hybrid B12 三专家危险决策数据集扩充", "",
             "本轮只在冻结 B12 的真实轨迹上**落子前**抽样并运行固定 PPO、BC、V2；不训练、不修改策略。新增采集组 1460000..1460031 与新增探索验证组 1461000..1461015 分开分析。原预留正式确认组 1452000..1452031 未运行。", "",
             "## 事前协议与数据完整性", "",
             "32＋16 局各最多 100K。A/B/C/D 的高度与凹凸条件、固定 SHA256 抽样概率、每 seed 类别配额及 500 块最小间隔均在 `runs/hybrid_b12_expert_dataset_extension/protocol.json` 事前冻结。为增加少数高危 A 类状态的观察机会，事前将 A 改为每局最多 4 个、概率 1；B 限为最多 2 个、概率 0.25；C 为 0.2、最多 1 个；D 为 0.001、最多 1 个。配额已满或步距不足时入选概率为 0。D 先抽签再调用 V2。只在落子前使用公开棋盘、Hold、Next、正式掩码；真实 Core/RNG 仅用于对局执行及相同起点反事实。扩充组的采样分布与上一批不同，不能直接合并作无偏效果估计。", "",
             "入选状态保存完整 Core/RNG 快照，未入选状态不保存。每局和每条分支原子提交并 SHA256 校验。5000 块触顶是右截断；生存差必须至少 500 块且较短分支确实死亡，才标为明显有益或有害。", "",
             "## 分组结果", "",
             "| 组别 | 完整 episode | 独立 V2 状态/seed | A/B/C/D 入选 | 有益状态/seed | 有害状态/seed | 不确定状态/seed | 安全落点增加：有益/有害/不确定 | 未增加：有益/有害/不确定 |",
             "|---|---:|---:|---|---|---|---|---|---|"]
    for group in ("train", "validation"):
        s = data[group]
        cat = "/".join(str(s["category_counts"][x]) for x in ("A", "B", "C", "D"))
        lab = [s["labels"][x] for x in ("beneficial", "harmful", "uncertain")]
        label_text = [f"{x['states']}/{x['seeds']}" for x in lab]
        mobility = []
        for key in ("positive", "zero_or_negative"):
            x = s["mobility_groups"][key]
            mobility.append(f"{x['beneficial']}/{x['harmful']}/{x['uncertain']}")
        lines.append(f"| {group} | {s['episodes']} | {s['independent_states']}/{s['independent_seeds']} | {cat} | "
                     f"{' | '.join(label_text)} | {' | '.join(mobility)} |")
    for group in ("train", "validation"):
        s = data[group]
        lines.append(f"\n{group}：总入选 {s['selected_total']}，其中不同动作、已运行反事实 {s['different_action_states']}。独立 V2 状态按 seed 汇总的平均短窗存活差 Bootstrap 95% 区间 {s['survival_difference_seed_mean_ci95']}；安全落点增加组的有益率按 seed Bootstrap 区间 {s['positive_safe_benefit_rate_seed_ci95']}。原 B12 分支在 100 块内真正死亡的独立状态 {s['near_death_b12_100']} 个（仅用于事后分层，不作在线特征）。\n")
    lines += ["## 公开特征与反例", "",
              "以下反例表只展示从落子前计算的可见下一块安全落点差与有限窗口标签。预测特征不能包含最终死亡时间。完整候选特征、四种动作的 Hold、消行、即时棋盘变化、下一块合法/安全落点和同状态分支得分保存在实验 JSON。", "",
              "| 组别 | Seed:步 | 类别 | 安全落点差 | B12/V2 续存 | 标签 |",
              "|---|---|---|---:|---|---|"]
    exceptions = [r for r in data["rows"] if
                  (r["features"]["v2_minus_b12_safe"] > 0 and r["label"] == "harmful") or
                  (r["features"]["v2_minus_b12_safe"] < 0 and r["label"] == "beneficial")]
    for r in exceptions[:30]:
        a, b = r["branches"]["b12"], r["branches"]["v2"]
        fmt = lambda z: f"{z['pieces_after']}{'+' if z['censored'] else ''}"
        lines.append(f"| {r['group']} | {r['seed']}:{r['piece']} | {r['category']} | "
                     f"{r['features']['v2_minus_b12_safe']:+d} | {fmt(a)}/{fmt(b)} | {r['label']} |")
    if not exceptions:
        lines.append("| — | — | — | — | — | 本次未采到方向相反的明确反例；不代表不存在 |")
    lines += ["", "## 标签质量与下一步", "",
              "同一 seed 的多个状态高度相关；计数同时报告状态与独立 seed，区间按 seed 重采样。采样条件和类配额改变了状态分布，不能从本样本直接推断所有对局决策的比例，也不能只用表面分类准确率评价特征。触顶两分支均为不确定，绝不当作真实寿命相等。", "",
              "若有益或有害标签及覆盖的独立 episode 不足，下一步应按相同预注册协议增加**新的**训练候选与探索验证 seed，并保持未来正式确认种子完全未动；优先提高少数标签的独立 seed 覆盖，而不是立即训练选择器或调整现有 B12/Rescue 阈值。", ""]
    REPORT.write_text("\n".join(lines))


def parallel(stage, tasks, func):
    status(stage, completed=0, total=len(tasks))
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn"), initializer=init_worker) as pool:
        futures = {pool.submit(func, x): x for x in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            status(stage, completed=count, total=len(tasks))


def main():
    try:
        freeze()
        parallel("collect_train", TRAIN, collect)
        parallel("collect_validation", VALID, collect)
        tasks = []
        for seed in TRAIN + VALID:
            ep = sealed_json(episode_path(seed))
            for item in ep["selected"]:
                if item["actions"]["b12"] != item["actions"]["v2"]:
                    tasks += [(seed, item["piece"], name) for name in ("b12", "v2")]
        atomic_json(RUN / "branch_plan.json", {"tasks": tasks, "states": len(tasks)//2})
        parallel("counterfactual_5000", tasks, branch)
        analyze()
        status("completed", episodes=len(TRAIN)+len(VALID), states=len(tasks)//2,
               branches=len(tasks))
    except Exception as exc:
        status("failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("freeze", "main", "analyze"), default="main")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "analyze":
        analyze()
    else:
        main()
