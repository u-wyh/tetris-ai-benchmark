"""Read-only, resumable B12 replay and same-state intervention study."""

import json
import os
import pickle
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from training.candidate_features import board_features
from training.env.failure_curriculum_env import file_sha256
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, _DeterministicAdapter, atomic_pickle, fast_action
from training.evaluation.hybrid_queue import BC, PPO, HASHES, digest, verify_models
from training.evaluation.ppo_death_capture import public_hash
from training.hybrid_policy import HybridConfig, HybridController
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/hybrid_b12_death_study"
SOURCE = ROOT / "runs/ppo30m_independent_confirmation/evaluation/B12"
REPORT = ROOT / "reports/experiments/hybrid_b12_death_study.md"
DEATHS = (1421001, 1421005, 1421007, 1421009, 1421010, 1421013, 1421014,
          1421016, 1421023, 1421024, 1421025, 1421027, 1421029)
CONTROLS = (1421000, 1421002, 1421003)
SEEDS = DEATHS + CONTROLS
WORKERS = 4
CAP = 100000
HORIZON = 5000
CHECKPOINT = 5000
MODELS = None


def identity():
    return {"ppo": digest(PPO), "bc": digest(BC),
            "core": digest(ROOT / "training/tetris_core/core.py")}


def old_row(seed):
    path = SOURCE / f"seed_{seed}.json"
    row = json.loads(path.read_text())
    if (row["seed"] != seed or row["cap"] != CAP or row["variant"] != "B12"
            or row["model_sha256"]["ppo"] != HASHES["ppo"]
            or row["model_sha256"]["bc"] != HASHES["bc"]
            or row["game_over"] == row["capped"]):
        raise RuntimeError(f"Source protocol mismatch: {path}")
    return row


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_models()
    for seed in SEEDS:
        row = old_row(seed)
        if (seed in DEATHS) != row["game_over"]:
            raise RuntimeError(f"Wrong death/control classification: {seed}")
    protocol = {"source": "ppo30m_independent_confirmation/evaluation/B12",
                "source_sha256": {str(seed): file_sha256(SOURCE / f"seed_{seed}.json") for seed in SEEDS},
                "model_sha256": identity(), "death_seeds": list(DEATHS),
                "control_seeds": list(CONTROLS), "cap": CAP, "horizon": HORIZON,
                "checkpoint_interval": CHECKPOINT, "workers": WORKERS,
                "selection": "first_different_bc_intercept; nearest_different_bc_intercept_to_first_height_15_and_18_within_100; last_different_bc_intercept_within_500_of_death; unique_max_4",
                "evidence_threshold": "paired_survival_difference_at_least_500_pieces_within_5000_window_or_one_branch_dead_and_other_censored",
                "inference": "frozen_deterministic_fast_candidate_argmax; public_observation_and_official_mask",
                "right_censoring": "replay_at_100k_and_branch_at_5000",
                "historical_exclusions": ["700017:329", "700026:268"],
                "started_at": now()}
    path = RUN / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k:v for k,v in old.items() if k != "started_at"} != {k:v for k,v in protocol.items() if k != "started_at"}:
            raise RuntimeError("Frozen study protocol changed")
        protocol = old
    else:
        atomic_json(path, protocol)
    return protocol


def status(stage, **extra):
    atomic_json(RUN / "status.json", {"stage": stage, "at": now(), "pid": os.getpid(), **extra})


def load_worker():
    global MODELS
    from sb3_contrib import MaskablePPO
    import torch
    torch.set_num_threads(1)
    MODELS = {"ppo": MaskablePPO.load(str(PPO), device="cpu"),
              "bc": MaskablePPO.load(str(BC), device="cpu")}
    for model in MODELS.values():
        if (model.policy.__class__.__name__ != "CandidateScoringPolicy"
                or model.action_space.n != 1840
                or model.observation_space["state"].shape != (237,)
                or model.observation_space["candidate"].shape != (1840, 16)):
            raise RuntimeError("Frozen model spaces changed")


def controller():
    return HybridController(_DeterministicAdapter(MODELS["ppo"]),
                            _DeterministicAdapter(MODELS["bc"]), HybridConfig("B", 12, 8))


def new_env(seed, cap=CAP):
    env = LongCandidateTetrisEnv(max_pieces=cap)
    obs, _ = env.reset(seed=seed)
    return env, obs


def event(index, public, obs, decision, action, placements):
    by_id = {p["actionId"]: p for p in placements}
    p, b = decision["ppo_action"], decision["bc_action"]
    return {"step": index, "source": decision["source"], "reason": decision["reason"],
            "actual_action": action, "ppo_action": p, "bc_action": b,
            "different": p != b, "ppo_hold": bool(by_id[p]["hold"]),
            "bc_hold": bool(by_id[b]["hold"]), "hold": public["hold"],
            "next3": public["next"], "ppo_risk": decision["ppo_risk"],
            "bc_risk": decision["bc_risk"], "public_hash": public_hash(public)}


def checkpoint_path(seed, piece):
    return RUN / "checkpoints" / f"seed_{seed}" / f"piece_{piece:06d}.pkl"


def save_checkpoint(seed, env, control, data, protocol):
    path = checkpoint_path(seed, env.pieces)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "piece": env.pieces, "source_sha256": protocol["source_sha256"][str(seed)],
               "model_sha256": protocol["model_sha256"], "env_state": env.get_state(),
               "control": (control.active, control.last_switch, control.step), "data": data}
    atomic_pickle(path, payload)
    atomic_json(path.with_suffix(".sha256.json"), {"sha256": digest(path), "piece": env.pieces})


def load_checkpoint(seed, path, protocol):
    seal = json.loads(path.with_suffix(".sha256.json").read_text())
    if digest(path) != seal["sha256"]:
        raise RuntimeError(f"Snapshot checksum mismatch: {path}")
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if (saved["seed"] != seed or saved["piece"] != seal["piece"]
            or saved["source_sha256"] != protocol["source_sha256"][str(seed)]
            or saved["model_sha256"] != protocol["model_sha256"]):
        raise RuntimeError(f"Snapshot identity mismatch: {path}")
    return saved


def board_metric(env, info, decision, action):
    public = env.core.get_public_observation()
    holes, _, height, bump, _ = board_features(public["board"])
    central = sum(public["board"][y][x] is not None for y in range(4) for x in range(3,7))
    return {"piece": info["pieces"], "height": height, "holes": holes,
            "bumpiness": bump, "central_top4": central, "legal_next": len(env._placement_cache or []),
            "hold": public["hold"], "cleared": info["cleared_lines"],
            "source": decision["source"], "action": action, "score": info["score"]}


def replay(seed, protocol):
    result_path = RUN / "replays" / f"seed_{seed}.json"
    old = old_row(seed)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result["source_sha256"] != protocol["source_sha256"][str(seed)]
                or result["model_sha256"] != protocol["model_sha256"]
                or result["pieces"] != old["pieces"] or result["score"] != old["score"]):
            raise RuntimeError(f"Completed replay mismatch: {seed}")
        return result
    env, obs = new_env(seed)
    control = controller()
    data = {"metrics": [], "events": [], "bc_actions": 0, "different_bc_actions": 0,
            "four_clears": 0, "lines": 0, "first_height15": None, "first_height18": None,
            "last_four_piece": None, "max_gap_four": 0, "source_counts": {"ppo":0,"bc":0}}
    # Resuming retains only the latest 5000 metrics; enough for all requested death windows.
    checkpoints = sorted((RUN / "checkpoints" / f"seed_{seed}").glob("piece_*.pkl"))
    if checkpoints:
        saved = load_checkpoint(seed, checkpoints[-1], protocol)
        env.set_state(saved["env_state"])
        control.active, control.last_switch, control.step = saved["control"]
        data = saved["data"]
        obs = env.get_observation()
    started = time.monotonic()
    try:
        while True:
            index = env.pieces
            public = env.core.get_public_observation()
            mask = env.action_masks()
            placements = env._placement_cache
            if index < 100 or index % 1000 == 0:
                if placements != env.core.get_legal_placements():
                    raise RuntimeError(f"Fast legal placement mismatch: {seed}:{index}")
            action, decision = control.decide(obs, mask, public, placements)
            if not mask[action]:
                raise RuntimeError(f"Illegal B12 action: {seed}:{index}")
            data["source_counts"][decision["source"]] += 1
            if decision["source"] == "bc":
                data["bc_actions"] += 1
                if decision["ppo_action"] != decision["bc_action"]:
                    data["different_bc_actions"] += 1
                data["events"].append(event(index, public, obs, decision, action, placements))
            obs, _, over, capped, info = env.step(action)
            # The new observation populates the official mask and next-step legal cache.
            metric = board_metric(env, info, decision, action)
            data["metrics"].append(metric)
            if len(data["metrics"]) > 5000:
                data["metrics"].pop(0)
            if metric["height"] >= 15 and data["first_height15"] is None:
                data["first_height15"] = info["pieces"]
            if metric["height"] >= 18 and data["first_height18"] is None:
                data["first_height18"] = info["pieces"]
            if info["cleared_lines"] == 4:
                data["four_clears"] += 1
                data["last_four_piece"] = info["pieces"]
            elif data["last_four_piece"] is not None:
                data["max_gap_four"] = max(data["max_gap_four"], info["pieces"]-data["last_four_piece"])
            data["lines"] += info["cleared_lines"]
            if info["pieces"] % CHECKPOINT == 0 and not (over or capped):
                save_checkpoint(seed, env, control, data, protocol)
            if over or capped:
                break
        if (info["pieces"] != old["pieces"] or info["score"] != old["score"]
                or bool(over) != old["game_over"] or bool(capped) != old["capped"]
                or data["four_clears"] != old["clears_0_to_4"][4]):
            raise RuntimeError(f"Frozen B12 replay differs from confirmation: seed={seed} got={info['pieces']}/{info['score']} expected={old['pieces']}/{old['score']}")
        result = {"seed": seed, "source_sha256": protocol["source_sha256"][str(seed)],
                  "model_sha256": protocol["model_sha256"], "verified": True,
                  "pieces": info["pieces"], "score": info["score"],
                  "game_over": bool(over), "capped": bool(capped), "data": data,
                  "wall_seconds_this_attempt": time.monotonic()-started, "completed_at": now()}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(result_path, result)
        return result
    finally:
        env.close()


def select_states(replays, protocol):
    path = RUN / "selection.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["model_sha256"] != protocol["model_sha256"]
                or len(saved["states"]) > 4*len(DEATHS)
                or any(i["seed"] not in DEATHS or i["actual_action"] == i["alternate_action"]
                       or i["source_sha256"] != protocol["source_sha256"][str(i["seed"])]
                       for i in saved["states"])):
            raise RuntimeError("Frozen counterfactual selection changed")
        return saved
    chosen = []
    for seed in DEATHS:
        r = replays[seed]
        different = [e for e in r["data"]["events"] if e["different"]]
        candidates = []
        if different:
            candidates.append(("first", different[0]))
        for label, key in (("height15", "first_height15"), ("height18", "first_height18")):
            crossing = r["data"][key]
            if crossing is not None and different:
                near = min(different, key=lambda e:(abs(e["step"]-crossing), e["step"]))
                if abs(near["step"]-crossing) <= 100:
                    candidates.append((label, near))
        late = [e for e in different if e["step"] >= r["pieces"]-500]
        if late:
            candidates.append(("late", late[-1]))
        used = set()
        for label, e in candidates:
            if e["step"] in used:
                continue
            used.add(e["step"])
            chosen.append({"seed": seed, "step": e["step"], "label": label,
                           "public_hash": e["public_hash"], "actual_action": e["actual_action"],
                           "alternate_action": e["ppo_action"] if e["source"] == "bc" else e["bc_action"],
                           "source_sha256": r["source_sha256"]})
            if len(used) == 4:
                break
    record = {"model_sha256": protocol["model_sha256"], "states": chosen,
              "selection_before_branch_results": True, "selected_at": now()}
    atomic_json(path, record)
    return record


def materialize(item, protocol):
    seed, step = item["seed"], item["step"]
    path = RUN / "states" / f"seed_{seed}_step_{step}.pkl"
    if path.exists():
        saved = load_checkpoint(seed, path, protocol)
        if saved["piece"] != step or saved["public_hash"] != item["public_hash"]:
            raise RuntimeError(f"Selected state mismatch: {path}")
        return path
    prior = sorted((RUN / "checkpoints" / f"seed_{seed}").glob("piece_*.pkl"))
    prior = [p for p in prior if int(p.stem.split("_")[-1]) <= step]
    env, obs = new_env(seed)
    control = controller()
    try:
        if prior:
            saved = load_checkpoint(seed, prior[-1], protocol)
            env.set_state(saved["env_state"])
            control.active, control.last_switch, control.step = saved["control"]
            obs = env.get_observation()
        while env.pieces < step:
            public = env.core.get_public_observation()
            mask = env.action_masks()
            action, _ = control.decide(obs, mask, public, env._placement_cache)
            obs, _, over, capped, _ = env.step(action)
            if over or capped:
                raise RuntimeError(f"Replay ended before selected state: {seed}:{step}")
        if public_hash(env.core.get_public_observation()) != item["public_hash"]:
            raise RuntimeError(f"Selected public state diverged: {seed}:{step}")
        mask = env.action_masks()
        control_state = (control.active, control.last_switch, control.step)
        action, decision = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
        if (action != item["actual_action"] or decision["ppo_action"] != item["alternate_action"]
                or not mask[action] or not mask[item["alternate_action"]]
                or action == item["alternate_action"]):
            raise RuntimeError(f"Selected actions diverged: {seed}:{step}")
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {"seed": seed, "piece": step, "source_sha256": protocol["source_sha256"][str(seed)],
                 "model_sha256": protocol["model_sha256"], "env_state": env.get_state(),
                 "control": control_state,
                 "public_hash": item["public_hash"], "decision": decision}
        atomic_pickle(path, saved)
        atomic_json(path.with_suffix(".sha256.json"), {"sha256": digest(path), "piece": step})
        return path
    finally:
        env.close()


def branch(item, role, protocol):
    seed, step = item["seed"], item["step"]
    path = RUN / "branches" / f"seed_{seed}_step_{step}_{role}.json"
    state_path = RUN / "states" / f"seed_{seed}_step_{step}.pkl"
    if path.exists():
        row = json.loads(path.read_text())
        seal = json.loads(path.with_suffix(".sha256.json").read_text())
        if (digest(path) != seal["sha256"] or row["seed"] != seed or row["step"] != step
                or row["label"] != item["label"] or row["role"] != role
                or row["snapshot_sha256"] != digest(state_path)
                or row["model_sha256"] != protocol["model_sha256"]):
            raise RuntimeError(f"Completed branch mismatch: {path}")
        return row
    saved = load_checkpoint(seed, state_path, protocol)
    cap = max(CAP, step+HORIZON+1)
    state = saved["env_state"].copy()
    state["max_pieces"] = cap
    env, obs = new_env(seed, cap)
    env.set_state(state)
    obs = env.get_observation()
    control = controller()
    control.active, control.last_switch, control.step = saved["control"]
    if public_hash(env.core.get_public_observation()) != item["public_hash"]:
        raise RuntimeError("Branch start public hash differs")
    initial_score = env.core.score
    action = item["actual_action"] if role == "actual" else item["alternate_action"]
    clears = [0]*5
    risk = []
    started = time.monotonic()
    try:
        for offset in range(HORIZON):
            mask = env.action_masks()
            if offset == 0:
                expected, decision = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
                if expected != item["actual_action"]:
                    raise RuntimeError("Baseline B12 first action changed")
                selected = action
            else:
                selected, decision = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            if not mask[selected]:
                raise RuntimeError("Branch action illegal")
            obs, _, over, capped, info = env.step(selected)
            clears[info["cleared_lines"]] += 1
            if offset == 0:
                public = env.core.get_public_observation()
                first = {"hold": public["hold"], "next3": public["next"],
                         "legal_next": int(env.action_masks().sum()) if not over else 0}
            if (offset+1)%100 == 0 or over or capped:
                public = env.core.get_public_observation()
                holes, _, height, bump, _ = board_features(public["board"])
                risk.append({"offset": offset+1, "height": height, "holes": holes,
                             "bumpiness": bump, "central_top4": sum(public["board"][y][x] is not None
                               for y in range(4) for x in range(3,7))})
            if over or capped:
                break
        row = {"seed": seed, "step": step, "label": item["label"], "role": role,
               "action": action, "model_sha256": protocol["model_sha256"],
               "snapshot_sha256": digest(state_path), "initial_public_hash": item["public_hash"],
               "start_score": initial_score, "pieces_after": env.pieces-step,
               "end_piece": env.pieces, "game_over": bool(over),
               "censored": not over, "score_gain": env.core.score-initial_score,
               "clears_0_to_4": clears, "first_state": first,
               "risk_checkpoints": risk, "wall_seconds": time.monotonic()-started}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, row)
        atomic_json(path.with_suffix(".sha256.json"), {"sha256": digest(path),
                    "snapshot_sha256": row["snapshot_sha256"]})
        return row
    finally:
        env.close()


def worker_replay(args):
    return replay(*args)


def worker_materialize(args):
    return str(materialize(*args))


def worker_branch(args):
    return branch(*args)


def parallel(stage, items, fn, workers=WORKERS):
    status(stage, completed=0, total=len(items))
    if not items:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=load_worker) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
            status(stage, completed=len(results), total=len(items))
    return results


def windows(metrics, pieces):
    output = {}
    for width in (5000,1000,500,100,20):
        rows = metrics[-width:]
        if not rows:
            continue
        output[str(width)] = {"observed": len(rows), "height_start": rows[0]["height"],
              "height_end": rows[-1]["height"], "height_max": max(x["height"] for x in rows),
              "holes_start": rows[0]["holes"], "holes_end": rows[-1]["holes"],
              "bumpiness_start": rows[0]["bumpiness"], "bumpiness_end": rows[-1]["bumpiness"],
              "central_max": max(x["central_top4"] for x in rows),
              "legal_next_min": min(x["legal_next"] for x in rows),
              "fours": sum(x["cleared"]==4 for x in rows),
              "bc_actions": sum(x["source"]=="bc" for x in rows),
              "hold_changes": sum(a["hold"]!=b["hold"] for a,b in zip(rows,rows[1:]))}
    return output


def classify(actual, alternate):
    if actual["action"] == alternate["action"]:
        return "no_action_difference"
    delta = alternate["pieces_after"]-actual["pieces_after"]
    if delta >= 500 and (actual["game_over"] or alternate["censored"]):
        return "bc_harmful"
    if delta <= -500 and (alternate["game_over"] or actual["censored"]):
        return "bc_helpful"
    return "insufficient"


def report(protocol, replays, selection):
    results = []
    for item in selection["states"]:
        actual = branch(item,"actual",protocol)
        alternate = branch(item,"alternate",protocol)
        old = replays[item["seed"]]
        if old["game_over"] and old["pieces"]-item["step"] <= HORIZON:
            if actual["end_piece"] != old["pieces"] or not actual["game_over"]:
                raise RuntimeError(f"Actual counterfactual branch does not reproduce death: {item['seed']}:{item['step']}")
        results.append({"item": item, "actual": actual, "alternate": alternate,
                        "classification": classify(actual, alternate)})
    by_seed = {}
    for seed in DEATHS:
        matches = [x for x in results if x["item"]["seed"]==seed]
        labels = sorted({x["classification"] for x in matches})
        if not labels:
            labels = ["no_action_difference"]
        by_seed[str(seed)] = {"classifications": labels,
                "windows": windows(replays[seed]["data"]["metrics"], replays[seed]["pieces"]),
                "case_count": len(matches)}
    report_json = {"protocol": protocol, "replays": {str(s): {k:r[k] for k in
             ("pieces","score","game_over","capped","source_sha256")}
             for s,r in replays.items()}, "states": results, "by_seed": by_seed,
             "completed_at": now()}
    atomic_json(RUN / "summary.json", report_json)
    lines = ["# Hybrid B12 独立失败机制研究", "",
             "基于独立确认组的事后机制分析；这 32 个种子以后不能用作据此设计的 B12 V3 独立确认集。冻结 PPO、BC 和 B12 参数，未训练或调参。",
             "", f"验证重放：{len(replays)} 局（13 真正死亡，3 触顶对照），最终存活、得分和四消次数逐局与原记录一致。反事实状态 {len(selection['states'])} 个，分支 {2*len(selection['states'])} 条，每支最多 5000 块；触窗右截断。",
             "", "## 死亡局及窗口", "",
             "| Seed | 死亡块数 | BC 动作 | 有差异拦截 | 首次高 15/18 | 末 5000 高度 | 末 5000 洞数 | 末 5000 四消 | 证据分类 |",
             "|---:|---:|---:|---:|---:|---|---|---:|---|"]
    for seed in DEATHS:
        r = replays[seed]; d = r["data"]; w = by_seed[str(seed)]["windows"]["5000"]
        lines.append(f"| {seed} | {r['pieces']} | {d['bc_actions']} | {d['different_bc_actions']} | {d['first_height15']}/{d['first_height18']} | {w['height_start']}→{w['height_end']} | {w['holes_start']}→{w['holes_end']} | {w['fours']} | {', '.join(by_seed[str(seed)]['classifications'])} |")
    lines += ["", "分类可重叠；一次 seed 只在每类统计一次。高度、洞数、四消中断属于伴随现象，不能单独证明致死原因。完整 5000/1000/500/100/20 窗口与截断对照位于 `runs/hybrid_b12_death_study/summary.json`。", "", "## 同状态反事实", "",
              "| Seed:块 | 位置 | B12 动作 | 替代动作 | 原分支存活 | 替代分支存活 | 原/替代得分增量 | 原/替代四消 | Hold 首步 | 分类 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for x in results:
        i,a,b = x["item"],x["actual"],x["alternate"]
        lines.append(f"| {i['seed']}:{i['step']} | {i['label']} | {a['action']} | {b['action']} | {a['pieces_after']}{'+' if a['censored'] else ''} | {b['pieces_after']}{'+' if b['censored'] else ''} | {a['score_gain']}/{b['score_gain']} | {a['clears_0_to_4'][4]}/{b['clears_0_to_4'][4]} | {a['first_state']['hold']}/{b['first_state']['hold']} | {x['classification']} |")
    counts = {key:sum(key in item["classifications"] for item in by_seed.values())
              for key in ("bc_harmful","bc_helpful","insufficient","no_action_difference")}
    lines += ["", "## 触顶对照", "",
              "| Seed | 已观测存活 | BC 动作 | 有差异拦截 | 首次高 15/18 | 最后 5000 高度 | 最后 5000 洞数 |",
              "|---:|---:|---:|---:|---:|---|---|"]
    for seed in CONTROLS:
        r = replays[seed]; d = r["data"]; w = windows(d["metrics"],r["pieces"])["5000"]
        lines.append(f"| {seed} | {r['pieces']}+ | {d['bc_actions']} | {d['different_bc_actions']} | {d['first_height15']}/{d['first_height18']} | {w['height_start']}→{w['height_end']} | {w['holes_start']}→{w['holes_end']} |")
    lines += ["", "## 解释与建议", "",
              f"独立死亡 seed 数：有害拦截证据 {counts['bc_harmful']}，有益拦截证据 {counts['bc_helpful']}，证据不足 {counts['insufficient']}，无可利用动作差异 {counts['no_action_difference']}。",
              "此前六局中，700017:329 与 700026:268 发现有害拦截；本轮未重复这两个旧状态。应以本轮新的同状态结果判断能否跨 seed 复现，不能用不同模型的整局寿命直接归因。",
              "所有同状态分支只改变首动作，之后由冻结 B12 续控。Hold 差异会改变后续方块使用顺序；相同 Core/RNG 起点不保证两支经历完全相同的出块。", ""]
    if counts["bc_harmful"] >= 2:
        lines.append("多个独立死亡 seed 发现有害拦截，值得另开新开发种子研究 B12 V3 的拦截条件；本组不能用于 V3 确认，也不能据短窗结果断言 V3 会提高 100K 存活。")
    else:
        lines.append("本轮未在多个独立 seed 找到充分一致的有害拦截证据；暂不建议开发 B12 V3。优先保留冻结 B12，并在新样本上继续区分 PPO 累积风险与局部拦截效果。")
    REPORT.write_text("\n".join(lines)+"\n")
    return report_json


def main():
    try:
        protocol = freeze()
        status("smoke")
        load_worker()
        env, obs = new_env(1421001, 50)
        control = controller()
        try:
            for _ in range(50):
                action,_ = control.decide(obs, env.action_masks(), env.core.get_public_observation(), env._placement_cache)
                obs,_,over,capped,_ = env.step(action)
                if over or capped: break
            if env.pieces != 50 or not capped:
                raise RuntimeError("B12 smoke did not reach 50-piece cap")
        finally:
            env.close()
        status("replay")
        replays = {r["seed"]:r for r in parallel("replay",[(seed,protocol) for seed in SEEDS],worker_replay)}
        selected = select_states(replays,protocol)
        parallel("materialize",[(item,protocol) for item in selected["states"]],worker_materialize)
        branch_tasks = [(item,role,protocol) for item in selected["states"] for role in ("actual","alternate")]
        parallel("branches",branch_tasks,worker_branch)
        status("report")
        result = report(protocol,replays,selected)
        with (ROOT / "OPERATIONS.md").open("a") as stream:
            stream.write(f"\n- {now()}｜Hybrid B12 独立失败机制研究完成：{len(replays)} 局重放、{len(selected['states'])} 个状态、{len(branch_tasks)} 条反事实分支。依据：{(RUN/'summary.json').relative_to(ROOT)}。\n")
        status("completed", states=len(selected["states"]), branches=len(branch_tasks))
        return result
    except Exception as error:
        status("failed", error=str(error), traceback=traceback.format_exc())
        with (ROOT / "OPERATIONS.md").open("a") as stream:
            stream.write(f"\n- {now()}｜Hybrid B12 独立失败机制队列停止：{type(error).__name__}。依据：{(RUN/'status.json').relative_to(ROOT)}。\n")
        raise


if __name__ == "__main__":
    main()
