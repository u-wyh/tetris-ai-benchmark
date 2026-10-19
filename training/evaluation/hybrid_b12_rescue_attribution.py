"""Read-only replay and bounded same-state attribution of B12/V2 rescue events."""

import json
import os
import pickle
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from training.candidate_features import board_features, candidate_board
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, atomic_pickle
from training.evaluation.hybrid_b12_v2_rescue_queue import (HASHES, ROOT, _DeterministicAdapter,
    PPO, BC, digest, verify_models, init_worker, MODELS)
from training.evaluation.ppo_death_capture import public_hash
from training.hybrid_b12_v2_rescue_policy import RescueController
from training.hybrid_policy import HybridConfig, HybridController
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public
from training.train_ppo import atomic_json, now

RUN = ROOT / "runs/hybrid_b12_rescue_attribution"
SOURCE = ROOT / "runs/hybrid_b12_v2_rescue/development_100k"
REPORT = ROOT / "reports/experiments/hybrid_b12_rescue_attribution.md"
SEEDS = (1440026, 1440012, 1440027, 1440024, 1440023)
EXPECTED = {1440026: 88170, 1440012: 41669, 1440027: 40664,
            1440024: -82546, 1440023: -2747}
CAP = 100000
HORIZON = 5000
MODELS = None


def source(seed, variant):
    path = SOURCE / variant / f"seed_{seed}.json"
    seal = path.with_suffix(".sha256.json")
    if not seal.exists() or json.loads(seal.read_text())["sha256"] != digest(path):
        raise RuntimeError(f"Source result checksum differs: {path}")
    row = json.loads(path.read_text())
    if row["seed"] != seed or row["variant"] != variant or row["cap"] != CAP:
        raise RuntimeError("Source result identity differs")
    return row


def protocol():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_models()
    rule = json.loads((ROOT / "runs/hybrid_b12_v2_rescue/protocol.json").read_text())
    if rule["models"] != HASHES or rule["rescue_rule_sha256"] != digest(ROOT / "training/hybrid_b12_v2_rescue_policy.py"):
        raise RuntimeError("Frozen rescue protocol changed")
    for seed in SEEDS:
        a, b = source(seed, "B12"), source(seed, "Rescue")
        if b["pieces"] - a["pieces"] != EXPECTED[seed]:
            raise RuntimeError(f"Source survival difference changed: {seed}")
    spec = {"seeds": list(SEEDS), "source_result_sha256": {
            f"{s}:{v}": digest(SOURCE / v / f"seed_{s}.json")
            for s in SEEDS for v in ("B12", "Rescue")},
            "models": HASHES, "b12_rule_sha256": digest(ROOT / "training/hybrid_policy.py"),
            "rescue_rule_sha256": rule["rescue_rule_sha256"],
            "v2_sha256": digest(ROOT / "training/traditional/v2.py"),
            "core_sha256": digest(ROOT / "training/tetris_core/core.py"),
            "max_piece": CAP, "short_horizon": HORIZON, "workers": 4,
            "selection": "all changed V2 events in five seeds; long branches at first changed event per seed and last changed event of 1440023",
            "safe_next_definition": "visible_next_only, no lock_out, resulting max_height<=18, at most one new hole",
            "continuations": ["B12_step_then_B12", "V2_step_then_B12", "V2_step_then_Rescue"],
            "censoring": "5000 continuation or total 100000 pieces is right-censored"}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Attribution protocol changed")
    if not path.exists():
        atomic_json(path, spec)
    return spec


def status(stage, **extra):
    atomic_json(RUN / "status.json", {"stage": stage, "pid": os.getpid(), "at": now(), **extra})


def sealed_json(path, value=None):
    seal = path.with_suffix(".sha256.json")
    if path.exists():
        if not seal.exists() or json.loads(seal.read_text())["sha256"] != digest(path):
            raise RuntimeError(f"Completed atomic result checksum differs: {path}")
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    atomic_json(seal, {"sha256": digest(path)})
    return value


def snapshot_path(seed, step):
    return RUN / "states" / f"seed_{seed}_step_{step}.pkl"


def save_snapshot(seed, step, env, base, decision, original_action, protocol_hash):
    path = snapshot_path(seed, step)
    if path.exists():
        seal = json.loads(path.with_suffix(".sha256.json").read_text())
        if seal["sha256"] != digest(path):
            raise RuntimeError("Completed snapshot checksum differs")
        return
    row = {"seed": seed, "step": step, "protocol_sha256": protocol_hash,
           "state": env.get_state(), "control": base,
           "public_hash": public_hash(env.core.get_public_observation()),
           "b12_action": original_action, "v2_action": decision["v2_action"],
           "ppo_action": decision["ppo_action"], "bc_action": decision["bc_action"],
           "base_source": decision["base_source"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle(path, row)
    atomic_json(path.with_suffix(".sha256.json"), {"sha256": digest(path)})


def load_snapshot(seed, step, proto):
    path = snapshot_path(seed, step)
    seal = json.loads(path.with_suffix(".sha256.json").read_text())
    if seal["sha256"] != digest(path):
        raise RuntimeError(f"Snapshot checksum mismatch: {path}")
    with path.open("rb") as stream:
        row = pickle.load(stream)
    if row["seed"] != seed or row["step"] != step or row["protocol_sha256"] != digest(RUN / "protocol.json"):
        raise RuntimeError("Snapshot identity mismatch")
    return row


def worker_init():
    global MODELS
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    MODELS = {"ppo": MaskablePPO.load(str(PPO), device="cpu"),
              "bc": MaskablePPO.load(str(BC), device="cpu")}


def controller(variant):
    p, b = _DeterministicAdapter(MODELS["ppo"]), _DeterministicAdapter(MODELS["bc"])
    return RescueController(p, b) if variant == "Rescue" else HybridController(p, b, HybridConfig("B", 12, 8))


def replay(task):
    seed, variant = task
    path = RUN / "replays" / variant / f"seed_{seed}.json"
    if path.exists():
        row = sealed_json(path)
        if row["source_sha256"] != digest(SOURCE / variant / f"seed_{seed}.json"):
            raise RuntimeError("Replay source changed")
        return row
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    control = controller(variant)
    original = source(seed, variant)
    actions = []
    changed = []
    clears = [0] * 5
    started = time.monotonic()
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            step = env.pieces
            mask = env.action_masks()
            public = env.core.get_public_observation()
            base = control.base if variant == "Rescue" else control
            before = (base.active, base.last_switch, base.step)
            action, decision = control.decide(obs, mask, public, env._placement_cache)
            if not mask[action]:
                raise RuntimeError("Replay illegal action")
            if variant == "Rescue" and decision["v2_changed"]:
                base_action = decision[decision["base_source"] + "_action"]
                save_snapshot(seed, step, env, before, decision, base_action, digest(RUN / "protocol.json"))
                changed.append({"step": step, "b12_action": base_action,
                                "v2_action": action, "public_hash": public_hash(public)})
            actions.append(action)
            obs, _, over, capped, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            if over or capped:
                break
        expected = [(e["piece"], e["v2_action"]) for e in original["intercepts"] if e["v2_changed"]] if variant == "Rescue" else []
        if (info["pieces"] != original["pieces"] or info["score"] != original["score"]
                or bool(over) != original["game_over"] or bool(capped) != original["capped"]
                or clears != original["clears_0_to_4"]
                or [(e["step"], e["v2_action"]) for e in changed] != expected):
            raise RuntimeError(f"Deterministic replay mismatch: {variant}:{seed}")
        row = {"seed": seed, "variant": variant,
               "source_sha256": digest(SOURCE / variant / f"seed_{seed}.json"),
               "pieces": info["pieces"], "score": info["score"],
               "game_over": bool(over), "capped": bool(capped),
               "clears_0_to_4": clears, "actions": actions, "changed": changed,
               "wall_seconds": time.monotonic() - started}
        return sealed_json(path, row)
    finally:
        env.close()


def first_divergence(seed):
    a = sealed_json(RUN / "replays" / "B12" / f"seed_{seed}.json")
    b = sealed_json(RUN / "replays" / "Rescue" / f"seed_{seed}.json")
    found = next((i for i, (x, y) in enumerate(zip(a["actions"], b["actions"])) if x != y), None)
    if found is None or found != b["changed"][0]["step"]:
        raise RuntimeError(f"First action difference does not match V2 intervention: {seed}")
    snap = load_snapshot(seed, found, None)
    if a["actions"][found] != snap["b12_action"] or b["actions"][found] != snap["v2_action"]:
        raise RuntimeError("First divergence action mismatch")
    return {"seed": seed, "step": found, "snapshot_sha256": digest(snapshot_path(seed, found))}


def candidate_features(seed, step):
    path = RUN / "candidates" / f"seed_{seed}_step_{step}.json"
    if path.exists():
        return sealed_json(path)
    snap = load_snapshot(seed, step, None)
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=seed)
        env.set_state(snap["state"])
        public = env.core.get_public_observation()
        if public_hash(public) != snap["public_hash"]:
            raise RuntimeError("Candidate public state mismatch")
        legal = env.core.get_legal_placements()
        by_id = {p["actionId"]: p for p in legal}
        mask = env.action_masks()
        holes, total, height, bump, wells = board_features(public["board"])
        actions = {k: snap[k + "_action"] for k in ("ppo", "bc", "b12", "v2")}
        result = {}
        for name, action in actions.items():
            if action not in by_id or not mask[action]:
                raise RuntimeError("Candidate action illegal")
            placement = by_id[action]
            board, cleared, lockout = candidate_board(public["board"], placement["occupiedCells"])
            h, agg, maximum, uneven, wells_after = board_features(board)
            visible, _ = simulate_public(public, placement)
            next_legal = next_safe = 0
            if visible["phase"] == "playing":
                next_moves = get_legal_placements(visible)
                next_legal = len(next_moves)
                for move in next_moves:
                    child, _, dead = candidate_board(visible["board"], move["occupiedCells"])
                    child_holes, _, child_height, _, _ = board_features(child)
                    next_safe += not dead and child_height <= 18 and child_holes <= h + 1
            result[name] = {"action": action, "hold": bool(placement["hold"]),
                            "height_after": maximum, "holes_after": h,
                            "new_holes": h - holes, "bump_after": uneven,
                            "cleared": cleared, "lock_out": bool(lockout),
                            "next_visible_legal": next_legal, "next_visible_safe": next_safe,
                            "visible_phase": visible["phase"]}
        row = {"seed": seed, "step": step, "public_hash": snap["public_hash"],
               "snapshot_sha256": digest(snapshot_path(seed, step)),
               "before": {"height": height, "holes": holes, "aggregate_height": total,
                          "bumpiness": bump, "wells": wells, "hold": public["hold"],
                          "next3": public["next"], "legal": len(legal)},
               "actions": result}
        return sealed_json(path, row)
    finally:
        env.close()


def branch(task):
    seed, step, role, horizon = task
    phase = "5000" if horizon == HORIZON else "100k"
    path = RUN / f"branches_{phase}" / f"seed_{seed}_step_{step}_{role}.json"
    if path.exists():
        row = sealed_json(path)
        if row["snapshot_sha256"] != digest(snapshot_path(seed, step)):
            raise RuntimeError("Branch snapshot changed")
        return row
    snap = load_snapshot(seed, step, None)
    cap = max(CAP, step + horizon + 1) if horizon == HORIZON else CAP
    state = snap["state"].copy()
    state["max_pieces"] = cap
    env = LongCandidateTetrisEnv(max_pieces=cap)
    env.reset(seed=seed)
    env.set_state(state)
    obs = env.get_observation()
    control = controller("Rescue" if role == "v2_then_rescue" else "B12")
    base = control.base if role == "v2_then_rescue" else control
    base.active, base.last_switch, base.step = snap["control"]
    score0 = env.core.score
    clears = [0] * 5
    score_trace = []
    started = time.monotonic()
    try:
        if public_hash(env.core.get_public_observation()) != snap["public_hash"]:
            raise RuntimeError("Branch start state changed")
        for offset in range(horizon):
            mask = env.action_masks()
            proposed, _ = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            selected = (snap["b12_action"] if role == "b12_then_b12" else snap["v2_action"]) if offset == 0 else proposed
            if offset == 0 and role == "b12_then_b12" and proposed != selected:
                raise RuntimeError("Baseline B12 proposal changed")
            if offset == 0 and role == "v2_then_rescue" and proposed != selected:
                raise RuntimeError("Rescue proposal changed")
            if not mask[selected]:
                raise RuntimeError("Branch action illegal")
            obs, _, over, capped, info = env.step(selected)
            clears[info["cleared_lines"]] += 1
            if offset < HORIZON:
                score_trace.append(info["score"] - score0)
            if over or capped:
                break
        row = {"seed": seed, "step": step, "role": role, "horizon": horizon,
               "first_action": snap["b12_action"] if role == "b12_then_b12" else snap["v2_action"],
               "snapshot_sha256": digest(snapshot_path(seed, step)),
               "pieces_after": env.pieces - step, "end_piece": env.pieces,
               "game_over": bool(over), "censored": not over,
               "score_gain": env.core.score - score0, "score_trace_first_5000": score_trace,
               "clears_0_to_4": clears, "wall_seconds": time.monotonic()-started}
        return sealed_json(path, row)
    finally:
        env.close()


def parallel(stage, tasks, func):
    status(stage, total=len(tasks), completed=0)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=worker_init) as pool:
        pending = {pool.submit(func, task): task for task in tasks}
        for completed, future in enumerate(as_completed(pending), 1):
            future.result()
            status(stage, total=len(tasks), completed=completed)


def main():
    try:
        protocol()
        parallel("replay", [(s, v) for s in SEEDS for v in ("B12", "Rescue")], replay)
        divergent = [first_divergence(seed) for seed in SEEDS]
        atomic_json(RUN / "first_divergence.json", divergent)
        all_events = []
        for seed in SEEDS:
            row = sealed_json(RUN / "replays" / "Rescue" / f"seed_{seed}.json")
            all_events += [(seed, e["step"]) for e in row["changed"]]
        for seed, step in all_events:
            candidate_features(seed, step)
        roles = ("b12_then_b12", "v2_then_b12", "v2_then_rescue")
        parallel("counterfactual_5000", [(s, t, role, HORIZON) for s, t in all_events for role in roles], branch)
        selected = [(s, t) for s, t in all_events if t == next(e["step"] for e in
                    sealed_json(RUN / "replays" / "Rescue" / f"seed_{s}.json")["changed"])]
        last_23 = max(t for s, t in all_events if s == 1440023)
        if (1440023, last_23) not in selected:
            selected.append((1440023, last_23))
        atomic_json(RUN / "long_selection.json", [{"seed": s, "step": t} for s, t in selected])
        parallel("counterfactual_100k", [(s, t, role, CAP-t) for s, t in selected for role in roles], branch)
        write_report(all_events, selected)
        status("completed", states=len(all_events), long_states=len(selected))
    except Exception as exc:
        status("failed", error=str(exc), traceback=traceback.format_exc())
        raise


def write_report(events, selected):
    divergent = json.loads((RUN / "first_divergence.json").read_text())
    lines = ["# B12＋V2 救援正负差异机制归因", "",
             "仅对已有开发种子作事后机制分析；不训练模型、不调整阈值、不产生独立确认结论。冻结 PPO、BC、B12、V2 与 Core，所有动作来自官方合法掩码。", "",
             "## 确定性重放与首次分歧", "",
             "五个 seed 的 B12 与 Rescue 从空盘重放，存活、得分、Game Over/触顶、消行分布和实际 V2 替换事件均与已有结果一致。首次动作分歧均与首次 V2 替换重合。", "",
             "| Seed | B12 最终块数 | Rescue 最终块数 | 首次分歧 | V2 实际替换数 | 起始高/洞/凹凸 | B12/V2 首步动作 |",
             "|---:|---:|---:|---:|---:|---|---|"]
    for d in divergent:
        s, t = d["seed"], d["step"]
        a = sealed_json(RUN / "replays" / "B12" / f"seed_{s}.json")
        b = sealed_json(RUN / "replays" / "Rescue" / f"seed_{s}.json")
        c = sealed_json(RUN / "candidates" / f"seed_{s}_step_{t}.json")
        m = c["before"]
        lines.append(f"| {s} | {a['pieces']}{'+' if a['capped'] else ''} | {b['pieces']}{'+' if b['capped'] else ''} | {t} | {len(b['changed'])} | {m['height']}/{m['holes']}/{m['bumpiness']} | {c['actions']['b12']['action']}/{c['actions']['v2']['action']} |")
    lines += ["", "## 同状态候选与反事实", "",
              "下表的 A 为原 B12 首步后由 B12 续控；B 为 V2 首步后由 B12 续控；C 为 V2 首步后继续原 Rescue。数字后 `+` 表示到观察窗右截断，不表示真实死亡。", "",
              "| Seed:步 | 当前高/洞/凹凸 | PPO/BC/B12/V2 动作 | V2 相对 B12 落后高/洞/凹凸 | 可见下一块安全落点 B12/V2 | A/B/C 5000 块续存 |",
              "|---|---|---|---|---|---|"]
    cases = []
    for s, t in events:
        c = sealed_json(RUN / "candidates" / f"seed_{s}_step_{t}.json")
        branches = {r: sealed_json(RUN / "branches_5000" / f"seed_{s}_step_{t}_{r}.json") for r in
                    ("b12_then_b12", "v2_then_b12", "v2_then_rescue")}
        x, y = c["actions"]["b12"], c["actions"]["v2"]
        fmt = lambda q: f"{q['pieces_after']}{'+' if q['censored'] else ''}"
        m = c["before"]
        lines.append(f"| {s}:{t} | {m['height']}/{m['holes']}/{m['bumpiness']} | "
                     f"{c['actions']['ppo']['action']}/{c['actions']['bc']['action']}/{x['action']}/{y['action']} | "
                     f"{y['height_after']-x['height_after']:+d}/{y['holes_after']-x['holes_after']:+d}/{y['bump_after']-x['bump_after']:+d} | "
                     f"{x['next_visible_safe']}/{y['next_visible_safe']} | "
                     f"{fmt(branches['b12_then_b12'])}/{fmt(branches['v2_then_b12'])}/{fmt(branches['v2_then_rescue'])} |")
        cases.append((s, t, c, branches))
    lines += ["", "安全落点仅对**可见下一块**计数：不立即 Lock Out、落后最高列≤18 且新增洞≤1；不知道隐藏 Bag。四个动作的完整结构、Hold、下一步合法/安全落点及三个分支的得分轨迹、四消次数见实验 JSON。", "",
              "## 延长至 100K 总块数的关键分支", "",
              "只延长每个 seed 的首次替换，另加入 `1440023` 的最后一次替换。", "",
              "| Seed:步 | A/B/C 后续存活 | A/B/C 得分增量 | A/B/C 四消 |",
              "|---|---|---|---|"]
    for s, t in selected:
        b = [sealed_json(RUN / "branches_100k" / f"seed_{s}_step_{t}_{role}.json") for role in
             ("b12_then_b12", "v2_then_b12", "v2_then_rescue")]
        fmt = lambda q: f"{q['pieces_after']}{'+' if q['censored'] else ''}"
        lines.append(f"| {s}:{t} | {'/'.join(map(fmt,b))} | {'/'.join(str(q['score_gain']) for q in b)} | {'/'.join(str(q['clears_0_to_4'][4]) for q in b)} |")
    lines += ["", "## 解释与边界", "",
              "同 seed 的完整策略结局差只能定位研究对象；单次动作的证据以**相同 Core/RNG 快照、相同冻结 B12 续控**的 A/B 分支为准。B/C 差异用于识别后续 Rescue 干预影响。Hold 改变若影响后续取块顺序，属于动作的真实后果，但分支不再经历完全相同的实际方块顺序。", "",
              "这些开发种子来自已经观察过的选模组，事件数量少且同一局内高度相关，尚不足以支持学习型专家选择器。下一阶段应在新的诊断种子收集所有高危调用点及未调用的匹配对照，记录公开状态、所有专家动作、短窗同状态反事实和共同存活得分，并按 episode 分组；训练、开发、独立确认 seed 必须分离。本轮不直接训练或制定新切换规则。", ""]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
