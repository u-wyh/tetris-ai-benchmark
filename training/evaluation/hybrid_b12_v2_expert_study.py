"""Resumable, exploratory V2 Hold interventions on frozen B12 snapshots."""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from training.evaluation import hybrid_b12_death_study as old
from training.evaluation.failure_curriculum_eval import atomic_pickle
from training.evaluation.ppo_death_capture import public_hash
from training.traditional.v2 import V2, V2Config
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/hybrid_b12_v2_expert_study"
SOURCE = old.RUN
HORIZON = 5000
REPORT = ROOT / "reports/experiments/hybrid_b12_v2_expert_study.md"


def sealed(path, value=None):
    seal = path.with_suffix(path.suffix + ".sha256.json")
    if path.exists():
        if not seal.exists() or json.loads(seal.read_text())["sha256"] != old.digest(path):
            raise RuntimeError(f"Invalid completed result: {path}")
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    atomic_json(seal, {"sha256": old.digest(path)})
    return value


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    old.verify_models()
    summary = json.loads((SOURCE / "summary.json").read_text())
    protocol = {"source_sha256": old.digest(SOURCE / "summary.json"),
                "model_sha256": old.identity(), "v2_sha256": old.digest(ROOT / "training/traditional/v2.py"),
                "v2_config": {"mode": "hold"}, "b12_config": {"mode": "B", "threshold": 12, "recovery": 8},
                "horizon": HORIZON, "death_seeds": list(old.DEATHS), "control_seeds": list(old.CONTROLS),
                "selection": "all 46 preverified death snapshots, plus first differing BC interception in each of three capped controls; maximum four per seed",
                "evidence": "V2 distinct from both PPO and BC; >=500-piece paired survival gain with baseline death and no large four-clear or common-window score loss; separate harmful evidence",
                "workers": 4}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Frozen protocol changed")
    if not path.exists():
        atomic_json(path, protocol)
    return summary, protocol


def selection(summary, protocol):
    path = RUN / "selection.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row["source_sha256"] != protocol["source_sha256"]:
            raise RuntimeError("Selection source changed")
        return row["states"]
    states = [x["item"] for x in summary["states"]]
    for seed in old.CONTROLS:
        replay = json.loads((SOURCE / "replays" / f"seed_{seed}.json").read_text())
        events = [e for e in replay["data"]["events"] if e["different"]]
        if not events:
            continue
        e = events[0]
        states.append({"seed": seed, "step": e["step"], "label": "control_first",
                       "public_hash": e["public_hash"], "actual_action": e["actual_action"],
                       "alternate_action": e["ppo_action"] if e["source"] == "bc" else e["bc_action"],
                       "source_sha256": replay["source_sha256"]})
    assert all(sum(x["seed"] == s for x in states) <= 4 for s in old.SEEDS)
    sealed(path, {"source_sha256": protocol["source_sha256"], "states": states})
    return states


def prepare(item):
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    old.materialize(item, protocol)
    state_path = SOURCE / "states" / f"seed_{item['seed']}_step_{item['step']}.pkl"
    saved = old.load_checkpoint(item["seed"], state_path, protocol)
    env, _ = old.new_env(item["seed"])
    try:
        env.set_state(saved["env_state"])
        public = env.core.get_public_observation()
        if public_hash(public) != item["public_hash"]:
            raise RuntimeError("Public state hash changed")
        placements = env._placement_cache or env.core.get_legal_placements()
        mask = env.action_masks()
        by_id = {p["actionId"] for p in placements}
        v2 = V2(V2Config(mode="hold")).choose(public, placements)
        ppo, bc = saved["decision"]["ppo_action"], saved["decision"]["bc_action"]
        actions = {"b12": item["actual_action"], "ppo": ppo, "bc": bc, "v2": v2}
        if any(a not in by_id or not mask[a] for a in actions.values()):
            raise RuntimeError("Expert proposed illegal action")
        row = {"seed": item["seed"], "step": item["step"], "label": item["label"],
               "actions": actions, "state_sha256": old.digest(state_path),
               "public_hash": item["public_hash"], "v2_timing_ms": V2(V2Config(mode="hold")).last_timing_ms}
        return sealed(RUN / "candidates" / f"seed_{item['seed']}_step_{item['step']}.json", row)
    finally:
        env.close()


def branch(task):
    item, action = task
    seed, step = item["seed"], item["step"]
    path = RUN / "branches" / f"seed_{seed}_step_{step}_action_{action}.json"
    if path.exists():
        row = sealed(path)
        if row["action"] != action or row["state_sha256"] != old.digest(SOURCE / "states" / f"seed_{seed}_step_{step}.pkl"):
            raise RuntimeError("Branch identity changed")
        return row
    source_protocol = json.loads((SOURCE / "protocol.json").read_text())
    state_path = SOURCE / "states" / f"seed_{seed}_step_{step}.pkl"
    saved = old.load_checkpoint(seed, state_path, source_protocol)
    state = saved["env_state"].copy()
    state["max_pieces"] = max(old.CAP, step + HORIZON + 1)
    env, obs = old.new_env(seed, state["max_pieces"])
    env.set_state(state)
    obs = env.get_observation()
    control = old.controller()
    control.active, control.last_switch, control.step = saved["control"]
    score0 = env.core.score
    clears = [0] * 5
    started = time.monotonic()
    try:
        if public_hash(env.core.get_public_observation()) != item["public_hash"]:
            raise RuntimeError("Branch state changed")
        for offset in range(HORIZON):
            mask = env.action_masks()
            expected, _ = control.decide(obs, mask, env.core.get_public_observation(), env._placement_cache)
            if offset == 0 and expected != item["actual_action"]:
                raise RuntimeError("Frozen B12 decision changed")
            selected = action if offset == 0 else expected
            if not mask[selected]:
                raise RuntimeError("Branch illegal action")
            obs, _, over, capped, info = env.step(selected)
            clears[info["cleared_lines"]] += 1
            if over or capped:
                break
        row = {"seed": seed, "step": step, "action": action,
               "state_sha256": old.digest(state_path), "model_sha256": old.identity(),
               "pieces_after": env.pieces-step, "game_over": bool(over),
               "censored": not over, "score_gain": env.core.score-score0,
               "clears_0_to_4": clears, "wall_seconds": time.monotonic()-started}
        return sealed(path, row)
    finally:
        env.close()


def worker_init():
    old.load_worker()


def run():
    summary, protocol = freeze()
    items = selection(summary, protocol)
    # A preexisting snapshot is required for every selected state.
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=worker_init) as pool:
        futures = [pool.submit(prepare, x) for x in items]
        candidates = [f.result() for f in as_completed(futures)]
    tasks = []
    for row in candidates:
        item = next(x for x in items if x["seed"] == row["seed"] and x["step"] == row["step"])
        tasks += [(item, a) for a in sorted(set(row["actions"].values()))]
    atomic_json(RUN / "status.json", {"stage": "branches", "total": len(tasks), "completed": 0, "at": now()})
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"), initializer=worker_init) as pool:
        futures = [pool.submit(branch, x) for x in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            atomic_json(RUN / "status.json", {"stage": "branches", "total": len(tasks), "completed": completed, "at": now()})
    report(summary, candidates)


def report(summary, candidates):
    outcomes = []
    for row in sorted(candidates, key=lambda x:(x["seed"], x["step"])):
        seed, step = row["seed"], row["step"]
        results = {name: sealed(RUN / "branches" / f"seed_{seed}_step_{step}_action_{action}.json")
                   for name, action in row["actions"].items()}
        baseline, v2 = results["b12"], results["v2"]
        independent = row["actions"]["v2"] not in (row["actions"]["ppo"], row["actions"]["bc"])
        gain = v2["pieces_after"]-baseline["pieces_after"]
        # Strong rescue needs real baseline death, substantial gain, and no large four-clear loss.
        rescue = (independent and baseline["game_over"] and gain >= 500
                  and v2["clears_0_to_4"][4] >= .9 * baseline["clears_0_to_4"][4])
        harm = independent and v2["game_over"] and gain <= -500
        outcomes.append({"seed": seed, "step": step, "label": row["label"], "actions": row["actions"],
                         "independent": independent, "gain": gain, "rescue": rescue,
                         "harm": harm, "branches": results})
    deaths = set(old.DEATHS)
    rescue_seeds = sorted({x["seed"] for x in outcomes if x["rescue"] and x["seed"] in deaths})
    harm_seeds = sorted({x["seed"] for x in outcomes if x["harm"]})
    sealed(RUN / "summary.json", {"states": outcomes, "rescue_death_seeds": rescue_seeds,
                                   "harm_seeds": harm_seeds})
    lines = ["# 传统 V2 Hold 对 B12 危险状态的探索性救援研究", "",
             "冻结 PPO、BC、B12 和 V2 Hold；复用 13 个死亡 seed 的 46 个已核验状态，并加入 3 个触顶对照状态。每支最多续控 5000 块，触窗为右截断。历史种子仅供机制研究。", "",
             f"独立 V2 动作状态：{sum(x['independent'] for x in outcomes)}/{len(outcomes)}；救援死亡 seed：{len(rescue_seeds)} {rescue_seeds}；明显有害 seed：{len(harm_seeds)} {harm_seeds}。", "",
             "| Seed:块 | B12/PPO/BC/V2 动作 | V2 独立 | B12/V2 存活 | B12/V2 得分增量 | B12/V2 四消 | 判定 |", "|---|---|---|---|---|---|---|",]
    for x in outcomes:
        a,b=x["branches"]["b12"],x["branches"]["v2"]
        fmt=lambda z:f"{z['pieces_after']}{'+' if z['censored'] else ''}"
        mark="救援" if x["rescue"] else "有害" if x["harm"] else "证据不足"
        lines.append(f"| {x['seed']}:{x['step']} | {x['actions']} | {x['independent']} | {fmt(a)}/{fmt(b)} | {a['score_gain']}/{b['score_gain']} | {a['clears_0_to_4'][4]}/{b['clears_0_to_4'][4]} | {mark} |")
    lines += ["", "救援判定仅指当前候选动作、冻结 B12 续控和 5000 块观察窗。不同寿命的总分不可直接解释为共同存活期间效率；本实验未保存逐块共同窗口得分，因此得分结论受限。离线结果不能成为线上可见信号。", "",
              "跨 seed 救援若不足，停止三专家方向；即使有重复案例，下一阶段也须先研究仅用公开棋盘特征识别调用时机，并在全新种子验证。本轮不开发混合策略或训练。", ""]
    REPORT.write_text("\n".join(lines))
    atomic_json(RUN / "status.json", {"stage": "completed", "states": len(outcomes),
                 "rescue_death_seeds": rescue_seeds, "harm_seeds": harm_seeds, "at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run", "report"])
    args = parser.parse_args()
    if args.command == "freeze":
        summary, protocol = freeze()
        selection(summary, protocol)
    elif args.command == "run":
        run()
    else:
        summary, _ = freeze()
        candidates = [json.loads(p.read_text()) for p in (RUN / "candidates").glob("*.json")]
        report(summary, candidates)
