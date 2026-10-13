"""Resumable, isolated PPO30M/BC hybrid development evaluation."""
import hashlib
import json
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.hybrid_policy import HybridConfig, HybridController
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / "runs/candidate_ppo30m_bc_hybrid"
PPO = ROOT / "runs/ppo_candidate_raw_30m_seed42/final/model.zip"
BC = ROOT / "runs/candidate_bc_next_stage/champion/model.zip"
HASHES = {"ppo": "eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34",
          "bc": "5f5613cb5eb4e80a02406b70902ad3d5c571eb8fe38bec4b40f73b34289ca663"}
TUNE = list(range(700000, 700016))
CONFIRM = list(range(700016, 700032))
SMOKE = [700100, 700101]
VARIANTS = {"ppo": None, "bc": None,
            **{f"A{h}": HybridConfig("A", h, h - 4) for h in (12, 14, 16)},
            **{f"B{h}": HybridConfig("B", h, h - 4) for h in (12, 14, 16)}}
WORKER_MODELS = None


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_models():
    from training.candidate_features import FEATURE_VERSION
    from training.env.candidate_env import OBSERVATION_VERSION
    expected = {"policy": "CandidateScoringPolicy", "observation_version": OBSERVATION_VERSION,
                "candidate_feature_version": FEATURE_VERSION,
                "action_space_version": "1840-legal-placement-v1"}
    for label, path in (("ppo", PPO), ("bc", BC)):
        if digest(path) != HASHES[label]:
            raise RuntimeError(f"Frozen {label} checkpoint SHA256 mismatch")
        metadata = (ROOT / "runs/ppo_candidate_raw_30m_seed42/config.json" if label == "ppo"
                    else ROOT / "runs/candidate_bc_next_stage/champion/provenance.json")
        record = json.loads(metadata.read_text())
        record = record if label == "ppo" else record["source_config"]
        if any(record.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Frozen {label} model protocol mismatch")
    return expected


def audit_seeds():
    for seed in TUNE + CONFIRM + SMOKE:
        for base in (ROOT / "runs", ROOT / "reports"):
            if any(not p.is_relative_to(RUN) for p in base.rglob(f"seed_{seed}.json")):
                raise RuntimeError(f"Development seed already used: {seed}")


def load_models():
    torch.set_num_threads(1)
    models = {label: MaskablePPO.load(str(path), device="cpu")
              for label, path in (("ppo", PPO), ("bc", BC))}
    for model in models.values():
        if (model.policy.__class__.__name__ != "CandidateScoringPolicy" or
                model.action_space.n != 1840 or model.observation_space["state"].shape != (237,) or
                model.observation_space["candidate"].shape != (1840, 16)):
            raise RuntimeError("Loaded model space mismatch")
    return models


def worker_init():
    global WORKER_MODELS
    WORKER_MODELS = load_models()


def worker_game(variant, seed, cap, output):
    row = run_game(WORKER_MODELS, variant, seed, cap)
    atomic_json(output, row)
    return seed, row["pieces_survived"], row["score"]


def run_game(models, variant, seed, cap):
    env = FastCandidateTetrisEnv(max_pieces=cap)
    control = HybridController(models["ppo"], models["bc"], VARIANTS[variant]) if VARIANTS[variant] else None
    started = time.monotonic()
    trace, switches, selected = [], [], {"ppo": 0, "bc": 0}
    clears, milestones = [0] * 5, []
    inference_seconds = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            begin = time.monotonic()
            if control:
                public = env.core.get_public_observation()
                action, decision = control.decide(obs, mask, public, env._placement_cache)
                source = decision["source"]
            else:
                source = variant
                action = int(models[source].predict(obs, deterministic=True, action_masks=mask)[0])
                decision = None
            inference_seconds += time.monotonic() - begin
            if not mask[action]:
                raise RuntimeError("Illegal model action")
            index = env.pieces
            if not trace or trace[-1]["source"] != source:
                trace.append({"start": index, "source": source})
                if decision:
                    switches.append({"piece": index, **decision})
            selected[source] += 1
            obs, _, terminated, truncated, info = env.step(action)
            clears[info["cleared_lines"]] += 1
            if info["pieces"] % 1000 == 0:
                milestones.append({"pieces": info["pieces"], "score": info["score"]})
            if terminated or truncated:
                break
        return {"seed": seed, "variant": variant, "max_pieces": cap,
                "model_sha256": HASHES,
                "pieces_survived": info["pieces"], "score": info["score"],
                "lines": info["lines"], "game_over": bool(terminated),
                "survived_cap": bool(truncated), "clears_0_to_4": clears,
                "score_milestones": milestones, "source_runs": trace,
                "switches": switches, "actions_by_source": selected,
                "height_at_end": info["height_after"], "holes_at_end": info["holes_after"],
                "inference_seconds": inference_seconds,
                "wall_seconds": time.monotonic() - started}
    finally:
        env.close()


def summary(rows):
    pieces = sorted(r["pieces_survived"] for r in rows)
    total = sum(r["pieces_survived"] for r in rows)
    rank = .1 * (len(pieces) - 1)
    low = int(rank)
    p10 = pieces[low] + (pieces[min(low + 1, len(pieces) - 1)] - pieces[low]) * (rank - low)
    return {"games": len(rows), "mean_pieces": statistics.mean(pieces),
            "p10_pieces": p10,
            "capped": sum(r["survived_cap"] for r in rows),
            "mean_score": statistics.mean(r["score"] for r in rows),
            "score_per_1000_pieces": 1000 * sum(r["score"] for r in rows) / total,
            "four_clear_fraction": sum(r["clears_0_to_4"][4] for r in rows) /
                max(1, sum(sum(r["clears_0_to_4"][1:]) for r in rows)),
            "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in rows)}


def evaluate(phase, variants, seeds, cap):
    for variant in variants:
        folder = RUN / phase / variant
        folder.mkdir(parents=True, exist_ok=True)
        config = {"phase": phase, "variant": variant, "rule": vars(VARIANTS[variant]) if VARIANTS[variant] else None,
                  "seeds": seeds, "cap": cap, "model_sha256": HASHES,
                  "deterministic": True, "action_mask": True}
        path = folder / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise RuntimeError(f"Protocol changed: {path}")
        if not path.exists():
            atomic_json(path, config)
        pending = []
        for seed in seeds:
            output = folder / f"seed_{seed}.json"
            if output.exists():
                row = json.loads(output.read_text())
                if (row["seed"] != seed or row["variant"] != variant or
                        row["max_pieces"] != cap or row.get("model_sha256") != HASHES or
                        row["pieces_survived"] > cap or
                        row["survived_cap"] == row["game_over"] or
                        sum(row["actions_by_source"].values()) != row["pieces_survived"]):
                    raise RuntimeError(f"Existing game mismatches protocol: {output}")
                continue
            pending.append((variant, seed, cap, output))
        if pending:
            with ProcessPoolExecutor(max_workers=min(8, len(pending)),
                                     mp_context=get_context("spawn"), initializer=worker_init) as pool:
                futures = [pool.submit(worker_game, *args) for args in pending]
                for future in as_completed(futures):
                    seed, pieces, score = future.result()
                    atomic_json(RUN / "status.json", {"state": "running", "phase": phase,
                                "variant": variant, "last_seed": seed,
                                "completed": sum((folder / f"seed_{s}.json").exists() for s in seeds),
                                "total": len(seeds), "updated_at": now()})
                    print(f"{phase} {variant} seed={seed} pieces={pieces} score={score}", flush=True)
        rows = [json.loads((folder / f"seed_{seed}.json").read_text()) for seed in seeds]
        atomic_json(folder / "summary.json", summary(rows))


def final_report(chosen):
    report = ROOT / "reports/experiments/candidate_ppo30m_bc_hybrid.md"
    lines = ["# PPO 30M + BC 双策略融合实验", "",
             "冻结模型：纯 PPO 30M 与 20M PPO + BC epoch 3；均为 CandidateScoringPolicy、公开 237 状态 + 16 维合法落点、1840 Action ID。",
             f"SHA256：PPO `{HASHES['ppo']}`；BC `{HASHES['bc']}`。推理使用 CPU，因此模型占用 GPU 显存为 0；未更改 checkpoint 或 Core。", "",
             "## 协议", "",
             "700000–700015 是调参组，700016–700031 是一次确认组；先 20K 比较 12/14/16 阈值，再冻结每种方案一项，跑 100K。两组均未使用历史正式测试种子。",
             "达到上限的局按右截断处理；表中的达标数是观测到的下界。", "",
             "## 结果", ""]
    for phase in ("tune_20k", "tune_100k", "confirm_100k"):
        lines += [f"### {phase}", "", "| 模型 | 平均存活 | P10 存活 | 跑满 | 平均总分 | 每千块得分 | 四消事件占比 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        variants = list(VARIANTS) if phase == "tune_20k" else ["ppo", "bc", *chosen]
        for variant in variants:
            s = json.loads((RUN / phase / variant / "summary.json").read_text())
            lines.append(f"| {variant} | {s['mean_pieces']:.0f} | {s['p10_pieces']:.0f} | {s['capped']}/{s['games']} | {s['mean_score']:.0f} | {s['score_per_1000_pieces']:.0f} | {s['four_clear_fraction']:.2%} |")
        lines.append("")
    lines += ["### 确认组逐 seed 配对", "",
              "| Seed | PPO 存活 | BC 存活 | " + " | ".join(f"{v} 存活" for v in chosen) + " |",
              "|---:|---:|---:|" + "---:|" * len(chosen)]
    confirm = {variant: {seed: json.loads((RUN / "confirm_100k" / variant / f"seed_{seed}.json").read_text())
                         for seed in CONFIRM} for variant in ["ppo", "bc", *chosen]}
    for seed in CONFIRM:
        lines.append("| " + " | ".join(str(value) for value in
                     [seed, *[confirm[v][seed]["pieces_survived"] for v in ["ppo", "bc", *chosen]]]) + " |")
    lines += ["", "### 共同存活区间得分", "",
              "每组配对在两方共同存活的最后一个完整千块点比较累计得分；较短局不足 1000 块的配对不纳入此项。", "",
              "| 对比 | 有效配对 | 混合策略平均分 | 基线平均分 |",
              "|---|---:|---:|---:|"]
    for hybrid in chosen:
        for baseline in ("ppo", "bc"):
            pairs = []
            for seed in CONFIRM:
                a, b = confirm[hybrid][seed], confirm[baseline][seed]
                point = min(a["pieces_survived"], b["pieces_survived"]) // 1000 - 1
                if point >= 0:
                    pairs.append((a["score_milestones"][point]["score"],
                                  b["score_milestones"][point]["score"]))
            if pairs:
                lines.append(f"| {hybrid} vs {baseline} | {len(pairs)} | {statistics.mean(x for x, _ in pairs):.0f} | {statistics.mean(y for _, y in pairs):.0f} |")
    lines.append("")
    confirmed = {v: json.loads((RUN / "confirm_100k" / v / "summary.json").read_text())
                 for v in ["ppo", "bc", *chosen]}
    a, b = (confirmed[v] for v in chosen)
    if a["mean_pieces"] >= b["mean_pieces"] and a["mean_score"] >= b["mean_score"]:
        comparison = f"{chosen[0]} 在本次确认组的平均存活和总分均不低于 {chosen[1]}。"
    elif b["mean_pieces"] >= a["mean_pieces"] and b["mean_score"] >= a["mean_score"]:
        comparison = f"{chosen[1]} 在本次确认组的平均存活和总分均不低于 {chosen[0]}。"
    else:
        comparison = "A/B 在本次确认组的平均存活与总分有取舍，不能用单项指标宣布胜者。"
    lines += ["## 判断", "", comparison, ""]
    for hybrid in chosen:
        s = confirmed[hybrid]
        lines.append(f"- {hybrid}：平均存活为 BC 的 {s['mean_pieces']/confirmed['bc']['mean_pieces']:.1%}；四消事件占比为 PPO 的 {s['four_clear_fraction']/max(1e-9,confirmed['ppo']['four_clear_fraction']):.1%}；平均总分为 BC 的 {s['mean_score']/confirmed['bc']['mean_score']:.1%}。")
    lines += ["", "这些是 16 个独立确认种子的描述性结果，没有显著性检验；长期生存达到 100K 上限的局均右截断。", ""]
    lines += ["## 配对与开销", "", "同 seed 的原始局、逐千块得分、消行次数、策略连续区间与切换风险保存在 `runs/candidate_ppo30m_bc_hybrid/`。",
              "累计总分和每千块得分分别列出；共同存活区间的比较需按各 seed 较短对局的最后一个完整千块点计算，不能只选双方跑满的局。",
              "每局 `wall_seconds` 与 `inference_seconds` 均为实测。", "",
              "## 下一阶段", "", "若确认组同时显示 BC 级生存与接近 PPO 的四消能力，可用风险状态的双教师标签蒸馏到单模型。",
              "样本标签在安全区来自 PPO、危险区只在 BC 有明确更低风险时来自 BC；按四消频率、生存 P10、总分三目标验证，并保留原 PPO 动作一致性约束。",
              "若融合无增益，应基于 PPO 30M 收集其危险棋盘、训练局部纠错示范，避免全状态 BC 覆盖 PPO 的四消策略。此任务不启动训练。", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        protocol = verify_models()
        audit_seeds()
        atomic_json(RUN / "model_protocol.json", {"models": HASHES, "versions": protocol,
                    "tune_seeds": TUNE, "confirm_seeds": CONFIRM, "smoke_seeds": SMOKE})
        while True:
            upstream = json.loads((ROOT / "runs/candidate_bc_next_stage/queue_status.json").read_text())
            if upstream["state"] == "completed":
                break
            if upstream["state"] == "failed":
                raise RuntimeError("DAgger queue failed; high-load hybrid evaluation held")
            atomic_json(RUN / "status.json", {"state": "waiting_for_dagger", "updated_at": now()})
            time.sleep(30)
        evaluate("smoke", ["ppo", "bc", "A14", "B14"], SMOKE, 100)
        evaluate("tune_20k", list(VARIANTS), TUNE, 20000)
        results = {v: json.loads((RUN / "tune_20k" / v / "summary.json").read_text()) for v in VARIANTS}
        # Keep scoring as a gate; freeze one of each mode before the longer runs.
        def choose(mode):
            candidates = [v for v in VARIANTS if v.startswith(mode)]
            eligible = [v for v in candidates if
                        results[v]["four_clear_fraction"] >= .5 * results["ppo"]["four_clear_fraction"] and
                        results[v]["mean_pieces"] >= results["ppo"]["mean_pieces"]]
            if eligible:
                return max(eligible, key=lambda v: (results[v]["mean_score"], results[v]["mean_pieces"]))
            return max(candidates, key=lambda v: (results[v]["mean_pieces"], results[v]["mean_score"]))
        chosen = [choose(mode) for mode in ("A", "B")]
        atomic_json(RUN / "selection.json", {"chosen": chosen,
                    "rule": "among variants with >= PPO mean survival and >= half PPO four-clear fraction, maximize mean score; fallback maximize survival then score",
                    "tune_summaries": results})
        evaluate("tune_100k", ["ppo", "bc", *chosen], TUNE, 100000)
        evaluate("confirm_100k", ["ppo", "bc", *chosen], CONFIRM, 100000)
        final_report(chosen)
        atomic_json(RUN / "status.json", {"state": "completed", "updated_at": now()})
    except Exception:
        atomic_json(RUN / "status.json", {"state": "failed", "error": traceback.format_exc(),
                    "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
