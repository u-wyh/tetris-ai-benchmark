"""Frozen, resumable 32-seed confirmation of the selected PPO checkpoints."""

import json
import os
import re
import statistics
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from training.candidate_features import FEATURE_VERSION
from training.env.candidate_env import OBSERVATION_VERSION
from training.evaluation.failure_curriculum_eval import _common_score, _summary, init_worker, task
from training.evaluation.hybrid_queue import BC, PPO, digest, verify_models
from training.train_ppo import ROOT, atomic_json, now
from training.train_vector_transaction import verify_task

RUN = ROOT / "runs/ppo30m_independent_confirmation"
REPORT = ROOT / "reports/experiments/ppo30m_independent_confirmation.md"
SEEDS = tuple(range(1421000, 1421032))
VARIANTS = ("ppo", "B12", "normal_994303", "normal_994202", "bc")
CAP = 100000
WORKERS = min(8, max(1, os.cpu_count() or 1))
SOURCE = ROOT / "runs/ppo30m_multiseed_followup/model_manifest.json"


def audit_seeds():
    """Check exact numeric tokens in historical text and seed-named files."""
    pattern = re.compile(r"(?<!\d)14210(?:0\d|1\d|2\d|3[01])(?!\d)")
    seed_field = re.compile(r"(?:\"seed\"\s*:\s*|seed[_=]|\bseed\s+)(14210(?:0\d|1\d|2\d|3[01]))\b", re.I)
    hits = []
    for base in (ROOT / "runs", ROOT / "reports", ROOT / "training", ROOT / "tests"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.is_relative_to(RUN) or path == REPORT or path == Path(__file__):
                continue
            if pattern.search(path.name):
                hits.append(str(path.relative_to(ROOT)))
                continue
            if path.suffix not in (".json", ".md", ".py", ".csv", ".txt"):
                continue
            if path.stat().st_size > 2_000_000:
                continue
            if seed_field.search(path.read_text(errors="replace")):
                hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise RuntimeError(f"Confirmation seeds have prior use: {hits[:20]}")
    return {"seeds": list(SEEDS), "conflicts": [], "audited_at": now()}


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    if not (RUN / "protocol.json").exists():
        atomic_json(RUN / "seed_audit.json", audit_seeds())
    versions = verify_models()
    manifest = json.loads(SOURCE.read_text())
    hashes = {"ppo": digest(PPO), "bc": digest(BC)}
    paths = {"ppo": PPO, "bc": BC}
    for name in ("normal_994303", "normal_994202"):
        path = Path(manifest[name]["path"])
        if digest(path) != manifest[name]["sha256"]:
            raise RuntimeError(f"Model SHA256 mismatch: {name}")
        run_dir = path.parents[2]
        verify_task(path.parent, 123, json.loads((run_dir / "config.json").read_text()))
        hashes[name] = manifest[name]["sha256"]
        paths["normal" if name == "normal_994303" else "normal_994202"] = path
    # The existing evaluator uses its `normal` model key for either selected run.
    paths_by_variant = {
        "ppo": {"ppo": PPO}, "bc": {"bc": BC}, "B12": {"ppo": PPO, "bc": BC},
        "normal_994303": {"normal": Path(manifest["normal_994303"]["path"])},
        "normal_994202": {"normal": Path(manifest["normal_994202"]["path"])},
    }
    for name in ("normal_994303", "normal_994202"):
        model = MaskablePPO.load(str(manifest[name]["path"]), device="cpu")
        if (model.policy.__class__.__name__ != "CandidateScoringPolicy"
                or model.observation_space["state"].shape != (237,)
                or model.observation_space["candidate"].shape != (1840, 16)
                or model.action_space.n != 1840):
            raise RuntimeError(f"Candidate protocol mismatch: {name}")
        del model
    protocol = {
        "seeds": list(SEEDS), "cap": CAP, "variants": list(VARIANTS),
        "hashes": hashes, "model_manifest_sha256": digest(SOURCE),
        "observation_version": OBSERVATION_VERSION,
        "candidate_feature_version": FEATURE_VERSION,
        "model_protocol": versions, "core_rule": "official_core_v1",
        "inference": "deterministic_candidate_argmax_v1", "legal": "official_mask_fast_legal_parity",
        "workers": WORKERS, "checkpoint_interval": 10000,
        "gate": {"four_drop_pp_max": 3, "common_score_ratio_min": .95,
                 "mean_survival_gain_min": 1000, "p10_ratio_min": 1.0,
                 "capped_excess_min": 0, "early_under_10000_excess_max": 0},
        "selected_primary": "normal_994303", "selected_secondary": "normal_994202",
        "confirmation_use": "one_shot_no_retuning", "reference_git_commit": git_commit(),
    }
    destination = RUN / "protocol.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if {k: v for k, v in old.items() if k != "reference_git_commit"} != {
                k: v for k, v in protocol.items() if k != "reference_git_commit"}:
            raise RuntimeError("Frozen confirmation protocol changed")
        protocol = old
    else:
        atomic_json(destination, protocol)
    return protocol, paths_by_variant


def git_commit():
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def validated_row(path, variant, seed, protocol):
    row = json.loads(path.read_text())
    name = "normal" if variant.startswith("normal_") else variant
    if (row.get("variant") != name or row.get("seed") != seed or row.get("cap") != CAP
            or row.get("model_sha256") != protocol["hashes"]
            or not (row.get("game_over") != row.get("capped"))
            or not 0 < row.get("pieces", 0) <= CAP
            or row.get("capped") != (row.get("pieces") == CAP and not row.get("game_over"))):
        raise RuntimeError(f"Completed game mismatch: {path}")
    return row


def status(stage, **extra):
    atomic_json(RUN / "status.json", {"stage": stage, "at": now(), "pid": os.getpid(), **extra})


def run_games(protocol, paths):
    for variant in VARIANTS:
        folder = RUN / "evaluation" / variant
        folder.mkdir(parents=True, exist_ok=True)
        pending = []
        for seed in SEEDS:
            file = folder / f"seed_{seed}.json"
            if file.exists():
                validated_row(file, variant, seed, protocol)
                file.with_suffix(".checkpoint.pkl").unlink(missing_ok=True)
            else:
                pending.append(seed)
        if not pending:
            continue
        status("evaluating", variant=variant, completed=32-len(pending), total=32)
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn"),
                                 initializer=init_worker, initargs=(paths[variant],)) as pool:
            futures = {}
            for seed in pending:
                checkpoint = folder / f"seed_{seed}.checkpoint.pkl"
                name = "normal" if variant.startswith("normal_") else variant
                args = (name, seed, CAP, protocol["hashes"], True, True, str(checkpoint))
                futures[pool.submit(task, args)] = seed
            for future in as_completed(futures):
                seed = futures[future]
                file = folder / f"seed_{seed}.json"
                atomic_json(file, future.result())
                validated_row(file, variant, seed, protocol)
                file.with_suffix(".checkpoint.pkl").unlink(missing_ok=True)
                status("evaluating", variant=variant,
                       completed=sum((folder / f"seed_{s}.json").exists() for s in SEEDS), total=32)


def paired_bootstrap(rows_a, rows_b, field, samples=10000):
    differences = np.asarray([a[field]-b[field] for a, b in zip(rows_a, rows_b)], dtype=float)
    rng = np.random.default_rng(1421032)
    choices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[choices].mean(axis=1)
    return {"mean": float(differences.mean()),
            "ci95": [float(x) for x in np.quantile(means, [.025, .975])]}


def summarize(protocol):
    rows = {v: [validated_row(RUN / "evaluation" / v / f"seed_{seed}.json", v, seed, protocol)
                for seed in SEEDS] for v in VARIANTS}
    summaries = {v: {**_summary(items), "median_pieces": statistics.median(r["pieces"] for r in items),
                     "min_pieces": min(r["pieces"] for r in items),
                     "game_over": sum(r["game_over"] for r in items),
                     "total_lines": sum(sum(k*r["clears_0_to_4"][k] for k in range(1, 5)) for r in items),
                     "four_count": sum(r["clears_0_to_4"][4] for r in items)}
                 for v, items in rows.items()}
    paired = {}
    for target in ("ppo", "B12"):
        base = rows[target]
        candidate = rows["normal_994303"]
        common = [_common_score(a, b) for a, b in zip(candidate, base)]
        valid = [pair for pair in common if pair]
        paired[target] = {
            "pieces": paired_bootstrap(candidate, base, "pieces"),
            "score": paired_bootstrap(candidate, base, "score"),
            "common_score_ratio": sum(a for a, _ in valid)/max(1, sum(b for _, b in valid)),
            "by_seed": [{"seed": seed, "pieces_delta": a["pieces"]-b["pieces"],
                         "score_delta": a["score"]-b["score"],
                         "common_score_ratio": None if pair is None else pair[0]/max(1,pair[1]),
                         "four_fraction_delta":
                         a["clears_0_to_4"][4]/max(1,sum(a["clears_0_to_4"][1:]))
                         - b["clears_0_to_4"][4]/max(1,sum(b["clears_0_to_4"][1:]))}
                        for seed, a, b, pair in zip(SEEDS, candidate, base, common)]}
    ppo, candidate = summaries["ppo"], summaries["normal_994303"]
    gate = protocol["gate"]
    checks = {
        "four_drop_pp": 100*(ppo["four_fraction"]-candidate["four_fraction"]) <= gate["four_drop_pp_max"],
        "common_score": paired["ppo"]["common_score_ratio"] >= gate["common_score_ratio_min"],
        "mean_survival": candidate["mean_pieces"]-ppo["mean_pieces"] >= gate["mean_survival_gain_min"],
        "p10": candidate["p10_pieces"] >= ppo["p10_pieces"],
        "capped": candidate["capped"] >= ppo["capped"],
        "early_death": candidate["early_under_10000"] <= ppo["early_under_10000"],
    }
    result = {"protocol": protocol, "summary": summaries, "paired": paired,
              "gate_checks": checks, "gate_pass": all(checks.values()), "completed_at": now()}
    atomic_json(RUN / "summary.json", result)
    write_report(result, rows)
    return result


def write_report(result, rows):
    s = result["summary"]
    p = result["paired"]
    lines = ["# PPO 30M 新候选独立 100K 确认", "",
             "固定 32 个一次性独立种子 `1421000..1421031`，每局最多 100000 块；触顶按右截断处理。训练和开发数据未参与这次选模。",
             "", "| 模型 | 平均存活 | 中位数 | P10 | 最短 | 真正死亡 | 触顶 | 平均得分（亿） | 每千块得分 | 四消次数 | 四消占比 | 总消行 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        x = s[v]
        lines.append(f"| {v} | {x['mean_pieces']:.0f} | {x['median_pieces']:.0f} | {x['p10_pieces']:.0f} | {x['min_pieces']} | {x['game_over']} | {x['capped']}/32 | {x['mean_score']/1e8:.2f} | {x['score_per_1000']:.0f} | {x['four_count']} | {x['four_fraction']:.1%} | {x['total_lines']} |")
    lines += ["", "## 预冻结门槛", "",
              f"normal_994303：**{'通过' if result['gate_pass'] else '未通过'}**。" ]
    for key, passed in result["gate_checks"].items():
        lines.append(f"- {key}: {'通过' if passed else '未通过'}")
    lines += ["", "门槛：四消占比下降不超过 3 个百分点；共同存活得分至少为 PPO 的 95%；平均已观测存活至少多 1000 块；P10 不低于 PPO；触顶数不低于 PPO；低于 10000 块死亡数不增加。",
              "", "## 配对结果", ""]
    for target in ("ppo", "B12"):
        x = p[target]
        lines.append(f"相对 {target}：平均已观测存活差 {x['pieces']['mean']:.0f} 块（按 seed 配对 Bootstrap 95% 区间 {x['pieces']['ci95'][0]:.0f} 至 {x['pieces']['ci95'][1]:.0f}）；平均总分差 {x['score']['mean']/1e8:.2f} 亿（区间 {x['score']['ci95'][0]/1e8:.2f} 至 {x['score']['ci95'][1]/1e8:.2f} 亿）；共同存活得分比 {x['common_score_ratio']:.3f}。")
    lines += ["", "Bootstrap 以 32 个 seed 为抽样单位。共同存活分数在双方较短寿命向下取整至最近 1000 块的里程碑比较；不同寿命的总分受等级增长影响。100K 触顶不等于真实寿命。",
              "", "## 逐 seed 对照", "",
              "| Seed | PPO 存活/亿分 | B12 存活/亿分 | 994303 存活/亿分 | 994202 存活/亿分 | BC 存活/亿分 | 994303−PPO 存活 | 994303−B12 存活 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, seed in enumerate(SEEDS):
        cells = [f"{rows[v][i]['pieces']}{'*' if rows[v][i]['capped'] else ''}/{rows[v][i]['score']/1e8:.2f}" for v in ("ppo", "B12", "normal_994303", "normal_994202", "bc")]
        lines.append(f"| {seed} | " + " | ".join(cells) + f" | {p['ppo']['by_seed'][i]['pieces_delta']:+d} | {p['B12']['by_seed'][i]['pieces_delta']:+d} |")
    lines += ["", "`*` 表示 100K 右截断。完整逐局消行、共同存活得分和模型哈希在 `runs/ppo30m_independent_confirmation/`。",
              "", "## 结论", ""]
    if result["gate_pass"]:
        lines.append("normal_994303 在预定 32-seed、100K 协议下通过高分候选确认，可登记为高分候选冠军；仍需长局和网页适配验证，网页默认 AI 不变。")
    else:
        lines.append("normal_994303 未通过预冻结独立确认门槛；保留原 PPO 30M 和 Hybrid B12，不追加训练或在这 32 个种子上继续选模。")
    b12 = s["B12"]
    c = s["normal_994303"]
    lines.append(f"与 B12 比较：候选平均存活差 {c['mean_pieces']-b12['mean_pieces']:.0f} 块，P10 差 {c['p10_pieces']-b12['p10_pieces']:.0f} 块，平均总分差 {(c['mean_score']-b12['mean_score'])/1e8:.2f} 亿。")
    lines.append(f"异常早死（<10000 块）：候选 {[r['seed'] for r in rows['normal_994303'] if r['pieces']<10000]}；PPO {[r['seed'] for r in rows['ppo'] if r['pieces']<10000]}。")
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    try:
        protocol, paths = freeze()
        status("frozen", seeds=len(SEEDS), variants=len(VARIANTS))
        run_games(protocol, paths)
        result = summarize(protocol)
        status("completed", gate_pass=result["gate_pass"])
        with (ROOT / "OPERATIONS.md").open("a") as stream:
            stream.write(f"\n- {now()}｜独立 32-seed PPO 候选 100K 评测完成；预冻结门槛{'通过' if result['gate_pass'] else '未通过'}。依据：{(RUN/'summary.json').relative_to(ROOT)}。\n")
    except Exception as error:
        status("failed", error=str(error), traceback=traceback.format_exc())
        with (ROOT / "OPERATIONS.md").open("a") as stream:
            stream.write(f"\n- {now()}｜独立 PPO 候选评测队列停止：{type(error).__name__}；依赖任务未继续。依据：{(RUN/'status.json').relative_to(ROOT)}。\n")
        raise


if __name__ == "__main__":
    main()
