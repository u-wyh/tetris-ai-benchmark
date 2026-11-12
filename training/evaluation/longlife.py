"""Cap-aware paired analysis of unchanged V1, V2 Hold-only, and V2 Beam-8 runs."""
import argparse
import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path

from training.evaluation.traditional import ROOT, atomic, now, percentile
from training.tetris_core import TetrisCore
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import V2, V2Config, board_score, features, simulate_public

MODES = ("v1", "hold", "beam")
PAIRS = (("v1", "hold"), ("v1", "beam"), ("hold", "beam"))


def read_run(folder, mode):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    if status["state"] != "complete":
        raise ValueError(f"Incomplete run: {folder}")
    if config["strategy"] != ("v1-adapter-1" if mode == "v1" else "v2-1"):
        raise ValueError(f"Strategy mismatch: {folder}")
    if mode != "v1" and (config["policy"]["mode"] != mode or
                         config["policy"]["beam_width"] != 8 or
                         config["policy"]["lookahead"] != .58):
        raise ValueError(f"V2 configuration mismatch: {folder}")
    rows = {}
    for seed in config["seeds"]:
        row = json.loads((folder / f"seed_{seed}.json").read_text())
        if (row["seed"] != seed or len(row["actions"]) != row["pieces_survived"] or
            len(row["decision_ms"]) != row["pieces_survived"] or
            row["game_over"] == row["truncated"] or
            (row["truncated"] and row["pieces_survived"] != config["max_pieces"])):
            raise ValueError(f"Invalid completed seed: {folder} {seed}")
        rows[seed] = row
    return config, rows


def bootstrap_mean(values, seed=20261008, samples=10000):
    generator = random.Random(seed)
    n = len(values)
    estimates = sorted(sum(generator.choice(values) for _ in range(n)) / n
                       for _ in range(samples))
    return [estimates[int(samples * .025)], estimates[int(samples * .975)]]


def score_at_targets(seed, actions, targets):
    if not targets:
        return {}
    core = TetrisCore(seed)
    found = {}
    for index, action in enumerate(actions[:max(targets)], 1):
        core.step(action)
        if index in targets:
            found[index] = core.score
    if set(found) != set(targets):
        raise AssertionError("Score replay missed a requested piece count")
    return found


def paired_equal_piece_scores(rows, seeds):
    targets = {(mode, seed): set() for mode in MODES for seed in seeds}
    for seed in seeds:
        for a, b in PAIRS:
            n = min(rows[a][seed]["pieces_survived"], rows[b][seed]["pieces_survived"])
            targets[a, seed].add(n)
            targets[b, seed].add(n)
    scores = {}
    for (mode, seed), requested in targets.items():
        row = rows[mode][seed]
        scores[mode, seed, row["pieces_survived"]] = row["score"]
        missing = requested - {row["pieces_survived"]}
        for n, score in score_at_targets(seed, row["actions"], missing).items():
            scores[mode, seed, n] = score
    return scores


def score_breakdown(row):
    """Replay official scoring; no strategy decisions are recomputed."""
    core = TetrisCore(row["seed"])
    clears = {str(count): 0 for count in range(5)}
    line_points = 0
    drop_points = 0
    for action in row["actions"]:
        old_score, old_lines, old_level = core.score, core.lines, core.level
        core.step(action)
        count = core.lines - old_lines
        clears[str(count)] += 1
        award = (0, 100, 300, 500, 800)[count] * old_level
        line_points += award
        drop_points += core.score - old_score - award
    if core.score != row["score"] or core.lines != row["lines"]:
        raise AssertionError("Score breakdown replay diverged from the saved episode")
    return dict(seed=row["seed"], pieces=row["pieces_survived"],
                clears=clears, line_points=line_points, drop_points=drop_points,
                total_score=core.score, lines=core.lines)


def summarize(configs, rows):
    seeds = configs["v1"]["seeds"]
    cap = configs["v1"]["max_pieces"]
    for mode in MODES:
        if configs[mode]["seeds"] != seeds or configs[mode]["max_pieces"] != cap:
            raise ValueError("Seed sets and caps must match exactly")
        if configs[mode]["git_commit"] != configs["v1"]["git_commit"]:
            raise ValueError("Runs must share a Git baseline")
    if configs["hold"]["code_sha256"] != configs["beam"]["code_sha256"]:
        raise ValueError("Hold and Beam must use the same V2 implementation")
    for field in ("weights", "death_penalty", "lookahead", "beam_width"):
        if configs["hold"]["policy"][field] != configs["beam"]["policy"][field]:
            raise ValueError("Hold and Beam must share V2 parameters")
    scores = paired_equal_piece_scores(rows, seeds)
    result = dict(created_at=now(), seeds=seeds, max_pieces=cap,
                  analysis_git_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  config={mode: configs[mode] for mode in MODES}, modes={}, pairs={})
    for mode in MODES:
        data = [rows[mode][seed] for seed in seeds]
        pieces = [r["pieces_survived"] for r in data]
        times = [t for r in data for t in r["decision_ms"]]
        total_pieces = sum(pieces)
        result["modes"][mode] = dict(
            restricted_mean_pieces=statistics.mean(pieces),
            restricted_mean_bootstrap_95ci=bootstrap_mean(pieces),
            median_observed_pieces=statistics.median(pieces),
            p90_observed_pieces=percentile(pieces, .9),
            game_over_rate=statistics.mean(r["game_over"] for r in data),
            truncation_rate=statistics.mean(r["truncated"] for r in data),
            mean_lines=statistics.mean(r["lines"] for r in data),
            lines_per_placed_piece=sum(r["lines"] for r in data) / total_pieces,
            mean_official_score=statistics.mean(r["score"] for r in data),
            final_score_per_placed_piece=sum(r["score"] for r in data) / total_pieces,
            mean_decision_ms=statistics.mean(times), p95_decision_ms=percentile(times, .95),
            total_wall_seconds=sum(r["wall_seconds"] for r in data),
            decisions_per_second=1000 / statistics.mean(times),
            end_to_end_pieces_per_second=total_pieces / sum(r["wall_seconds"] for r in data))
    for a, b in PAIRS:
        key = f"{b}_minus_{a}"
        paired = []
        for seed in seeds:
            x, y = rows[a][seed], rows[b][seed]
            n = min(x["pieces_survived"], y["pieces_survived"])
            sx, sy = scores[a, seed, n], scores[b, seed, n]
            paired.append(dict(seed=seed, equal_pieces=n,
                a_pieces=x["pieces_survived"], b_pieces=y["pieces_survived"],
                a_game_over=x["game_over"], b_game_over=y["game_over"],
                a_truncated=x["truncated"], b_truncated=y["truncated"],
                observed_pieces_difference=y["pieces_survived"]-x["pieces_survived"],
                a_score_at_equal_pieces=sx, b_score_at_equal_pieces=sy,
                score_difference_at_equal_pieces=sy-sx,
                score_per_piece_difference_at_equal_pieces=(sy-sx)/n))
        diffs = [r["observed_pieces_difference"] for r in paired]
        equal_score_diffs = [r["score_per_piece_difference_at_equal_pieces"] for r in paired]
        result["pairs"][key] = dict(
            paired=paired, restricted_mean_difference=statistics.mean(diffs),
            restricted_mean_difference_bootstrap_95ci=bootstrap_mean(diffs),
            mean_equal_piece_score_per_piece_difference=statistics.mean(equal_score_diffs),
            mean_equal_piece_score_per_piece_difference_bootstrap_95ci=bootstrap_mean(equal_score_diffs),
            both_truncated=sum(r["a_truncated"] and r["b_truncated"] for r in paired),
            a_over_b_truncated=sum(r["a_game_over"] and r["b_truncated"] for r in paired),
            a_truncated_b_over=sum(r["a_truncated"] and r["b_game_over"] for r in paired),
            observed_b_wins=sum(d > 0 for d in diffs),
            observed_a_wins=sum(d < 0 for d in diffs),
            observed_ties=sum(d == 0 for d in diffs))
    result["seeds_detail"] = [{"seed": seed, **{
        mode: {key: rows[mode][seed][key] for key in
               ("pieces_survived", "lines", "score", "game_over", "truncated", "wall_seconds")}
        for mode in MODES}} for seed in seeds]
    return result


def action_explanation(observation, action, weights, lookahead):
    placement = next(p for p in get_legal_placements(observation) if p["actionId"] == action)
    state, cleared = simulate_public(observation, placement)
    immediate = board_score(state["board"], cleared, weights) - (1e6 if state["gameOver"] else 0)
    future = None
    if state["phase"] == "playing":
        second = get_legal_placements(state)
        if second:
            future = max(board_score(s["board"], c, weights) - (1e6 if s["gameOver"] else 0)
                         for s, c in (simulate_public(state, p) for p in second))
        else:
            future = -1e6
    return dict(action_id=action, path=placement["path"], hold=bool(placement["hold"]),
                immediate_heuristic=immediate, best_visible_second_step=future,
                two_step_value=immediate + (lookahead * future if future is not None else 0),
                cleared=cleared, resulting_features=features(state["board"]),
                resulting_phase=state["phase"], resulting_score=state["score"])


def common_continuation(rows, case, first_action, horizon=200):
    """Fork a reconstructed Core; only the first action differs between forks."""
    core = TetrisCore(case["seed"])
    for action in rows[case["source"]][case["seed"]]["actions"][:case["step"]]:
        core.step(action)
    if first_action not in {p["actionId"] for p in core.get_legal_placements()}:
        raise AssertionError("Counterfactual first action is illegal")
    core.step(first_action)
    survived = 1
    policy = V2(V2Config(mode="hold"))
    while core.phase == "playing" and survived < horizon:
        observation = core.get_public_observation()
        placements = core.get_legal_placements()
        core.step(policy.choose(observation, placements))
        survived += 1
    return dict(pieces=survived, game_over=core.phase == "over",
                truncated=core.phase != "over", lines=core.lines, score=core.score,
                horizon=horizon, continuation_policy="hold")


def find_disagreements(rows, seeds, max_per_source=10000, stride=100, keep=8,
                       counterfactual_horizon=200):
    hold_policy = V2(V2Config(mode="hold"))
    beam_policy = V2(V2Config(mode="beam"))
    candidates = []
    for source in ("hold", "beam"):
        for seed in seeds[:min(3, len(seeds))]:
            core = TetrisCore(seed)
            actions = rows[source][seed]["actions"]
            for index, action in enumerate(actions[:max_per_source]):
                height = features(core.board)["max_height"] if core.phase == "playing" else 0
                if (index % stride == 0 or (height >= 12 and index % 10 == 0)) and core.phase == "playing":
                    observation = core.get_public_observation()
                    legal = core.get_legal_placements()
                    hold_action = hold_policy.choose(observation, legal)
                    beam_action = beam_policy.choose(observation, legal)
                    if hold_action != beam_action:
                        weights = hold_policy.config.weights
                        a = action_explanation(observation, hold_action, weights, .58)
                        b = action_explanation(observation, beam_action, weights, .58)
                        candidates.append(dict(source=source, seed=seed, step=index,
                            max_height=height, observation=observation,
                            hold_only=a, beam_8=b))
                core.step(action)
    # Preserve both high-risk and ordinary differences; no hidden state enters a decision.
    ranked = sorted(candidates, key=lambda item: (-item["max_height"], item["seed"], item["step"]))
    chosen = ranked[:keep // 2]
    chosen += [row for row in candidates if row not in chosen][:keep-len(chosen)]
    for case in chosen:
        case["counterfactual"] = {
            "hold_first": common_continuation(rows, case, case["hold_only"]["action_id"], counterfactual_horizon),
            "beam_first": common_continuation(rows, case, case["beam_8"]["action_id"], counterfactual_horizon)}
    return dict(created_at=now(), sampled_states=max_per_source,
                sampling_stride=stride, high_risk_extra_stride=10,
                high_risk_height_threshold=12,
                total_disagreements=len(candidates),
                saved=chosen)


def markdown(result, disagreements=None, breakdowns=None):
    cap = result["max_pieces"]
    lines = ["# Traditional AI long-life comparison", "",
        f"{len(result['seeds'])} paired validation seeds; cap {cap} Core placements per seed.",
        "All three policies use unchanged Benchmark v1.0 rules and the same seed set. V1 Adapter is not original web V1.",
        "Every capped observation is right-censored. Restricted means use min(lifetime, cap); paired bootstrap 95% intervals resample whole seeds (10,000 fixed-seed draws). These intervals describe capped outcomes, not uncensored lifetime.",
        "When both policies cap on a seed, their relative lifetime is unknown. Official score depends on levels and survival; equal-piece score is replayed at the shorter observed episode length in each pair.", "",
        "| Mode | Restricted mean pieces [bootstrap 95%] | Median/P90 observed | Death/cap | Mean lines | Lines/piece | Mean score | Decision mean/P95 ms | Total wall h | End-to-end pieces/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        m = result["modes"][mode]
        lo, hi = m["restricted_mean_bootstrap_95ci"]
        lines.append(f"| {mode} | {m['restricted_mean_pieces']:.1f} [{lo:.1f}, {hi:.1f}] | {m['median_observed_pieces']:.1f}/{m['p90_observed_pieces']:.1f} | {m['game_over_rate']:.1%}/{m['truncation_rate']:.1%} | {m['mean_lines']:.1f} | {m['lines_per_placed_piece']:.4f} | {m['mean_official_score']:.1f} | {m['mean_decision_ms']:.3f}/{m['p95_decision_ms']:.3f} | {m['total_wall_seconds']/3600:.2f} | {m['end_to_end_pieces_per_second']:.2f} |")
    lines += ["", "| Pair (B minus A) | Restricted mean Δ [bootstrap 95%] | Both capped | A death/B cap | A cap/B death | Equal-piece score/piece Δ [bootstrap 95%] |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, pair in result["pairs"].items():
        lo, hi = pair["restricted_mean_difference_bootstrap_95ci"]
        slo, shi = pair["mean_equal_piece_score_per_piece_difference_bootstrap_95ci"]
        lines.append(f"| {name} | {pair['restricted_mean_difference']:.1f} [{lo:.1f}, {hi:.1f}] | {pair['both_truncated']} | {pair['a_over_b_truncated']} | {pair['a_truncated_b_over']} | {pair['mean_equal_piece_score_per_piece_difference']:.3f} [{slo:.3f}, {shi:.3f}] |")
    lines += ["", "| Seed | V1 pieces/end | Hold pieces/end | Beam pieces/end | V1 lines/score | Hold lines/score | Beam lines/score |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["seeds_detail"]:
        def pe(mode):
            r = row[mode]
            return f"{r['pieces_survived']}/{'cap' if r['truncated'] else 'over'}"
        def ls(mode):
            r = row[mode]
            return f"{r['lines']}/{r['score']}"
        lines.append(f"| {row['seed']} | {pe('v1')} | {pe('hold')} | {pe('beam')} | {ls('v1')} | {ls('hold')} | {ls('beam')} |")
    if breakdowns is not None:
        lines += ["", "## Official score components on selected seeds", "",
            "Core replay separates official line-clear points and Hard Drop points. Clear counts are for 0/1/2/3/4 lines per placement. This describes observed trajectories; it does not isolate the causal effect of a single decision.", "",
            "| Seed | Mode | Line points | Drop points | 1/2/3/4-line clears |",
            "|---:|---|---:|---:|---:|"]
        for seed in sorted(breakdowns):
            for mode in MODES:
                b = breakdowns[seed][mode]
                clear_text = "/".join(str(b["clears"][str(n)]) for n in range(1, 5))
                lines.append(f"| {seed} | {mode} | {b['line_points']} | {b['drop_points']} | {clear_text} |")
    if disagreements is not None:
        lines += ["", "## Hold-only vs Beam decisions", "",
            f"Scanned up to {disagreements['sampled_states']} placements from each of the first three seeds of both V2 trajectories; evaluated every {disagreements['sampling_stride']}th state and extra states with height ≥12 every 10 placements. Found {disagreements['total_disagreements']} disagreements; full public observations and paths are in `disagreements.json`.",
            "Immediate heuristic favors Hold-only at a disagreement by construction; Beam trades that score for the best next visible placement. The 200-piece counterfactual forks the same Core state, forces each first action, then uses Hold-only for both continuations. This tests the local choice; it does not prove which full policy lives longer. Neither policy sees hidden state.", "",
            "| Seed/step/source | Height | Hold action (immediate/two-step) | Beam action (immediate/two-step) | Fork survival Hold/Beam |",
            "|---|---:|---:|---:|---:|"]
        for r in disagreements["saved"]:
            a, b = r["hold_only"], r["beam_8"]
            fork = r["counterfactual"]
            lines.append(f"| {r['seed']}/{r['step']}/{r['source']} | {r['max_height']} | {a['action_id']} ({a['immediate_heuristic']:.2f}/{a['two_step_value']:.2f}) | {b['action_id']} ({b['immediate_heuristic']:.2f}/{b['two_step_value']:.2f}) | {fork['hold_first']['pieces']}/{fork['beam_first']['pieces']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode}-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disagreements", action="store_true")
    parser.add_argument("--score-breakdown", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs, rows = {}, {}
    for mode in MODES:
        configs[mode], rows[mode] = read_run(getattr(args, f"{mode}_dir"), mode)
    result = summarize(configs, rows)
    atomic(args.output / "paired.json", result)
    disagreements = find_disagreements(rows, result["seeds"]) if args.disagreements else None
    if disagreements is not None:
        atomic(args.output / "disagreements.json", disagreements)
    breakdowns = ({seed: {mode: score_breakdown(rows[mode][seed]) for mode in MODES}
                   for seed in result["seeds"][:3]} if args.score_breakdown else None)
    if breakdowns is not None:
        atomic(args.output / "score_breakdown.json", breakdowns)
    (args.output / "report.md").write_text(markdown(result, disagreements, breakdowns))
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
