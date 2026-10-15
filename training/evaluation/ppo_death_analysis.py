"""Episode-level death windows, matched survivors, and exploratory mechanisms."""
import json
import statistics
from collections import Counter

import numpy as np

from training.evaluation.ppo_death_capture import RUN
from training.evaluation.ppo_death_replay import load_episode
from training.train_ppo import atomic_json

WINDOWS = (5000, 1000, 500, 100, 20)


def _record(row):
    return {key: row[key] for key in ("step", "max_height", "holes", "covered_holes",
            "total_height", "row_transitions", "column_transitions", "bumpiness",
            "deepest_well_column", "deepest_well_depth", "birth_zone", "four_rate_100",
            "hold_rate_100", "score", "action_probability", "heights")}


def windows(rows):
    result = {}
    for width in WINDOWS:
        segment = rows[max(0, len(rows)-width):]
        first, last = segment[0], segment[-1]
        result[str(width)] = {"start_step": first["step"], "end_step": last["step"],
            "max_height_start": first["max_height"], "max_height_end": last["max_height"],
            "max_height_peak": max(r["max_height"] for r in segment),
            "holes_start": first["holes"], "holes_end": last["holes"],
            "covered_holes_start": first["covered_holes"], "covered_holes_end": last["covered_holes"],
            "well_start": first["deepest_well_depth"], "well_end": last["deepest_well_depth"],
            "birth_zone_peak": max(r["birth_zone"] for r in segment),
            "four_count": sum(r["cleared"] == 4 for r in segment),
            "hold_count": sum(r["used_hold"] for r in segment),
            "mean_action_probability": statistics.mean(r["action_probability"] for r in segment),
            "actual_width": len(segment)}
    return result


def match_survivor(target, survival_rows, excluded_seed, target_step):
    """One control per death, matching an already known survivor near the same step."""
    if not survival_rows:
        return None
    candidates = []
    for seed, rows in survival_rows.items():
        if seed == excluded_seed:
            continue
        for row in rows:
            index = row["step"]
            if abs(index-target_step) > 500:
                continue
            distance = (abs(index-target_step)/500 + abs(row["max_height"]-target["max_height"])/4
                        + abs(row["holes"]-target["holes"])/4
                        + abs(row["four_counts"]["1000"]-target["four_counts"]["1000"])/20
                        + abs(row["deepest_well_depth"]-target["deepest_well_depth"])/4)
            candidates.append((distance, seed, index, row))
    if not candidates:
        return None
    distance, seed, index, row = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {"seed": seed, "step": index, "distance": distance, "death_state": _record(target),
            "survivor_state": _record(row), "survived_to_cap": True,
            "match_variables": ["step", "max_height", "holes", "four_count_1000", "deepest_well_depth"]}


def classify(win, rows):
    labels, evidence = [], {}
    w500, w1000, w5000 = win["500"], win["1000"], win["5000"]
    if w500["max_height_end"]-w500["max_height_start"] >= 4:
        labels.append("A_rapid_stack")
        evidence["A_rapid_stack"] = "max height increased >=4 in final 500"
    if w5000["holes_end"]-w5000["holes_start"] >= 3 and w5000["covered_holes_end"] >= w5000["covered_holes_start"]:
        labels.append("B_hole_accumulation")
        evidence["B_hole_accumulation"] = "holes +>=3 with covered holes nondecreasing in final 5000"
    if (w5000["well_start"]-w5000["well_end"] >= 3 and
        w1000["four_count"]/max(1,w1000["actual_width"]) <
        .75*w5000["four_count"]/max(1,w5000["actual_width"])):
        labels.append("C_well_and_four_decline")
        evidence["C_well_and_four_decline"] = "deepest well shrank >=3 and four-clear rate fell"
    # A hold-rate change is exploratory evidence, not proof of scheduling causation.
    earlier = rows[max(0,len(rows)-5000):max(0,len(rows)-1000)]
    earlier_rate = sum(r["used_hold"] for r in earlier)/max(1,len(earlier))
    recent_rate = w1000["hold_count"]/max(1,w1000["actual_width"])
    if abs(recent_rate-earlier_rate) >= .12:
        labels.append("D_hold_pattern_change")
        evidence["D_hold_pattern_change"] = "hold use rate changed >=12pp; causal role unproven"
    if not labels:
        labels.append("F_insufficient_pattern")
    return labels, evidence


def analyze():
    capture = json.loads((RUN / "capture_completed.json").read_text())
    seeds = capture["initial_seeds"] + capture["extra_seeds"]
    episodes = {}
    survivors = {}
    for seed in seeds:
        meta = json.loads((RUN / "traces" / f"seed_{seed}.json").read_text())
        episodes[seed] = meta
        if meta["capped"]:
            _, rows, _ = load_episode(seed)
            survivors[seed] = rows[::100]
            del rows
    results = []
    for seed, meta in episodes.items():
        if not meta["game_over"]:
            continue
        _, rows, _ = load_episode(seed)
        win = windows(rows)
        target_step = max(0, meta["pieces"]-1000)
        target = rows[target_step]
        match = match_survivor(target, survivors, seed, target_step)
        labels, evidence = classify(win, rows)
        results.append({"seed": seed, "pieces": meta["pieces"], "score": meta["score"],
                        "trace_sha256": meta["trace_sha256"], "windows": win,
                        "target_step": target_step, "match": match,
                        "labels": labels, "label_evidence": evidence})
        del rows
    result = {"episode_count": len(seeds), "death_count": len(results),
              "right_censored_count": len(survivors),
              "mean_pieces": statistics.mean(meta["pieces"] for meta in episodes.values()),
              "p10_pieces": float(np.percentile([meta["pieces"] for meta in episodes.values()],10)),
              "class_counts": dict(Counter(label for row in results for label in row["labels"])),
              "death_episodes": results}
    atomic_json(RUN / "death_analysis.json", result)
    return result


def figures(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    death_rows = result["death_episodes"]
    if not death_rows:
        return []
    axes_names = (("max_height", "最高列"), ("holes", "洞数"),
                  ("deepest_well_depth", "最深井"), ("four_rate_100", "最近100块四消率"))
    representative = sorted(death_rows, key=lambda x:x["pieces"])[len(death_rows)//2]
    _, rows, _ = load_episode(representative["seed"])
    tail = rows[max(0,len(rows)-5000):]
    fig, axes = plt.subplots(4,1,figsize=(11,9),sharex=True)
    for ax,(key,label) in zip(axes,axes_names):
        ax.plot([r["step"] for r in tail],[r[key] for r in tail],linewidth=.8)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("已放置块数")
    fig.suptitle(f"PPO 死亡前轨迹 seed {representative['seed']}")
    fig.tight_layout()
    output = RUN / "figures" / f"death_seed_{representative['seed']}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=150)
    plt.close(fig)
    return [str(output)]


if __name__ == "__main__":
    row = analyze()
    print(row["episode_count"], row["death_count"], figures(row), flush=True)
