"""Reproduce the paper's tables and survival figure from the archived episodes."""

import csv
import math
import random
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
PROTOCOLS = {
    "A1": (1421000, 32, 100_000),
    "A2": (710100, 32, 100_000),
    "B": (1400000, 8, 3_000_000),
}
NAMES = {
    "bc": "BC Champion",
    "B12": "Hybrid B12",
    "ppo": "PPO 30M",
    "traditional_v2_hold": "V2 Hold-only",
}
STYLES = {
    "bc": ("#17625f", "-"),
    "B12": ("#28578a", "--"),
    "ppo": ("#a53624", "-."),
    "traditional_v2_hold": ("#57555f", ":"),
}
ITERATIONS = 20_000
BOOTSTRAP_SEED = 20261123


def load_episodes():
    episodes = {}
    with (HERE / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            key = raw["protocol"], raw["policy"], int(raw["seed"])
            if key in episodes:
                raise ValueError(f"Duplicate episode {key}")
            row = {k: int(raw[k]) for k in (
                "cap", "pieces", "game_over", "capped", "score",
                "clears_1", "clears_2", "clears_3", "clears_4",
            )}
            episodes[key] = row
    expected = {
        (protocol, policy, seed)
        for protocol, (start, count, _) in PROTOCOLS.items()
        for policy in (NAMES if protocol == "B" else ("bc", "B12", "ppo"))
        for seed in range(start, start + count)
    }
    if episodes.keys() != expected:
        raise ValueError("Archive does not contain the complete paired design")
    for key, row in episodes.items():
        _, _, cap = PROTOCOLS[key[0]]
        if (row["cap"] != cap or not 0 < row["pieces"] <= cap
                or row["game_over"] + row["capped"] != 1
                or (row["capped"] and row["pieces"] != cap)):
            raise ValueError(f"Inconsistent episode outcome {key}")
    return episodes


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def rows_for(episodes, protocol, policy):
    start, count, _ = PROTOCOLS[protocol]
    return [episodes[protocol, policy, seed] for seed in range(start, start + count)]


def tex_number(value, places=0):
    return f"{value:,.{places}f}".replace(",", r"\,")


def table_row(*cells):
    return " & ".join(map(str, cells)) + r" \\"


def summary_rows(episodes, protocols):
    result = []
    for protocol in protocols:
        if result:
            result.append(r"\midrule")
        order = ("bc", "traditional_v2_hold", "B12", "ppo") if protocol == "B" else ("bc", "B12", "ppo")
        for index, policy in enumerate(order):
            rows = rows_for(episodes, protocol, policy)
            pieces = [row["pieces"] for row in rows]
            total = sum(row[f"clears_{n}"] for row in rows for n in range(1, 5))
            four_percent = 100 * sum(row["clears_4"] for row in rows) / total
            result.append(table_row(
                protocol if index == 0 else "", NAMES[policy],
                tex_number(mean(pieces)), tex_number(percentile(pieces, .1)),
                f'{sum(row["capped"] for row in rows)}/{len(rows)}',
                tex_number(mean(row["score"] for row in rows) / 1e8, 2),
                f"{four_percent:.2f}",
            ))
    return "\n".join(result)


def effect_rows(episodes):
    contrasts = (
        ("BC Champion $-$ PPO 30M", "bc", "ppo", "pieces", 1_000, 2),
        ("Hybrid B12 $-$ PPO 30M", "B12", "ppo", "pieces", 1_000, 2),
        ("Hybrid B12 $-$ PPO 30M", "B12", "ppo", "score", 1e8, 2),
        ("Hybrid B12 $-$ BC Champion", "B12", "bc", "score", 1e8, 2),
    )
    long_contrast = ("BC Champion $-$ V2 Hold-only", "bc", "traditional_v2_hold", "pieces", 1_000, 2)
    rng = random.Random(BOOTSTRAP_SEED)
    result = []
    for protocol, comparisons in (("A1", contrasts), ("A2", contrasts), ("B", (long_contrast,))):
        if result:
            result.append(r"\midrule")
        for index, (label, first, second, key, scale, places) in enumerate(comparisons):
            differences = [
                (a[key] - b[key]) / scale for a, b in zip(
                    rows_for(episodes, protocol, first), rows_for(episodes, protocol, second),
                )
            ]
            samples = [mean(rng.choices(differences, k=len(differences))) for _ in range(ITERATIONS)]
            low, high = (percentile(samples, p) for p in (.025, .975))
            result.append(table_row(
                protocol if index == 0 else "", label,
                "存活（千块）" if key == "pieces" else "总分（亿）",
                tex_number(mean(differences), places),
                f"[{tex_number(low, places)}, {tex_number(high, places)}]",
            ))
    return "\n".join(result)


def survival_figure(episodes):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.85, 2.65), sharey=True)
    for ax, (protocol, (_, _, cap)) in zip(axes, PROTOCOLS.items()):
        scale = 1e6 if protocol == "B" else 1e3
        policies = NAMES if protocol == "B" else ("bc", "B12", "ppo")
        for policy in policies:
            rows = rows_for(episodes, protocol, policy)
            grid = sorted({0, cap, *(row["pieces"] + 1 for row in rows if row["pieces"] < cap)})
            reached = [sum(row["pieces"] >= x for row in rows) / len(rows) for x in grid]
            color, linestyle = STYLES[policy]
            ax.step([x / scale for x in grid], reached, where="post", label=NAMES[policy],
                    color=color, linestyle=linestyle, linewidth=1.8)
        ax.set_title(protocol)
        ax.set_xlim(0, cap / scale)
        ax.set_ylim(-.02, 1.04)
        ax.set_xticks([0, 1, 2, 3] if protocol == "B" else [0, 25, 50, 75, 100])
        ax.set_xlabel("Pieces (millions)" if protocol == "B" else "Pieces (thousands)")
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.24)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Fraction reaching threshold")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False,
               bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.083, right=.985, top=.89, bottom=.31, wspace=.28)
    fig.savefig(HERE / "survival.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    episodes = load_episodes()
    for name, command, rows in (
        ("results-short", "ResultsShort", summary_rows(episodes, ("A1", "A2"))),
        ("results-long", "ResultsLong", summary_rows(episodes, ("B",))),
        ("results-effects", "ResultsEffects", effect_rows(episodes)),
    ):
        (HERE / f"{name}.tex").write_text(
            "% Generated from episodes.csv by analysis.py.\n"
            + rf"\newcommand{{\{command}}}{{%" + "\n" + rows + "\n}\n",
            encoding="utf-8",
        )
    survival_figure(episodes)


if __name__ == "__main__":
    main()
