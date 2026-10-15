"""Extract full, post-game death context from atomic V3 game files."""
from training.evaluation.ppo_lookahead_v3_audit import RUN
from training.evaluation.ppo_lookahead_v3_queue import read_gzip
from training.train_ppo import atomic_json


def extract():
    outputs = []
    for path in RUN.glob("*/**/seed_*.json.gz"):
        row = read_gzip(path)
        if not row["game_over"]:
            continue
        important = []
        for event in row["replacements"][-20:]:
            important.append({"step": event["step"], "pieces_until_death": row["pieces"]-event["step"],
                              "public_hash": event["public_hash"], "board_heights": event["public_summary"]["heights"],
                              "current_hold_next": {key: event["public_summary"][key] for key in ("current", "hold", "next")},
                              "hold_changed": event["hold_changed"], "ppo": event["ppo"],
                              "search": event["search"], "score_gap": event["score_gap"],
                              "trigger": event["trigger"]})
        result = {"seed": row["seed"], "variant": row["variant"], "cap": row["cap"],
                  "pieces": row["pieces"], "score": row["score"], "game_over": True,
                  "total_replacements": len(row["replacements"]),
                  "last_important_replacements": important,
                  "source": str(path.relative_to(RUN))}
        destination = RUN / "diagnostics/deaths" / path.parent.parent.name / row["variant"] / f"seed_{row['seed']}.json"
        atomic_json(destination, result)
        outputs.append(destination)
    return outputs


if __name__ == "__main__":
    print(len(extract()), flush=True)
