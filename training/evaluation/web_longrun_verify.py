"""Atomic, resumable Python Core parity checker for real browser HTTP runs."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

from training.tetris_core.core import TetrisCore


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sealed(path):
    seal = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not seal.exists() or seal.read_text().strip() != digest(path.read_bytes()):
        raise RuntimeError(f"Missing or invalid atomic result: {path}")
    return path.read_bytes()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    seal = path.with_suffix(path.suffix + ".sha256")
    temp = seal.with_name(seal.name + f".{os.getpid()}.tmp")
    temp.write_text(digest(data))
    os.replace(temp, seal)


def committed(run):
    index = 0
    while True:
        n = index + 1
        summary = run / "summaries" / f"{n:06d}.json"
        checkpoint = run / "checkpoints" / f"{n:06d}.pkl"
        if not summary.exists() or not checkpoint.exists():
            break
        sealed(summary)
        saved = pickle.loads(sealed(checkpoint))
        if saved["chunk"] != n or json.loads(summary.read_text())["end_piece"] != saved["pieces"]:
            raise RuntimeError(f"Checkpoint and summary disagree at {n}")
        index = n
    return index


def resume(run, seed):
    index = committed(run)
    if index:
        saved = pickle.loads(sealed(run / "checkpoints" / f"{index:06d}.pkl"))
        if saved["seed"] != seed:
            raise RuntimeError("Seed changed")
        return {"chunk": index, "pieces": saved["pieces"], "browser_save": saved["browser_save"],
                "game_over": saved["core"].phase == "over"}
    return {"chunk": 0, "pieces": 0, "browser_save": None, "game_over": False}


def compare(core, browser, step):
    public = core.get_public_observation()
    for field in ("board", "currentPiece", "next", "hold", "score", "level", "lines", "gameOver"):
        if browser[field] != public[field]:
            raise AssertionError(f"State drift at piece {step}, field {field}")
    for field in ("holdAvailable", "phase"):
        if field in browser and browser[field] != public[field]:
            raise AssertionError(f"State drift at piece {step}, field {field}")
    if browser["placed"] != step:
        raise AssertionError(f"Browser piece count drift at piece {step}")


def verify_chunk(run, seed, pending):
    data = json.loads(pending.read_text())
    previous = resume(run, seed)
    if data["chunk"] != previous["chunk"] + 1 or data["start_piece"] != previous["pieces"]:
        raise RuntimeError("Chunk order or starting piece changed")
    if previous["chunk"]:
        saved = pickle.loads(sealed(run / "checkpoints" / f"{previous['chunk']:06d}.pkl"))
        core = saved["core"]
    else:
        core = TetrisCore(seed)
    for offset, row in enumerate(data["entries"]):
        piece = data["start_piece"] + offset
        compare(core, row["before"], piece)
        legal = [p["actionId"] for p in core.get_legal_placements()]
        if row["action"] not in legal or row["legal_ids"] != legal:
            raise AssertionError(f"Illegal or mismatched Action ID at piece {piece}")
        core.step(row["action"])
        compare(core, row["after"], piece + 1)
        if core.phase == "over" and offset != len(data["entries"]) - 1:
            raise AssertionError("Browser continued after Game Over")
    end = data["start_piece"] + len(data["entries"])
    if data["end_piece"] != end or data["game_over"] != (core.phase == "over"):
        raise AssertionError("Chunk ending status differs")
    chunk = data["chunk"]
    summary = {"chunk": chunk, "start_piece": data["start_piece"], "end_piece": end,
               "steps": len(data["entries"]), "game_over": data["game_over"],
               "score": core.score, "lines": core.lines, "level": core.level,
               "trace_sha256": digest(pending.read_bytes()), "metrics": data["metrics"]}
    checkpoint = {"chunk": chunk, "seed": seed, "pieces": end, "core": core,
                  "browser_save": data["browser_save"]}
    atomic(run / "checkpoints" / f"{chunk:06d}.pkl", pickle.dumps(checkpoint))
    atomic(run / "summaries" / f"{chunk:06d}.json", json.dumps(summary).encode())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("resume", "verify"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pending", type=Path)
    args = parser.parse_args()
    result = resume(args.run, args.seed) if args.command == "resume" else verify_chunk(
        args.run, args.seed, args.pending)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
