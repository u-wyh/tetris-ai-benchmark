"""Run paired, three-million-piece games after the curriculum queue completes."""
import hashlib
import json
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from training.evaluation.failure_curriculum_eval import RUN as TRAIN_RUN, atomic_pickle, evaluate
from training.evaluation.fast_legal import get_legal_placements as fast_placements
from training.tetris_core import TetrisCore
from training.traditional.v2 import V2, V2Config
from training.train_ppo import ROOT, atomic_json, now
from training.evaluation.hybrid_queue import digest

SEEDS = tuple(range(1400000, 1400008))
CAP = 3000000
WORKERS = 8
CHECKPOINT_INTERVAL = 10000
RUN = ROOT / "runs/post_curriculum_3m"
PHASE = "post_curriculum_3m_fastbfs"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def traditional(seed, code_hash):
    core, policy = TetrisCore(seed), V2(V2Config(mode="hold"))
    started = time.monotonic()
    clears = [0] * 5
    milestones = []
    pieces = 0
    checkpoint = RUN / "traditional_v2_hold" / f"seed_{seed}.checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["identity"] != (seed, CAP, code_hash):
            raise RuntimeError("Traditional long game checkpoint protocol mismatch")
        core = saved["core"]
        pieces, clears, milestones = saved["pieces"], saved["clears"], saved["milestones"]
        if not 0 < pieces < CAP or core.phase != "playing":
            raise RuntimeError("Traditional long game checkpoint step mismatch")
        started -= saved["elapsed_seconds"]
    while core.phase == "playing" and pieces < CAP:
        public = core.get_public_observation()
        placements = fast_placements(public)
        if pieces < 100 or pieces % 1000 == 0:
            if placements != core.get_legal_placements():
                raise RuntimeError("Accelerated V2 legal placements differ from official Core")
        action = policy.choose(public, placements)
        if action not in {placement["actionId"] for placement in placements}:
            raise RuntimeError("Traditional V2 Hold chose an illegal action")
        previous_lines = core.lines
        if "get_legal_placements" in core.__dict__:
            raise RuntimeError("Core legal placements already overridden")
        core.get_legal_placements = lambda: placements
        try:
            core.step(action)
        finally:
            del core.get_legal_placements
        pieces += 1
        clears[core.lines - previous_lines] += 1
        if pieces % 1000 == 0:
            milestones.append({"pieces": pieces, "score": core.score})
        if pieces % CHECKPOINT_INTERVAL == 0:
            atomic_json(RUN / f"traditional_progress_{seed}.json", {"seed": seed,
                        "pieces": pieces, "at": now()})
            if pieces < CAP and core.phase == "playing":
                atomic_pickle(checkpoint, {"identity": (seed, CAP, code_hash),
                                           "core": core, "pieces": pieces,
                                           "clears": clears, "milestones": milestones,
                                           "elapsed_seconds": time.monotonic() - started})
    return {"variant": "traditional_v2_hold", "seed": seed, "cap": CAP,
            "model_sha256": {"traditional_v2_code": code_hash},
            "pieces": pieces, "game_over": core.phase == "over",
            "capped": core.phase != "over" and pieces == CAP,
            "score": core.score, "clears_0_to_4": clears,
            "score_milestones": milestones,
            "wall_seconds": time.monotonic() - started}


def traditional_task(seed, code_hash):
    import torch
    torch.set_num_threads(1)
    row = traditional(seed, code_hash)
    atomic_json(RUN / "traditional_v2_hold" / f"seed_{seed}.json", row)
    (RUN / "traditional_v2_hold" / f"seed_{seed}.checkpoint.pkl").unlink(missing_ok=True)
    return seed


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    protocol = {"seeds": list(SEEDS), "games_per_model": len(SEEDS),
                "cap_per_game": CAP, "workers": WORKERS,
                "inference": "candidate_argmax_v1",
                "legal_placements": "bitboard_public_bfs_v1",
                "variants": ["ppo", "bc", "B12", "normal", "curriculum", "traditional_v2_hold"],
                "training_dependency": "ppo30m_failure_curriculum/completed"}
    path = RUN / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        old = json.loads(path.read_text())
        old_stage = json.loads((RUN / "status.json").read_text())["stage"]
        if (old.get("cap_per_game") in (100000, CAP) and
                old.get("games_per_model") in (30, 4, 8) and
                not list(RUN.rglob("seed_*.json")) and
                old_stage in ("waiting_for_curriculum", "running", "failed") and
                (old_stage == "waiting_for_curriculum" or
                 old.get("legal_placements") is None)):
            archive = RUN / f"superseded_protocol_{old['games_per_model']}_workers_{old.get('workers', 4)}_canonical.json"
            if archive.exists() and json.loads(archive.read_text()) != old:
                raise RuntimeError("Previous protocol archive differs")
            atomic_json(archive, old)
        else:
            raise RuntimeError("Long evaluation protocol changed")
    atomic_json(path, protocol)
    while True:
        state = json.loads((TRAIN_RUN / "status.json").read_text())
        if state["stage"] == "completed":
            break
        if state["stage"] == "failed":
            raise RuntimeError("Curriculum queue failed; long evaluation stopped")
        atomic_json(RUN / "status.json", {"stage": "waiting_for_curriculum", "at": now()})
        time.sleep(30)
    old_phase = TRAIN_RUN / "post_curriculum_3m"
    new_phase = TRAIN_RUN / PHASE
    allowed = (old_phase, new_phase)
    for seed in SEEDS:
        for base in (ROOT / "runs", ROOT / "reports"):
            if any(not (p.is_relative_to(RUN) or
                        any(p.is_relative_to(folder) for folder in allowed))
                   for p in base.rglob(f"seed_{seed}.json")):
                raise RuntimeError(f"Seed already used: {seed}")
    copied = []
    if old_phase.exists():
        old_protocol = json.loads((old_phase / "protocol.json").read_text())
        if (old_protocol["seeds"] != list(SEEDS) or old_protocol["cap"] != CAP or
                old_protocol.get("deterministic_inference") != "candidate_argmax_v1"):
            raise RuntimeError("Previous completed episodes use a different protocol")
        for source in old_phase.glob("*/seed_*.json"):
            row = json.loads(source.read_text())
            if (row["seed"] not in SEEDS or row["cap"] != CAP or
                    row["variant"] not in protocol["variants"][:-1] or
                    row["model_sha256"] != old_protocol["hashes"]):
                raise RuntimeError(f"Invalid previous episode: {source}")
            destination = new_phase / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and digest(destination) != digest(source):
                raise RuntimeError(f"Previously imported episode changed: {destination}")
            if not destination.exists():
                shutil.copy2(source, destination)
            copied.append({"source": str(source.relative_to(ROOT)),
                           "sha256": digest(source), "destination": str(destination.relative_to(ROOT))})
    atomic_json(RUN / "imported_canonical_games.json", copied)
    gate_path = TRAIN_RUN / "development_gate.json"
    gate = json.loads(gate_path.read_text())
    task_number = 489 if gate["extend_curriculum"] else 123
    atomic_json(RUN / "status.json", {"stage": "running", "at": now(),
                                      "pid": os.getpid(), "curriculum_task": task_number})
    report = evaluate(PHASE, tuple(protocol["variants"][:-1]), SEEDS,
                      CAP, task_number, workers=WORKERS, fast=True,
                      fast_legal=True)
    code_hash = sha256(ROOT / "training/traditional/v2.py")
    folder = RUN / "traditional_v2_hold"
    folder.mkdir(exist_ok=True)
    pending = []
    for seed in SEEDS:
        destination = folder / f"seed_{seed}.json"
        if destination.exists():
            row = json.loads(destination.read_text())
            if (row["seed"] != seed or row["cap"] != CAP or
                    row["model_sha256"]["traditional_v2_code"] != code_hash):
                raise RuntimeError(f"Traditional episode protocol mismatch: {seed}")
        else:
            pending.append(seed)
    atomic_json(RUN / "status.json", {"stage": "traditional_v2_hold",
                "completed": len(SEEDS) - len(pending), "total": len(SEEDS),
                "workers": WORKERS, "at": now(), "pid": os.getpid()})
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(traditional_task, seed, code_hash) for seed in pending]
        for future in as_completed(futures):
            future.result()
            atomic_json(RUN / "status.json", {"stage": "traditional_v2_hold",
                        "completed": len(SEEDS) - sum(not (folder / f"seed_{s}.json").exists()
                                                      for s in SEEDS),
                        "total": len(SEEDS), "workers": WORKERS, "at": now(),
                        "pid": os.getpid()})
    from training.evaluation.failure_curriculum_eval import _summary
    report["summary"]["traditional_v2_hold"] = _summary(
        [json.loads((folder / f"seed_{seed}.json").read_text()) for seed in SEEDS])
    atomic_json(RUN / "summary.json", report)
    atomic_json(RUN / "status.json", {"stage": "completed", "at": now(),
                                      "pid": os.getpid(), "curriculum_task": task_number})
    with (ROOT / "OPERATIONS.md").open("a") as stream:
        stream.write(f"\n- {now()}｜完成课程训练后六策略、每局上限 300 万块的 8 种子长局比较。依据：runs/post_curriculum_3m/summary.json。\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        RUN.mkdir(parents=True, exist_ok=True)
        atomic_json(RUN / "status.json", {"stage": "failed", "at": now(), "error": repr(error)})
        with (ROOT / "OPERATIONS.md").open("a") as stream:
            stream.write(f"\n- {now()}｜课程训练后长局比较失败并停止。依据：runs/post_curriculum_3m/status.json。\n")
        raise
