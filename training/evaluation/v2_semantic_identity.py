"""Bind the historical V2 Hold result to the later, verified fast implementation."""

import json
from pathlib import Path

from training.tetris_core import TetrisCore
from training.traditional.v2 import V2, V2Config

FIXTURE = Path(__file__).parent / "fixtures/v2_hold_historical_prefix.json"
HISTORICAL_SHA256 = "6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4"
OPTIMIZED_SHA256 = "23e4872dbe45d9ec3717f8306c6e53076d7fd3c0cbb8d7c4ceedf97637fe5e0e"


def verify_historical_hold(reference, current_digest):
    """Require a known source version and replay its frozen action/score prefixes."""
    if reference["code_sha256"] != HISTORICAL_SHA256:
        raise RuntimeError("Historical V2 Hold source identity changed")
    if current_digest not in (HISTORICAL_SHA256, OPTIMIZED_SHA256):
        raise RuntimeError("V2 Hold source version is not semantically verified")
    fixture = json.loads(FIXTURE.read_text())
    if (fixture["source_code_sha256"] != HISTORICAL_SHA256 or
            fixture["optimized_code_sha256"] != OPTIMIZED_SHA256 or
            fixture["source_commit"] != "0598b9fb704dbc31bb6ac3b0dadada049714629e"):
        raise RuntimeError("V2 Hold historical fixture metadata changed")
    policy = V2(V2Config(**reference["policy"]))
    for row in fixture["rows"]:
        core = TetrisCore(row["seed"])
        if len(row["actions"]) != fixture["steps_per_seed"]:
            raise RuntimeError("V2 Hold historical fixture length changed")
        for step, expected in enumerate(row["actions"]):
            chosen = policy.choose(core.get_public_observation(), core.get_legal_placements())
            if chosen != expected:
                raise RuntimeError(f"V2 Hold semantic mismatch at seed {row['seed']}, step {step}")
            core.step(chosen)
        if (core.score, core.lines) != (row["score"], row["lines"]):
            raise RuntimeError(f"V2 Hold score/lines mismatch at seed {row['seed']}")
    return V2Config(**reference["policy"])
