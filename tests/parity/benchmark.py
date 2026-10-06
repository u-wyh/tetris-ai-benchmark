"""Measure Python mask throughput on four seeded game states, three runs each."""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.tetris_core import TetrisCore  # noqa: E402

durations = []
counts = []
for seed, locked_pieces in ((12345, 0), (54321, 1), (2026, 2), (8675309, 3)):
    core = TetrisCore(seed)
    for index in range(locked_pieces):
        placements = [item for item in core.get_legal_placements()
                      if item["hold"] == 0 and item["rotation"] == 0]
        key = min if index % 2 == 0 else max
        target_x = key(item["x"] for item in placements)
        chosen = max((item for item in placements if item["x"] == target_x),
                     key=lambda item: item["y"])
        core.step(chosen["actionId"])
    for _ in range(3):
        start = time.perf_counter()
        mask = core.get_action_mask()
        durations.append((time.perf_counter() - start) * 1000)
        counts.append(sum(mask))

average = statistics.mean(durations)
print(json.dumps({"samples": len(durations), "averageMs": round(average, 2),
                  "medianMs": round(statistics.median(durations), 2),
                  "masksPerSecond": round(1000 / average, 1),
                  "averagePlacements": round(statistics.mean(counts), 1)}, indent=2))
