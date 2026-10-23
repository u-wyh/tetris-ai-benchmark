# Evaluation legal placement edge parity

2026-10-23 12:54 +0800. Protocol: [edge parity design](ppo_fast_legal_edge_parity_design.md).

The evaluation-only bitboard placement generator matched formal Core exactly
on all 84 fixed states: seven current pieces × four board patterns × three
hold states. Comparison covered ordered action IDs, paths, and occupied cells.
The empty-board cases also confirmed legal placements exist and hold actions
appear exactly when hold is available. The existing seeded environment and
feature tests passed alongside these cases: **97/97** in 2.47 seconds.

Evidence: `runs/ppo_fast_legal_edge_parity_20261023/tests.log`.
Both 8080 root and 8081 `/api/health` returned HTTP 200 after the test.
This adds differential regression coverage; it does not claim a throughput
or policy improvement. No training steps, formal Core, models, or web defaults
were changed.
