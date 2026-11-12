# Matched public-state decision profile

15 observations × 3 repetitions. One root BFS shared by all policies; no Action Mask. The policies see only public observations.
Timing excludes Core.step and record I/O. Batch-run timing includes V1's Action Mask generation, so use this profile for policy-cost comparison under matched conditions.

| Mode | Mean total ms | Median total ms | Mean policy ms |
|---|---:|---:|---:|
| v1 | 47.683 | 42.907 | 44.183 |
| hold | 6.498 | 6.426 | 2.998 |
| beam | 50.623 | 49.366 | 47.123 |

Common root BFS mean: 3.500 ms.
