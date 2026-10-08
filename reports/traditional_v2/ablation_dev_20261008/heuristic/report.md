# Traditional V2 independent benchmark

Version: v2-1; mode: heuristic; beam width: 8; lookahead: 0.58.
Seeds: 4; max pieces: 5000; Git base: `130fc5e044e4cf2cefb14f544605ffbaadf673d9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 4 |
| mean_pieces | 4734.250000 |
| median_pieces | 5000.000000 |
| p90_pieces | 5000 |
| mean_lines | 1886 |
| median_lines | 1993.000000 |
| p90_lines | 1996 |
| mean_score | 20210850 |
| median_score | 22364650.000000 |
| p90_score | 22446800 |
| game_over_rate | 0.250000 |
| cap_rate | 0.750000 |
| mean_decision_ms | 4.293957 |
| p95_decision_ms | 5.241524 |
| decisions_per_second | 232.885437 |
| end_to_end_pieces_per_second | 136.539242 |
| lines_per_piece | 0.398374 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 2.962370 |
| mean_future_legal_ms | 0.000000 |
| mean_simulation_features_ms | 1.027508 |
| mean_search_overhead_ms | 0.276282 |
| hold_action_rate | 0.000000 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 310100 | 5000 | 1996 | 22408400 | cap | 0 |
| 310101 | 5000 | 1991 | 22320900 | cap | 0 |
| 310102 | 3937 | 1562 | 13667300 | Game Over | 0 |
| 310103 | 5000 | 1995 | 22446800 | cap | 0 |
