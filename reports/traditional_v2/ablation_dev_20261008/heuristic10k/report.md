# Traditional V2 independent benchmark

Version: v2-1; mode: heuristic; beam width: 8; lookahead: 0.58.
Seeds: 2; max pieces: 10000; Git base: `130fc5e044e4cf2cefb14f544605ffbaadf673d9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 2 |
| mean_pieces | 6797 |
| median_pieces | 6797.000000 |
| p90_pieces | 7421 |
| mean_lines | 2702.500000 |
| median_lines | 2702.500000 |
| p90_lines | 2952 |
| mean_score | 41657100 |
| median_score | 41657100.000000 |
| p90_score | 49220900 |
| game_over_rate | 1 |
| cap_rate | 0 |
| mean_decision_ms | 4.824627 |
| p95_decision_ms | 7.392748 |
| decisions_per_second | 207.269918 |
| end_to_end_pieces_per_second | 121.883194 |
| lines_per_piece | 0.397602 |
| evaluation_saturated | False |
| metric_censored_by_cap | False |
| mean_root_legal_ms | 3.306971 |
| mean_future_legal_ms | 0.000000 |
| mean_simulation_features_ms | 1.166641 |
| mean_search_overhead_ms | 0.317216 |
| hold_action_rate | 0.000000 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 310100 | 6173 | 2453 | 34093300 | Game Over | 0 |
| 310103 | 7421 | 2952 | 49220900 | Game Over | 0 |
