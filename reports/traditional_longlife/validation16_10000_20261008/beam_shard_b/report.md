# Traditional V2 independent benchmark

Version: v2-1; mode: beam; beam width: 8; lookahead: 0.58.
Seeds: 6; max pieces: 10000; Git base: `308bc90427d62bb9fde954072f79503101fb07e9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 6 |
| mean_pieces | 10000 |
| median_pieces | 10000.000000 |
| p90_pieces | 10000 |
| mean_lines | 3998.833333 |
| median_lines | 3999.000000 |
| p90_lines | 3999 |
| mean_score | 83213233.333333 |
| median_score | 83172600.000000 |
| p90_score | 83524600 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 71.769526 |
| p95_decision_ms | 91.531360 |
| decisions_per_second | 13.933490 |
| end_to_end_pieces_per_second | 13.008686 |
| lines_per_piece | 0.399883 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 4.980068 |
| mean_future_legal_ms | 39.264834 |
| mean_simulation_features_ms | 25.387431 |
| mean_search_overhead_ms | 1.332560 |
| hold_action_rate | 0.372650 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100026 | 10000 | 3999 | 83524600 | cap | 3755 |
| 100027 | 10000 | 3999 | 83272700 | cap | 3724 |
| 100028 | 10000 | 3999 | 83054100 | cap | 3834 |
| 100029 | 10000 | 3999 | 83380300 | cap | 3651 |
| 100030 | 10000 | 3998 | 83072500 | cap | 3704 |
| 100031 | 10000 | 3999 | 82975200 | cap | 3691 |
