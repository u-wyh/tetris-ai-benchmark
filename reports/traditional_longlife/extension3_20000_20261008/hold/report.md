# Traditional V2 independent benchmark

Version: v2-1; mode: hold; beam width: 8; lookahead: 0.58.
Seeds: 3; max pieces: 20000; Git base: `308bc90427d62bb9fde954072f79503101fb07e9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 3 |
| mean_pieces | 20000 |
| median_pieces | 20000 |
| p90_pieces | 20000 |
| mean_lines | 7999 |
| median_lines | 7999 |
| p90_lines | 7999 |
| mean_score | 333091766.666667 |
| median_score | 332765400 |
| p90_score | 334320700 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 8.427740 |
| p95_decision_ms | 11.769223 |
| decisions_per_second | 118.655771 |
| end_to_end_pieces_per_second | 75.436459 |
| lines_per_piece | 0.399950 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 4.727914 |
| mean_future_legal_ms | 0.000000 |
| mean_simulation_features_ms | 2.800600 |
| mean_search_overhead_ms | 0.822776 |
| hold_action_rate | 0.348533 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100016 | 20000 | 7999 | 332189200 | cap | 6949 |
| 100017 | 20000 | 7999 | 334320700 | cap | 6982 |
| 100018 | 20000 | 7999 | 332765400 | cap | 6981 |
