# Traditional V2 independent benchmark

Version: v2-1; mode: beam; beam width: 8; lookahead: 0.58.
Seeds: 7; max pieces: 10000; Git base: `308bc90427d62bb9fde954072f79503101fb07e9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 7 |
| mean_pieces | 10000 |
| median_pieces | 10000 |
| p90_pieces | 10000 |
| mean_lines | 3999 |
| median_lines | 3999 |
| p90_lines | 3999 |
| mean_score | 83120442.857143 |
| median_score | 83155300 |
| p90_score | 83448400 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 68.796606 |
| p95_decision_ms | 90.511995 |
| decisions_per_second | 14.535601 |
| end_to_end_pieces_per_second | 13.567212 |
| lines_per_piece | 0.399900 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 4.773360 |
| mean_future_legal_ms | 37.701718 |
| mean_simulation_features_ms | 24.302309 |
| mean_search_overhead_ms | 1.275347 |
| hold_action_rate | 0.371214 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100019 | 10000 | 3999 | 83155300 | cap | 3610 |
| 100020 | 10000 | 3999 | 83448400 | cap | 3718 |
| 100021 | 10000 | 3999 | 83135000 | cap | 3779 |
| 100022 | 10000 | 3999 | 82581200 | cap | 3664 |
| 100023 | 10000 | 3999 | 83384300 | cap | 3792 |
| 100024 | 10000 | 3999 | 82780300 | cap | 3723 |
| 100025 | 10000 | 3999 | 83358600 | cap | 3699 |
