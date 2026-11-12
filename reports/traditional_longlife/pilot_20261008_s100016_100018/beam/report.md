# Traditional V2 independent benchmark

Version: v2-1; mode: beam; beam width: 8; lookahead: 0.58.
Seeds: 3; max pieces: 10000; Git base: `b57e0fea560ed4abdc14c901516a32253a2add24`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 3 |
| mean_pieces | 10000 |
| median_pieces | 10000 |
| p90_pieces | 10000 |
| mean_lines | 3998.666667 |
| median_lines | 3999 |
| p90_lines | 3999 |
| mean_score | 83292266.666667 |
| median_score | 83388000 |
| p90_score | 83425200 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 54.010107 |
| p95_decision_ms | 73.417037 |
| decisions_per_second | 18.515053 |
| end_to_end_pieces_per_second | 17.279055 |
| lines_per_piece | 0.399867 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 3.780161 |
| mean_future_legal_ms | 29.753298 |
| mean_simulation_features_ms | 19.099798 |
| mean_search_overhead_ms | 0.939978 |
| hold_action_rate | 0.369967 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100016 | 10000 | 3998 | 83425200 | cap | 3711 |
| 100017 | 10000 | 3999 | 83063600 | cap | 3704 |
| 100018 | 10000 | 3999 | 83388000 | cap | 3684 |
