# Traditional V2 independent benchmark

Version: v2-1; mode: hold; beam width: 8; lookahead: 0.58.
Seeds: 16; max pieces: 10000; Git base: `b57e0fea560ed4abdc14c901516a32253a2add24`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 16 |
| mean_pieces | 10000 |
| median_pieces | 10000.000000 |
| p90_pieces | 10000 |
| mean_lines | 3998.750000 |
| median_lines | 3999.000000 |
| p90_lines | 3999 |
| mean_score | 83637306.250000 |
| median_score | 83558150.000000 |
| p90_score | 84020900 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 6.675092 |
| p95_decision_ms | 9.875905 |
| decisions_per_second | 149.810673 |
| end_to_end_pieces_per_second | 95.182586 |
| lines_per_piece | 0.399875 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 3.745309 |
| mean_future_legal_ms | 0.000000 |
| mean_simulation_features_ms | 2.227181 |
| mean_search_overhead_ms | 0.643887 |
| hold_action_rate | 0.348456 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100016 | 10000 | 3999 | 83937700 | cap | 3399 |
| 100017 | 10000 | 3999 | 83895100 | cap | 3468 |
| 100018 | 10000 | 3999 | 83582200 | cap | 3448 |
| 100019 | 10000 | 3999 | 83332000 | cap | 3509 |
| 100020 | 10000 | 3998 | 83269400 | cap | 3511 |
| 100021 | 10000 | 3999 | 83471800 | cap | 3494 |
| 100022 | 10000 | 3998 | 84020900 | cap | 3497 |
| 100023 | 10000 | 3999 | 83921500 | cap | 3403 |
| 100024 | 10000 | 3999 | 83225000 | cap | 3556 |
| 100025 | 10000 | 3998 | 83365900 | cap | 3515 |
| 100026 | 10000 | 3999 | 83662000 | cap | 3445 |
| 100027 | 10000 | 3999 | 83525500 | cap | 3581 |
| 100028 | 10000 | 3998 | 84358400 | cap | 3478 |
| 100029 | 10000 | 3999 | 83317200 | cap | 3443 |
| 100030 | 10000 | 3999 | 83778200 | cap | 3518 |
| 100031 | 10000 | 3999 | 83534100 | cap | 3488 |
