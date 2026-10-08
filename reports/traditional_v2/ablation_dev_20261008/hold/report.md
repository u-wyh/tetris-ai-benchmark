# Traditional V2 independent benchmark

Version: v2-1; mode: hold; beam width: 8; lookahead: 0.58.
Seeds: 4; max pieces: 5000; Git base: `130fc5e044e4cf2cefb14f544605ffbaadf673d9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 4 |
| mean_pieces | 5000 |
| median_pieces | 5000.000000 |
| p90_pieces | 5000 |
| mean_lines | 1998.750000 |
| median_lines | 1999.000000 |
| p90_lines | 1999 |
| mean_score | 20988925 |
| median_score | 21001350.000000 |
| p90_score | 21030900 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 5.661865 |
| p95_decision_ms | 7.149891 |
| decisions_per_second | 176.620236 |
| end_to_end_pieces_per_second | 111.779435 |
| lines_per_piece | 0.399750 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 3.223166 |
| mean_future_legal_ms | 0.000000 |
| mean_simulation_features_ms | 1.860462 |
| mean_search_overhead_ms | 0.530503 |
| hold_action_rate | 0.354500 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 310100 | 5000 | 1998 | 21003300 | cap | 1762 |
| 310101 | 5000 | 1999 | 20922100 | cap | 1775 |
| 310102 | 5000 | 1999 | 20999400 | cap | 1817 |
| 310103 | 5000 | 1999 | 21030900 | cap | 1736 |
