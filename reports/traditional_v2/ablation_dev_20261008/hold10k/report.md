# Traditional V2 independent benchmark

Version: v2-1; mode: hold; beam width: 8; lookahead: 0.58.
Seeds: 2; max pieces: 10000; Git base: `130fc5e044e4cf2cefb14f544605ffbaadf673d9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 2 |
| mean_pieces | 10000 |
| median_pieces | 10000.000000 |
| p90_pieces | 10000 |
| mean_lines | 3998.500000 |
| median_lines | 3998.500000 |
| p90_lines | 3999 |
| mean_score | 83352700 |
| median_score | 83352700.000000 |
| p90_score | 83571500 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 6.346567 |
| p95_decision_ms | 9.414727 |
| decisions_per_second | 157.565496 |
| end_to_end_pieces_per_second | 99.890356 |
| lines_per_piece | 0.399850 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 3.584341 |
| mean_future_legal_ms | 0.000000 |
| mean_simulation_features_ms | 2.100387 |
| mean_search_overhead_ms | 0.607702 |
| hold_action_rate | 0.351050 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 310100 | 10000 | 3999 | 83133900 | cap | 3538 |
| 310103 | 10000 | 3998 | 83571500 | cap | 3483 |
