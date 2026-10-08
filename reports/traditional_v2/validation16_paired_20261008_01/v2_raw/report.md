# Traditional V2 independent benchmark

Version: v2-1; mode: beam; beam width: 8; lookahead: 0.58.
Seeds: 4; max pieces: 5000; Git base: `ccebb369776251992b2cf794283dff7c976bebf5`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 4 |
| mean_pieces | 5000 |
| median_pieces | 5000.000000 |
| p90_pieces | 5000 |
| mean_lines | 1999 |
| median_lines | 1999.000000 |
| p90_lines | 1999 |
| mean_score | 20876725 |
| median_score | 20874000.000000 |
| p90_score | 20942000 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 64.041848 |
| p95_decision_ms | 84.408366 |
| decisions_per_second | 15.614790 |
| end_to_end_pieces_per_second | 14.564639 |
| lines_per_piece | 0.399800 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 4.467716 |
| mean_future_legal_ms | 35.145176 |
| mean_simulation_features_ms | 22.582284 |
| mean_search_overhead_ms | 1.160415 |
| hold_action_rate | 0.380400 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100000 | 5000 | 1999 | 20816900 | cap | 1929 |
| 100001 | 5000 | 1999 | 20832600 | cap | 1889 |
| 100002 | 5000 | 1999 | 20942000 | cap | 1904 |
| 100003 | 5000 | 1999 | 20915400 | cap | 1886 |
