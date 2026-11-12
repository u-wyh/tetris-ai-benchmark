# Traditional V2 independent benchmark

Version: v2-1; mode: beam; beam width: 8; lookahead: 0.58.
Seeds: 10; max pieces: 100000; Git base: `308bc90427d62bb9fde954072f79503101fb07e9`; code SHA-256: `6dc6c6d31fc3d02fdc3ee0bd369dbd07aac438b14f98bce2803da546c4a469f4`.
Weights: `{"aggregate_height": -0.47, "bumpiness": -0.28, "cleared": 8.2, "column_transitions": -0.22, "covered_holes": -0.16, "holes": -4.2, "max_height": -0.18, "row_transitions": -0.14, "top_risk": -0.9, "wells": -0.08}`.
Only public Board/Current/Next 3/Hold/score/lines/level are supplied to the policy. All searched placements come from canonical BFS.
One action means one Core placement; reaching the piece cap is truncation, not death.

| Metric | Value |
|---|---:|
| num_seeds | 10 |
| mean_pieces | 100000 |
| median_pieces | 100000.000000 |
| p90_pieces | 100000 |
| mean_lines | 39999 |
| median_lines | 39999.000000 |
| p90_lines | 39999 |
| mean_score | 8304122670 |
| median_score | 8303973350.000000 |
| p90_score | 8313979300 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 47.371379 |
| p95_decision_ms | 70.177628 |
| decisions_per_second | 21.109793 |
| end_to_end_pieces_per_second | 19.588876 |
| lines_per_piece | 0.399990 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 3.301600 |
| mean_future_legal_ms | 26.170111 |
| mean_simulation_features_ms | 16.824201 |
| mean_search_overhead_ms | 0.839275 |
| hold_action_rate | 0.372354 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100016 | 100000 | 39999 | 8286838100 | cap | 37129 |
| 100017 | 100000 | 39999 | 8311468900 | cap | 37341 |
| 100018 | 100000 | 39999 | 8290868200 | cap | 37325 |
| 100019 | 100000 | 39999 | 8308953600 | cap | 37260 |
| 100020 | 100000 | 39999 | 8313979300 | cap | 37145 |
| 100021 | 100000 | 39999 | 8299370000 | cap | 37242 |
| 100022 | 100000 | 39999 | 8301765700 | cap | 37237 |
| 100023 | 100000 | 39999 | 8303649800 | cap | 37416 |
| 100024 | 100000 | 39999 | 8304296900 | cap | 37029 |
| 100025 | 100000 | 39999 | 8320036200 | cap | 37230 |
