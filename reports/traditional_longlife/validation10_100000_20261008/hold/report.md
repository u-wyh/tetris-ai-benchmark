# Traditional V2 independent benchmark

Version: v2-1; mode: hold; beam width: 8; lookahead: 0.58.
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
| mean_lines | 39998.700000 |
| median_lines | 39999.000000 |
| p90_lines | 39999 |
| mean_score | 8334245430 |
| median_score | 8336835700.000000 |
| p90_score | 8342372800 |
| game_over_rate | 0 |
| cap_rate | 1 |
| mean_decision_ms | 6.591729 |
| p95_decision_ms | 10.666852 |
| decisions_per_second | 151.705268 |
| end_to_end_pieces_per_second | 97.148367 |
| lines_per_piece | 0.399987 |
| evaluation_saturated | True |
| metric_censored_by_cap | True |
| mean_root_legal_ms | 3.622714 |
| mean_future_legal_ms | 0.000000 |
| mean_simulation_features_ms | 2.265061 |
| mean_search_overhead_ms | 0.647781 |
| hold_action_rate | 0.349245 |

| Seed | Pieces | Lines | Score | End | Hold actions |
|---:|---:|---:|---:|---|---:|
| 100016 | 100000 | 39998 | 8325082700 | cap | 34890 |
| 100017 | 100000 | 40000 | 8330822900 | cap | 34839 |
| 100018 | 100000 | 39999 | 8323302000 | cap | 34964 |
| 100019 | 100000 | 39999 | 8336406400 | cap | 34910 |
| 100020 | 100000 | 39999 | 8337265000 | cap | 35217 |
| 100021 | 100000 | 39998 | 8342372800 | cap | 35048 |
| 100022 | 100000 | 39999 | 8344165300 | cap | 34699 |
| 100023 | 100000 | 39998 | 8338390400 | cap | 34706 |
| 100024 | 100000 | 39998 | 8325089500 | cap | 35071 |
| 100025 | 100000 | 39999 | 8339557300 | cap | 34901 |
