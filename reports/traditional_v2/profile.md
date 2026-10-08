# Matched-state V1/V2 CPU decision profile

13 public states × 3 repeats, same root BFS candidates; V2 beam width 8.
V1 Python wrapper profiling adds small overhead; both policies ran sequentially under the same host load.
Root legal BFS is counted once for both. V1 future BFS is inside next_legal_ms; V2 future BFS is future_legal. Timings exclude Core.step and file I/O.

| Metric | Mean ms |
|---|---:|
| mean_root_legal_ms | 4.288 |
| mean_v1_policy_ms | 61.872 |
| mean_v2_policy_ms | 61.175 |
| mean_v1_next_legal_ms | 50.147 |
| mean_v1_simulation_ms | 1.973 |
| mean_v1_features_ms | 8.820 |
| mean_v2_root_legal | 0.000 |
| mean_v2_simulation_features | 25.118 |
| mean_v2_future_legal | 34.170 |
| mean_v2_search_overhead | 1.228 |
| mean_v2_policy_total | 60.516 |
| mean_v1_total_ms | 66.160 |
| mean_v2_total_ms | 65.463 |
