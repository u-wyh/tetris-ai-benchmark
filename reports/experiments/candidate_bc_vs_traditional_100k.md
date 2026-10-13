# 20M PPO + BC versus traditional V1 and V2 Hold at 100,000 pieces

Completed 2026-10-13 13:38 Asia/Shanghai. All three policies were evaluated on seeds 100016–100025 with the same `TetrisCore` and a 100,000-piece cap. These seeds are part of the existing validation set, so this is a long-horizon validation comparison, not a fresh final test.

| Policy | Mean pieces survived | Median | Minimum | Reached 100,000 cap | Mean lines |
|---|---:|---:|---:|---:|---:|
| Traditional V1 adapter | 26,002.6 | 15,971.5 | 3,681 | 1/10 | 10,392.1 |
| Traditional V2 Hold-only | 100,000 | 100,000 | 100,000 | 10/10 | 39,998.7 |
| Candidate 20M PPO + BC epoch 3 | 100,000 | 100,000 | 100,000 | 10/10 | 39,998.6 |

BC matched V2 Hold at the test cap and greatly outperformed V1 on these seeds. Since both BC and V2 were capped in every game, the data cannot rank their survival beyond 100,000 pieces. All 10 BC runs were evaluated with deterministic legal-action masking; the frozen model SHA256 was `5f5613cb5eb4e80a02406b70902ad3d5c571eb8fe38bec4b40f73b34289ca663`.

Machine-readable per-seed results and model identity are in `runs/imitation_v2_hold_20m/bc_vs_traditional_100k/`; traditional reference rows are in `reports/traditional_longlife/validation10_100000_20261008/`.
