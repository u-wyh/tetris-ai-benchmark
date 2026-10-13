# Candidate imitation learning: V2 Hold → BC → PPO

Generated: 2026-10-13T14:56:53+08:00. Teacher: V2 Hold-only. Expert samples: 214809.
Teacher config SHA256: 744b2336f746642c99e8998b67047408f1b88914c1c61737531bbc9cc356d8cf. Shards: 108.
Hold labels: 75051; danger height ≥12: 83.
Train/BC validation split by complete episode seed. Official final-test seeds excluded.
BC validation Top-1: 0.420 → 0.951.
BC validation teacher probability: 0.390 → 0.930.

| Model | Mean pieces | P10 | Cap/32 | Mean lines |
|---|---:|---:|---:|---:|
| pure_20m | 14986.7 | 1713 | 19 | 5986.3 |
| pure_30m | 19626.4 | 20000 | 31 | 7846.7 |
| bc | 20000.0 | 20000 | 32 | 7998.8 |
| bc_ppo_2m | 15368.5 | 8990 | 16 | 6138.9 |
| traditional_v2_hold | 20000.0 | 20000 | 32 | 7998.7 |

Automatic 10M extension: False.
All gameplay comparisons, including V2 Hold, use the same 32 validation seeds and 20,000-piece cap.
50M/100M extension is not authorized by this experiment.
