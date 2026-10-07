# SubprocVecEnv MaskablePPO benchmark — 2026-10-07

AMD Ryzen 7 6800H: 8 physical cores / 16 logical threads (`nproc=16`). PyTorch 2.10.0+cu130; RTX 3050 Laptop GPU. All eight runs used `SubprocVecEnv(start_method="forkserver")`, worker seeds `42 + rank`, `TetrisEnv.action_masks()`, 237 observations, 1840 actions, and the existing 256/256 Tanh `MaskablePPO` parameters. Each run performed four 4096-sample rollout/update cycles (16384 steps), with `batch_size=256`. Startup was outside the training timer; no separate warm-up was run. Order: CPU 1→2→4→8, then CUDA 1→2→4→8. Raw results: `runs/vector_env_benchmark/*/results.json` (Git-ignored).

| Device | Envs | n_steps | Time (s) | Steps/s | Speedup | Efficiency | Rollout (s) | PPO update (s) | Process CPU cores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 1 | 4096 | 128.73 | 127.28 | 1.00× | 100% | 117.77 | 10.95 | 0.99 |
| CPU | 2 | 2048 | 83.10 | 197.16 | 1.55× | 77% | 72.13 | 10.97 | 1.52 |
| CPU | 4 | 1024 | 56.02 | 292.46 | 2.30× | 57% | 45.09 | 10.93 | 2.39 |
| CPU | 8 | 512 | 39.07 | 419.37 | 3.29× | 41% | 28.12 | 10.95 | 3.57 |
| CUDA | 1 | 4096 | 127.39 | 128.62 | 1.00× | 100% | 125.73 | 1.65 | 1.00 |
| CUDA | 2 | 2048 | 78.02 | 210.01 | 1.63× | 82% | 76.53 | 1.49 | 1.58 |
| CUDA | 4 | 1024 | 47.71 | 343.38 | 2.67× | 67% | 46.23 | 1.49 | 2.64 |
| CUDA | 8 | 512 | 28.90 | 566.83 | 4.41× | 55% | 27.40 | 1.50 | 4.53 |

Speedup and parallel efficiency use each device's one-worker result. Total system CPU utilization averaged 6.4/9.8/15.2/22.6% across CPU runs and 6.9/10.6/17.2/29.0% across CUDA runs (16 logical threads). The process-core column measures the trainer plus workers. CPU throughput continues improving at eight workers, though its per-worker efficiency declines from four workers onward.

Rollout collection throughput (1/2/4/8 envs) was 139.11/227.14/363.36/582.63 steps/s on CPU and 130.31/214.10/354.43/597.92 steps/s on CUDA.

| CUDA envs | GPU util mean/peak | Temperature mean/peak | VRAM peak | GPU power mean/peak |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.8% / 43% | 48.8 / 50 °C | 191 MiB | 10.9 / 16.4 W |
| 2 | 4.9% / 71% | 50.5 / 52 °C | 189 MiB | 11.4 / 17.0 W |
| 4 | 4.9% / 63% | 52.6 / 55 °C | 191 MiB | 11.9 / 20.5 W |
| 8 | 5.4% / 74% | 55.5 / 58 °C | 191 MiB | 12.7 / 17.5 W |

GPU and CPU resource values were sampled approximately once per second; brief GPU spikes may fall between samples.

Worker-side environment and mask work in the eight-worker CUDA run totaled 121.9 CPU-seconds across workers; rollout wall time was 27.4 seconds. A separate 64-query mask microprofile measured about 3.65 ms worker computation plus 0.40 ms IPC/synchronization per query with eight workers (CPU: 3.41 + 0.21 ms). IPC grows but is not the dominant cost; Python environment/mask computation and rollout policy/synchronization remain the bottleneck. CUDA reduces PPO update time from about 11 to 1.5 seconds per run, lifting eight-worker throughput 35% above CPU eight-worker throughput. The GPU stays mostly idle between short inference/update bursts.

**Fastest configuration:** CUDA, 8 envs, `n_steps=512`, 566.83 steps/s.

**Recommended training configuration:** CUDA, 8 envs, `n_steps=512`, based on the largest measured throughput and moderate temperature/VRAM use. The existing transaction trainer remains single-environment and CPU by default; using this configuration for resumable formal training requires a later multi-worker checkpoint implementation. No formal training was started here.

**Primary bottleneck after vectorization:** rollout-side Tetris environment and Action Mask computation, followed by policy/synchronization overhead; mask IPC is secondary.

These are single-seed, single-pass measurements. CPU/CUDA action trajectories can diverge despite identical seed and settings, so small differences should not be overinterpreted.
