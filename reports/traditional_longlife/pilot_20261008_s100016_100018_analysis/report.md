# Traditional AI long-life comparison

3 paired validation seeds; cap 10000 Core placements per seed.
All three policies use unchanged Benchmark v1.0 rules and the same seed set. V1 Adapter is not original web V1.
Every capped observation is right-censored. Restricted means use min(lifetime, cap); paired bootstrap 95% intervals resample whole seeds (10,000 fixed-seed draws). These intervals describe capped outcomes, not uncensored lifetime.
When both policies cap on a seed, their relative lifetime is unknown. Official score depends on levels and survival; equal-piece score is replayed at the shorter observed episode length in each pair.

| Mode | Restricted mean pieces [bootstrap 95%] | Median/P90 observed | Death/cap | Mean lines | Lines/piece | Mean score | Decision mean/P95 ms | Total wall h | End-to-end pieces/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 10000.0 [10000.0, 10000.0] | 10000.0/10000.0 | 0.0%/100.0% | 3998.3 | 0.3998 | 88088133.3 | 56.507/98.772 | 0.50 | 16.62 |
| hold | 10000.0 [10000.0, 10000.0] | 10000.0/10000.0 | 0.0%/100.0% | 3999.0 | 0.3999 | 83805000.0 | 5.552/7.008 | 0.07 | 113.80 |
| beam | 10000.0 [10000.0, 10000.0] | 10000.0/10000.0 | 0.0%/100.0% | 3998.7 | 0.3999 | 83292266.7 | 54.010/73.417 | 0.48 | 17.28 |

| Pair (B minus A) | Restricted mean Δ [bootstrap 95%] | Both capped | A death/B cap | A cap/B death | Equal-piece score/piece Δ [bootstrap 95%] |
|---|---:|---:|---:|---:|---:|
| hold_minus_v1 | 0.0 [0.0, 0.0] | 3 | 0 | 0 | -428.313 [-466.950, -369.770] |
| beam_minus_v1 | 0.0 [0.0, 0.0] | 3 | 0 | 0 | -479.587 [-531.370, -389.190] |
| beam_minus_hold | 0.0 [0.0, 0.0] | 3 | 0 | 0 | -51.273 [-83.150, -19.420] |

| Seed | V1 pieces/end | Hold pieces/end | Beam pieces/end | V1 lines/score | Hold lines/score | Beam lines/score |
|---:|---:|---:|---:|---:|---:|---:|
| 100016 | 10000/cap | 10000/cap | 10000/cap | 3998/88607200 | 3999/83937700 | 3998/83425200 |
| 100017 | 10000/cap | 10000/cap | 10000/cap | 3999/88377300 | 3999/83895100 | 3999/83063600 |
| 100018 | 10000/cap | 10000/cap | 10000/cap | 3998/87279900 | 3999/83582200 | 3999/83388000 |

## Official score components on selected seeds

Core replay separates official line-clear points and Hard Drop points. Clear counts are for 0/1/2/3/4 lines per placement. This describes observed trajectories; it does not isolate the causal effect of a single decision.

| Seed | Mode | Line points | Drop points | 1/2/3/4-line clears |
|---:|---|---:|---:|---:|
| 100016 | v1 | 88607200 | 0 | 3206/312/48/6 |
| 100016 | hold | 83937700 | 0 | 3644/154/13/2 |
| 100016 | beam | 83425200 | 0 | 3689/142/7/1 |
| 100017 | v1 | 88377300 | 0 | 3230/310/39/8 |
| 100017 | hold | 83895100 | 0 | 3635/163/10/2 |
| 100017 | beam | 83063600 | 0 | 3720/124/9/1 |
| 100018 | v1 | 87279900 | 0 | 3331/274/33/5 |
| 100018 | hold | 83582200 | 0 | 3676/146/9/1 |
| 100018 | beam | 83388000 | 0 | 3687/141/10/0 |

## Hold-only vs Beam decisions

Scanned up to 10000 placements from each of the first three seeds of both V2 trajectories; evaluated every 100th state and extra states with height ≥12 every 10 placements. Found 105 disagreements; full public observations and paths are in `disagreements.json`.
Immediate heuristic favors Hold-only at a disagreement by construction; Beam trades that score for the best next visible placement. The 200-piece counterfactual forks the same Core state, forces each first action, then uses Hold-only for both continuations. This tests the local choice; it does not prove which full policy lives longer. Neither policy sees hidden state.

| Seed/step/source | Height | Hold action (immediate/two-step) | Beam action (immediate/two-step) | Fork survival Hold/Beam |
|---|---:|---:|---:|---:|
| 100016/9900/hold | 6 | 249 (-38.40/-61.86) | 340 (-39.20/-52.85) | 200/200 |
| 100016/700/hold | 5 | 364 (-22.86/-37.21) | 137 (-22.94/-30.24) | 200/200 |
| 100018/9400/hold | 5 | 549 (-18.51/-28.17) | 986 (-18.51/-27.62) | 200/200 |
| 100016/1700/hold | 4 | 938 (-25.70/-41.21) | 66 (-28.75/-39.09) | 200/200 |
| 100016/2000/hold | 2 | 1078 (-21.47/-35.55) | 112 (-21.87/-25.21) | 200/200 |
| 100016/2200/hold | 4 | 986 (-17.14/-28.22) | 342 (-17.98/-21.75) | 200/200 |
| 100016/2500/hold | 3 | 90 (-18.68/-30.98) | 1170 (-19.16/-30.88) | 200/200 |
| 100016/3000/hold | 4 | 66 (-9.04/-14.90) | 1216 (-9.04/-12.22) | 200/200 |
