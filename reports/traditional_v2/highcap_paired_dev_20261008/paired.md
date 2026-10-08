# V1 Adapter / V2 paired comparison

Same 2 seeds and 10000 piece cap. Truncation is not death.
Differences are V2 minus V1. Bootstrap resamples seed pairs 10,000 times with fixed analysis seed; intervals describe the cap-limited sample mean, not uncensored lifetime.
The sign test uses observed unequal survival counts; jointly capped pairs remain unresolved.
Official score includes level multipliers and is not a substitute for survival.

| Metric | Result |
|---|---:|
| mean_pieces_difference | 3806 |
| median_pieces_difference | 3806.0 |
| mean_pieces_difference_bootstrap_95ci | [0.0, 7612.0] |
| mean_lines_difference | 1527 |
| median_lines_difference | 1527.0 |
| mean_lines_difference_bootstrap_95ci | [0.0, 3054.0] |
| mean_score_difference | 36231450 |
| median_score_difference | 36231450.0 |
| mean_score_difference_bootstrap_95ci | [-5595500.0, 78058400.0] |
| v1_cap_rate | 0.5 |
| v2_cap_rate | 1 |
| both_cap_count | 1 |
| v1_dead_v2_cap_count | 1 |
| v1_cap_v2_dead_count | 0 |
| observed_wins | 1 |
| observed_losses | 0 |
| equal_or_both_capped | 1 |
| sign_test_two_sided_p | 1.0 |

| Seed | V1 pieces | V2 pieces | Δ pieces | V1 lines | V2 lines | Δ lines | V1 score | V2 score | Δ score | End V1/V2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 310100 | 10000 | 10000 | 0 | 3999 | 3999 | 0 | 88672400 | 83076900 | -5595500 | cap/cap |
| 310103 | 2388 | 10000 | 7612 | 945 | 3999 | 3054 | 4947000 | 83005400 | 78058400 | over/cap |
