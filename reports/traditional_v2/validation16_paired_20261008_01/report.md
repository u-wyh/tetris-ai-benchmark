# V1 Adapter / V2 paired comparison

Same 16 seeds and 5000 piece cap. Truncation is not death.
Differences are V2 minus V1. Bootstrap resamples seed pairs 10,000 times with fixed analysis seed; intervals describe the cap-limited sample mean, not uncensored lifetime.
The sign test uses observed unequal survival counts; jointly capped pairs remain unresolved.
Official score includes level multipliers and is not a substitute for survival.

| Metric | Result |
|---|---:|
| mean_pieces_difference | 955.3125 |
| median_pieces_difference | 0.0 |
| mean_pieces_difference_bootstrap_95ci | [324.1875, 1680.9375] |
| mean_lines_difference | 386 |
| median_lines_difference | 2.0 |
| mean_lines_difference_bootstrap_95ci | [132.1875, 677.625] |
| mean_score_difference | 4662637.5 |
| median_score_difference | -1171950.0 |
| mean_score_difference_bootstrap_95ci | [812150.0, 9062943.75] |
| v1_cap_rate | 0.625 |
| v2_cap_rate | 1 |
| both_cap_count | 10 |
| v1_dead_v2_cap_count | 6 |
| v1_cap_v2_dead_count | 0 |
| observed_wins | 6 |
| observed_losses | 0 |
| equal_or_both_capped | 10 |
| sign_test_two_sided_p | 0.03125 |

| Seed | V1 pieces | V2 pieces | Δ pieces | V1 lines | V2 lines | Δ lines | V1 score | V2 score | Δ score | End V1/V2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100000 | 5000 | 5000 | 0 | 1997 | 1999 | 2 | 21885000 | 20816900 | -1068100 | cap/cap |
| 100001 | 2952 | 5000 | 2048 | 1168 | 1999 | 831 | 7885100 | 20832600 | 12947500 | over/cap |
| 100002 | 1648 | 5000 | 3352 | 649 | 1999 | 1350 | 2358700 | 20942000 | 18583300 | over/cap |
| 100003 | 5000 | 5000 | 0 | 1996 | 1999 | 3 | 22464100 | 20915400 | -1548700 | cap/cap |
| 100004 | 5000 | 5000 | 0 | 1997 | 1999 | 2 | 22463900 | 20919400 | -1544500 | cap/cap |
| 100005 | 5000 | 5000 | 0 | 1997 | 1999 | 2 | 22187800 | 20914700 | -1273100 | cap/cap |
| 100006 | 5000 | 5000 | 0 | 1999 | 1999 | 0 | 22291000 | 20773800 | -1517200 | cap/cap |
| 100007 | 5000 | 5000 | 0 | 1998 | 1999 | 1 | 22318600 | 20842300 | -1476300 | cap/cap |
| 100008 | 5000 | 5000 | 0 | 1997 | 1999 | 2 | 22121700 | 20944400 | -1177300 | cap/cap |
| 100009 | 2407 | 5000 | 2593 | 953 | 1999 | 1046 | 5045700 | 20903600 | 15857900 | over/cap |
| 100010 | 1312 | 5000 | 3688 | 517 | 1999 | 1482 | 1478500 | 20958500 | 19480000 | over/cap |
| 100011 | 5000 | 5000 | 0 | 1999 | 1999 | 0 | 22223100 | 21001100 | -1222000 | cap/cap |
| 100012 | 5000 | 5000 | 0 | 1999 | 1999 | 0 | 22032800 | 20788700 | -1244100 | cap/cap |
| 100013 | 5000 | 5000 | 0 | 2000 | 1999 | -1 | 22068000 | 20901400 | -1166600 | cap/cap |
| 100014 | 1942 | 5000 | 3058 | 768 | 1999 | 1231 | 3230100 | 20786000 | 17555900 | over/cap |
| 100015 | 4454 | 5000 | 546 | 1773 | 1998 | 225 | 17407200 | 20822700 | 3415500 | over/cap |
