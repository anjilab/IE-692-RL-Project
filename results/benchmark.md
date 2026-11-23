# Benchmark (2026-11-23 14:17, limit 600 s)

## Kernel throughput (one core, compilation excluded)

| problem | states | TOMC steps/s | True Online TD steps/s | exact scoring of one policy | certificate of one policy |
|---|---|---|---|---|---|
| GridWorld | 100 | 29.8 M | 10.5 M | 0.3 ms | 0.19 s |
| GridWorld30 | 900 | 30.5 M | 10.9 M | 2.5 ms | 0.86 s |
| Blackjack | 32 | 4.4 M | 12.4 M | 0.0 ms | 0.00 s |
| Inventory | 90 | 21.5 M | 6.7 M | 1.7 ms | 0.72 s |

## Projected pipeline time

7 worker processes; tuning on 10% of each budget with 3 seeds, then 20 test seeds on the full budget.

| problem | method | combinations | environment steps | projected time |
|---|---|---|---|---|
| GridWorld | baseline | 9 | 4.54e+08 | 3.5 s |
| GridWorld | entropy_reg | 27 | 5.62e+08 | 5.5 s |
| GridWorld | entropy_reg_annealed | 27 | 5.62e+08 | 5.5 s |
| GridWorld | true_online_td | 27 | 5.62e+08 | 10.9 s |
| GridWorld | bregman_euclidean | 9 | 4.54e+08 | 3.5 s |
| GridWorld | bregman_tsallis | 9 | 4.54e+08 | 3.5 s |
| Blackjack | baseline | 9 | 2.72e+06 | 0.1 s |
| Blackjack | entropy_reg | 27 | 3.37e+06 | 0.1 s |
| Blackjack | entropy_reg_annealed | 27 | 3.37e+06 | 0.1 s |
| Blackjack | true_online_td | 27 | 3.37e+06 | 0.1 s |
| Blackjack | bregman_euclidean | 9 | 2.72e+06 | 0.1 s |
| Blackjack | bregman_tsallis | 9 | 2.72e+06 | 0.1 s |

**Projected time: 33 s** (plus about 1-3 s of compilation per worker). TOMC rows assume every estimate is as long as one under the uniform policy; an infinite projection means a single estimate does not finish. CPUs that mix performance and efficiency cores give less than a linear speed-up, so use --run for the measured time.

## Measured pipeline

`tune.py` end to end: **47.8 s** wall time, peak memory 191 MB per process (within the limit).

## Memory

| structure | before | now |
|---|---|---|
| TOMC stream (stored trajectory vs ring buffer) | 105 B per step: 10.5 GB for 10^8 steps | O(S A + 350) values: 90 kB on 30x30, any length |
| exact model, 30x30 (dense P vs sparse) | 26 MB | 0.32 MB |
| TD tabular features, 30x30 (dense Phi vs index table) | 104 MB | 29 kB |
| TD history (Q snapshot per record vs V(s0)) | 29 kB per record | 16 B per record |

## GridWorld 30x30 feasibility (TOMC, uniform policy)

| step budget | required pairs covered | time |
|---|---|---|
| 1e+06 | 1464/3600 pairs | 0.0 s |
| 1e+07 | 2081/3600 pairs | 0.3 s |
| 1e+08 | 2754/3600 pairs | 3.3 s |
| 1e+09 | 3139/3600 pairs | 33.9 s |

Hazards cover 10% of the 30x30 grid and send the agent back to the start with probability 0.8, so a uniformly random walk rarely reaches the far side: coverage grows by only a few hundred pairs per tenfold increase in steps. Extrapolating that trend, a single TOMC estimate would not finish within the ~1.8e10 steps one core samples in 10 minutes. This is a property of the problem, not of the step rate.
