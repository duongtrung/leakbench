# LeakBench scale study

- host: `Linux-7.0.0-1006-aws-x86_64-with-glibc2.43`
- started: 2026-08-08T01:42:27Z  (wall clock 243 s)
- generator: 600 traces/class, measured: 130 traces/class x 5 seeds, live: 70/class
- seed: 20270214, taxonomy: tax-v1/12c-4f (hash `8ba36962dec5`)

Every number below is computed from the calibrated generator, real local tools, or captured loopback traffic at run time. None is a measurement of a deployed product.

## RQ1 - leakage by vantage

| vantage | advantage | MI (bits) |
|---|---|---|
| network | 0.637 | 2.01 |
| platform | 0.877 | 3.03 |
| providers | 0.993 | 3.53 |

## RQ2 - which features leak

- length_only: 0.085
- size_only: 0.440
- size_plus_timing_all: 0.637
- timing_only: 0.345

## RQ4 - defense frontier (network vantage)

| defense | advantage | bandwidth | latency |
|---|---|---|---|
| none | 0.637 | +0% | +0% |
| pad-512 | 0.625 | +21% | +0% |
| shuffle | 0.637 | +0% | +0% |
| constant-rate | 0.615 | +0% | +0% |
| cover | 0.598 | +90% | +90% |
| pad-max | 0.333 | +4163% | +0% |
| full | 0.220 | +7996% | +98% |
| gen-shaping | 0.114 | +46% | +67% |
| dp-shaping | 0.005 | +607% | +115% |

## RQ6 - measured harness

| vantage | advantage (mean +/- std) | MI (bits) |
|---|---|---|
| network | 0.187 +/- 0.026 | 0.49 |
| platform | 0.587 +/- 0.039 | 2.10 |
| providers | 0.794 +/- 0.019 | 2.65 |
- network<platform: holds in 5/5 seeds (mean gap 0.399, p=5e-05)
- platform<providers: holds in 5/5 seeds (mean gap 0.208, p=0.00041)

## Open-world (out-of-taxonomy other)

| monitored base rate | precision | recall | F1 |
|---|---|---|---|
| 50% | 0.85 | 0.79 | 0.82 |
| 20% | 0.85 | 0.58 | 0.69 |
| 5% | 0.83 | 0.14 | 0.24 |

## RQ7 - taxonomy size

| \|T\| | H(C) | advantage | MI (bits) |
|---|---|---|---|
| 8 | 3.00 | 0.937 | 2.64 |
| 12 | 3.58 | 0.904 | 3.04 |
| 16 | 4.00 | 0.853 | 3.21 |
| 24 | 4.58 | 0.779 | 3.41 |

## RQ8 - live captured traffic (tls_captured_proxy)

| seed | network | platform | providers | size-only | timing-only |
|---|---|---|---|---|---|
| 20270214 | 0.614 | 0.786 | 0.793 | 0.521 | 0.457 |
| 20270215 | 0.493 | 0.807 | 0.750 | 0.457 | 0.350 |

## Granularity - coarse (4-way family) vs fine (12-way class)

| vantage | class Adv | class MI | family Adv | family MI |
|---|---|---|---|---|
| network | 0.637 | 2.01 | 0.625 | 0.76 |
| platform | 0.877 | 3.03 | 0.984 | 1.89 |
| providers | 0.993 | 3.53 | 0.990 | 1.93 |

## RQ2 - feature importance (corroborates the ablation)

- size_stats: 29.1%
- timing_stats: 27.0%
- size_hist: 17.1%
- volume: 14.8%
- timing_hist: 12.0%

## Separability sweep (knob: dirichlet_alpha)

| dirichlet_alpha | advantage | MI (bits) |
|---|---|---|
| 5 | 0.798 | 2.78 |
| 1.5 | 0.839 | 2.78 |
| 0.6 | 0.877 | 3.03 |
| 0.3 | 0.892 | 3.05 |
| 0.1 | 0.956 | 3.35 |

advantage monotone along the separability knob: True

Companion curve --- the *family-structure* knob (signature up-weight), which decreases advantage because the signature is shared by three classes:

| signature up-weight | advantage | MI (bits) |
|---|---|---|
| 0.3 | 0.882 | 2.94 |
| 0.6 | 0.889 | 3.04 |
| 1 | 0.875 | 3.04 |
| 1.5 | 0.862 | 3.03 |
| 2.5 | 0.850 | 3.04 |

## Variance across generator draws

| vantage | advantage (mean +/- std) |
|---|---|
| network | 0.661 +/- 0.045 |
| platform | 0.875 +/- 0.019 |
| providers | 0.971 +/- 0.008 |

## Prior sensitivity (uniform vs Zipf)

| prior | network | platform | providers |
|---|---|---|---|
| uniform | 0.637 | 0.877 | 0.993 |
| zipf | 0.510 | 0.833 | 0.990 |

## DP shaping: epsilon -> advantage

| epsilon | adjacency | advantage | ceiling (d=1) | bandwidth |
|---|---|---|---|---|
| 0.5 | all_pairs | 0.003 | 0.059 | +1369% |
| 1 | all_pairs | 0.000 | 0.156 | +1364% |
| 2 | all_pairs | 0.000 | 0.581 | +1355% |
| 4 | all_pairs | 0.006 | 1.000 | +1335% |
| 8 | all_pairs | 0.004 | 1.000 | +1298% |
| 16 | all_pairs | 0.003 | 1.000 | +1227% |
| 32 | all_pairs | 0.010 | 1.000 | +1098% |
| 64 | all_pairs | 0.017 | 1.000 | +892% |
| 128 | all_pairs | 0.035 | 1.000 | +621% |
| 0.5 | per_window | 0.003 | 0.059 | +1227% |
| 1 | per_window | 0.010 | 0.156 | +1098% |
| 2 | per_window | 0.017 | 0.581 | +892% |
| 4 | per_window | 0.035 | 1.000 | +621% |
| 8 | per_window | 0.085 | 1.000 | +359% |
| 16 | per_window | 0.148 | 1.000 | +185% |
| 32 | per_window | 0.202 | 1.000 | +92% |
| 64 | per_window | 0.254 | 1.000 | +44% |
| 128 | per_window | 0.301 | 1.000 | +19% |

Ceiling violations: 0
