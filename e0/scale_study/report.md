# LeakBench scale study

- host: `macOS-27.2-arm64-arm-64bit-Mach-O`
- started: 2026-11-08T09:09:52Z  (wall clock 28 s)
- generator: 600 traces/class, measured: 130 traces/class x 5 seeds, live: 70/class
- seed: 20270214, taxonomy: tax-v1/12c-4f (hash `8ba36962dec5`)

Every number below is computed from the calibrated generator, real local tools, or captured loopback traffic at run time. None is a measurement of a deployed product.

## RQ1 - leakage by vantage

| vantage | advantage | MI (bits) |
|---|---|---|
| network | 0.641 | 2.02 |
| platform | 0.875 | 3.03 |
| providers | 0.993 | 3.53 |

## RQ6 - measured harness

| vantage | advantage (mean +/- std) | MI (bits) |
|---|---|---|
| network | 0.181 +/- 0.029 | 0.37 |
| platform | 0.591 +/- 0.035 | 2.10 |
| providers | 0.792 +/- 0.008 | 2.59 |
- network<platform: holds in 5/5 seeds (mean gap 0.409, p=2e-05)
- platform<providers: holds in 5/5 seeds (mean gap 0.201, p=7.5e-05)

## Open-world (out-of-taxonomy other)

| monitored base rate | precision | recall | F1 |
|---|---|---|---|
| 50% | 0.85 | 0.78 | 0.82 |
| 20% | 0.85 | 0.57 | 0.69 |
| 5% | 0.83 | 0.15 | 0.25 |

## RQ8 - live captured traffic (tls_captured_proxy)

| seed | network | platform | providers | size-only | timing-only |
|---|---|---|---|---|---|
| 20270214 | 0.571 | 0.786 | 0.857 | 0.507 | 0.421 |
| 20270215 | 0.457 | 0.807 | 0.786 | 0.436 | 0.250 |
