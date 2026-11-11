#!/usr/bin/env python3
"""Does the WAN run really leak *more* at the operator vantages, and if so, why?

On the first seed the WAN platform advantage came out at 0.93 against 0.79 on
loopback, which is the opposite of what one expects from a noisier channel and so
needs a mechanism rather than a guess.

Hypothesis: on loopback every inter-call gap is sub-millisecond, so the platform's
timing histogram collapses into its lowest bucket and carries nothing; over a real
link each gap is ~250 ms plus a component that grows with the response size, so the
histogram becomes a size proxy for an observer that never sees sizes.

This script captures both conditions at a matched seed -- which also makes it the
second WAN seed the paper reports -- and re-estimates the platform with and without
the timing-histogram block.

**What it found, on seed 20270215.** The mechanism is real but small: the timing
histogram scored *alone* goes from advantage 0.000 on loopback to 0.043 on the WAN.
It is nowhere near large enough to explain the first seed's +0.14, and on this seed
the platform gain is only +0.01. The honest conclusion, which the paper now states,
is that the operator vantages rise on the WAN in 2/2 seeds but the magnitude is not
established -- a single seed would have supported a much stronger claim than the
evidence carries.

Run with the tunnel already up (see run_wan_capture.py's docstring).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakbench.data.live_capture import LiveCaptureHarness
from leakbench.data.loader import dump_trace_dataset
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset, platform_feature_slices
from leakbench.utils.logging import get_logger
from leakbench.utils.provenance import capture_provenance

logger = get_logger("leakbench.wan_vs_loopback")


def score(ds, vantage, prior, seed, drop=None):
    X, y = featurize_dataset(ds, vantage)
    if drop:
        keep = np.ones(X.shape[1], dtype=bool)
        for block in drop:
            keep[platform_feature_slices()[block]] = False
        X = X[:, keep]
    r = estimate_leakage(X, y, vantage=vantage, prior=prior, bootstrap=200, seed=seed)
    return {"advantage": r.advantage, "mi_bits": r.mi_lower_bound_bits}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-per-class", type=int, default=70)
    ap.add_argument("--seed", type=int, default=20270215, help="second seed, so this "
                    "doubles as the WAN repeat the paper reports")
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, required=True)
    ap.add_argument("--server-ca", required=True)
    ap.add_argument("--outdir", default="outputs/wan")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conditions = {}

    for name, kwargs in (
        ("loopback", {}),
        ("wan", {"server_host": args.server_host, "server_port": args.server_port,
                 "server_ca_file": args.server_ca}),
    ):
        logger.info("capturing %s ...", name)
        ds = LiveCaptureHarness(seed=args.seed, **kwargs).capture_dataset(
            args.traces_per_class)
        dump_trace_dataset(ds, outdir / f"traces_{name}_seed{args.seed}.jsonl")

        labels = sorted({t.label for t in ds.traces})
        prior = np.full(len(labels), 1.0 / len(labels))
        gaps = np.concatenate([t.gaps()[1:] for t in ds.traces if t.n > 1])
        conditions[name] = {
            "n_traces": len(ds.traces),
            "median_gap_s": float(np.median(gaps)),
            "vantages": {v: score(ds, v, prior, args.seed)
                         for v in ("network", "platform", "providers")},
            "platform_without_timing_hist": score(ds, "platform", prior, args.seed,
                                                  drop=("timing_hist",)),
            "platform_timing_hist_only": score(
                ds, "platform", prior, args.seed,
                drop=("length", "unigram", "bigram")),
        }
        c = conditions[name]
        logger.info("%s: gap=%.4fs  plat=%.3f  plat-no-timing=%.3f  timing-only=%.3f",
                    name, c["median_gap_s"], c["vantages"]["platform"]["advantage"],
                    c["platform_without_timing_hist"]["advantage"],
                    c["platform_timing_hist_only"]["advantage"])

    lb, wan = conditions["loopback"], conditions["wan"]
    verdict = {
        "platform_gain_from_wan": (wan["vantages"]["platform"]["advantage"]
                                   - lb["vantages"]["platform"]["advantage"]),
        "platform_gain_without_timing_hist": (
            wan["platform_without_timing_hist"]["advantage"]
            - lb["platform_without_timing_hist"]["advantage"]),
        "gap_ratio_wan_over_loopback": wan["median_gap_s"] / max(lb["median_gap_s"], 1e-9),
    }
    verdict["timing_hist_explains_the_gain"] = bool(
        verdict["platform_gain_without_timing_hist"]
        < 0.5 * verdict["platform_gain_from_wan"])

    payload = {"status": "live_captured_traffic_over_wan",
               "result": {"rq": "RQ8_wan_vs_loopback", "seed": args.seed,
                          "traces_per_class": args.traces_per_class,
                          "conditions": conditions, "verdict": verdict},
               "provenance": capture_provenance(
                   seed=args.seed, taxonomy_hash="live", taxonomy_name="live/6c",
                   config={"traces_per_class": args.traces_per_class,
                           "server_host": args.server_host,
                           "server_port": args.server_port},
                   note="paired loopback/WAN capture at one seed").to_dict()}
    (outdir / f"wan_vs_loopback_seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
