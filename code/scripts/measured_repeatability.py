#!/usr/bin/env python3
"""How reproducible is the measured harness at a *fixed* seed?

Everything else in this benchmark is bit-reproducible from its seed. The measured
harness is not, and the reason is the point of it: it records **real wall-clock**
inter-call timing from **real tool execution**, so the gaps differ between processes
even when the seed, the taxonomy, and the tool programs are identical. Payload sizes
are deterministic (the tools return what they return); the timing channel is not.

That shows up as small disagreements between stages that each build their own measured
dataset -- e.g. the RQ6 multi-seed stage and the case-study projection stage report
0.152 and 0.145 for the same quantity. This script measures the effect directly so the
paper can state it rather than leave a reader to discover it by re-running.

Usage:  python scripts/measured_repeatability.py [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from leakbench.data.measured import MeasuredHarness
from leakbench.data.wire import HTTP2WireModel, apply_wire_model_dataset
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset
from leakbench.taxonomy import reference_taxonomy

VANTAGES = ("network", "platform", "providers")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--traces-per-class", type=int, default=130)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--output", default="outputs/scale_study/measured_repeatability.json")
    args = ap.parse_args()

    tax = reference_taxonomy()
    started = time.time()
    runs = []
    for i in range(args.repeats):
        ds = MeasuredHarness(taxonomy=tax, seed=args.seed,
                             overlap=args.overlap).sample_dataset(args.traces_per_class)
        wired = apply_wire_model_dataset(ds, HTTP2WireModel(seed=args.seed))
        row = {}
        for v in VANTAGES:
            X, y = featurize_dataset(wired if v == "network" else ds, v)
            row[v] = estimate_leakage(X, y, vantage=v, prior=tax.prior, bootstrap=50,
                                      seed=args.seed).advantage
        # Separate the deterministic part (sizes) from the measured part (timing).
        row["_total_bytes"] = float(sum(t.total_sizes().sum() for t in ds.traces))
        gaps = np.concatenate([t.gaps()[1:] for t in ds.traces if t.n > 1])
        row["_median_gap_s"] = float(np.median(gaps))
        runs.append(row)
        print(f"repeat {i+1}: " + "  ".join(f"{v}={row[v]:.4f}" for v in VANTAGES)
              + f"   bytes={row['_total_bytes']:.0f}  median gap={row['_median_gap_s']*1e6:.0f} us",
              flush=True)

    summary = {}
    for v in VANTAGES:
        a = np.array([r[v] for r in runs])
        summary[v] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                      "min": float(a.min()), "max": float(a.max()),
                      "range": float(a.max() - a.min())}
    sizes_identical = len({r["_total_bytes"] for r in runs}) == 1

    payload = {
        "status": "real_computation_at_paper_scale",
        "result": {
            "rq": "measured_harness_repeatability",
            "seed": args.seed,
            "repeats": args.repeats,
            "traces_per_class": args.traces_per_class,
            "overlap": args.overlap,
            "runs": runs,
            "summary": summary,
            "payload_bytes_identical_across_repeats": sizes_identical,
            "note": "same seed, separate processes. Sizes are deterministic because the "
                    "tools return what they return; inter-call gaps are real wall-clock "
                    "and are not, so the network vantage -- the one that depends on "
                    "timing -- carries a repeat spread that the bootstrap CI does not "
                    "cover and the cross-seed std does not separate out",
        },
        "provenance": {"python_version": sys.version.split()[0],
                       "platform": platform.platform(),
                       "elapsed_s": time.time() - started},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print("\npayload bytes identical across repeats:", sizes_identical)
    for v in VANTAGES:
        s = summary[v]
        print(f"  {v:<10} {s['mean']:.4f} +/- {s['std']:.4f}   range {s['range']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
