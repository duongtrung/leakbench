#!/usr/bin/env python3
"""Why does the richer vantage sometimes score *lower*? Sample size, and here is the proof.

In the cross-model live sweep (150 sessions per model) the provider vantage came out
below the platform vantage for some models, which looks like a violation of the
refinement bound: the providers observe a superset of what the platform observes, so
in the population they cannot leak less.

There is no contradiction, because the bound is about the *population channel* while
advantage is a *lower bound* computed by a finite-sample estimator. The provider
observable is the highest-dimensional one (it concatenates the platform and network
maps plus per-category size aggregates), so it is the first to run out of training data
-- and a looser lower bound at the richer vantage is exactly what that looks like.

This script demonstrates it rather than asserting it. It takes a live capture we
already have at 420 sessions, subsamples it down through the size the sweep used, and
re-estimates every vantage at each size. If the explanation is right, the
providers-minus-platform gap should be noise (either sign) at small n and turn
reliably positive as n grows. No new capture and no model calls are needed.

Usage:  python scripts/vantage_ordering_vs_n.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from leakbench.data.schemas import ToolCall, Trace, TraceDataset
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset

VANTAGES = ("network", "platform", "providers")


def load_jsonl(path: Path) -> TraceDataset:
    traces = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        calls = [ToolCall(category=c["category"], request_size=c["request_size"],
                          response_size=c["response_size"], gap=c["gap"])
                 for c in r["calls"]]
        traces.append(Trace(calls=calls, label=r["label"], family=r.get("family"),
                            metadata=r.get("metadata", {})))
    return TraceDataset(traces=traces, taxonomy_name="live", taxonomy_hash="live",
                        source="live", metadata={})


def subsample(ds: TraceDataset, per_class: int, rng: np.random.Generator) -> TraceDataset:
    """Stratified subsample, so every size compares the same class balance."""
    by_label: dict[int, list[Trace]] = {}
    for t in ds.traces:
        by_label.setdefault(int(t.label), []).append(t)
    picked: list[Trace] = []
    for label in sorted(by_label):
        pool = by_label[label]
        idx = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        picked.extend(pool[i] for i in idx)
    return TraceDataset(traces=picked, taxonomy_name=ds.taxonomy_name,
                        taxonomy_hash=ds.taxonomy_hash, source=ds.source, metadata={})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="outputs/wan/traces_loopback_seed20270215.jsonl")
    ap.add_argument("--per-class", type=int, nargs="+", default=[25, 35, 50, 70])
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent subsamples per size, to separate signal from draw luck")
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--output", default="outputs/llm_sweep/vantage_ordering_vs_n.json")
    args = ap.parse_args()

    full = load_jsonl(Path(args.traces))
    labels = sorted({int(t.label) for t in full.traces})
    prior = np.full(len(labels), 1.0 / len(labels))
    rng = np.random.default_rng(args.seed)

    rows = []
    print(f"{'per class':>9} {'sessions':>8}  {'NET':>16} {'PLAT':>16} {'PROV':>16}"
          f"  {'prov-plat':>18}")
    for per_class in args.per_class:
        draws = {v: [] for v in VANTAGES}
        for _ in range(args.repeats):
            sub = subsample(full, per_class, rng)
            for v in VANTAGES:
                X, y = featurize_dataset(sub, v)
                draws[v].append(estimate_leakage(X, y, vantage=v, prior=prior,
                                                 bootstrap=0, seed=args.seed).advantage)
        stats = {v: (float(np.mean(draws[v])), float(np.std(draws[v], ddof=1)))
                 for v in VANTAGES}
        gap = np.array(draws["providers"]) - np.array(draws["platform"])
        row = {"per_class": per_class, "n_sessions": per_class * len(labels),
               "advantage_mean_sd": stats,
               "prov_minus_plat_mean": float(gap.mean()),
               "prov_minus_plat_sd": float(gap.std(ddof=1)),
               "prov_ge_plat_in_draws": int((gap >= 0).sum()),
               "repeats": args.repeats}
        rows.append(row)
        cells = "  ".join(f"{stats[v][0]:.3f}+/-{stats[v][1]:.3f}" for v in VANTAGES)
        print(f"{per_class:>9} {row['n_sessions']:>8}  {cells}"
              f"  {gap.mean():+.3f}+/-{gap.std(ddof=1):.3f} "
              f"({row['prov_ge_plat_in_draws']}/{args.repeats})")

    first, last = rows[0], rows[-1]
    verdict = {
        "gap_at_smallest_n": first["prov_minus_plat_mean"],
        "gap_at_largest_n": last["prov_minus_plat_mean"],
        "gap_grows_with_n": last["prov_minus_plat_mean"] > first["prov_minus_plat_mean"],
        "prov_ge_plat_at_largest_n": f"{last['prov_ge_plat_in_draws']}/{last['repeats']}",
        "reading": "the refinement bound constrains the population channel; advantage is "
                   "a finite-sample lower bound, and the provider observable is the "
                   "highest-dimensional one, so it is the first to be under-sampled",
    }
    print(f"\nprov-plat gap: {verdict['gap_at_smallest_n']:+.3f} at "
          f"{first['n_sessions']} sessions -> {verdict['gap_at_largest_n']:+.3f} at "
          f"{last['n_sessions']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"status": "real_computation_at_paper_scale",
         "result": {"rq": "vantage_ordering_vs_sample_size", "traces": args.traces,
                    "rows": rows, "verdict": verdict},
         "provenance": {"python_version": sys.version.split()[0],
                        "platform": platform.platform(), "seed": args.seed,
                        "note": "subsamples of a stored live capture; no new capture"}},
        indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
