#!/usr/bin/env python3
"""Collect the per-model live runs into one result, and say what generalizes.

The paper's most striking live finding --- a real LLM planner erases the timing
channel while the size channel *strengthens* --- rested on a single model, which makes
it an anecdote rather than a finding. This aggregates one live capture per model,
across model families and vendors, and reports which conclusions survive.

The three claims under test, stated so they can fail:

1. **Vantage ordering**, tested as two separate steps because they have different
   status. NET < PLAT is an *empirical* claim (the vantages are incomparable, so
   nothing forces it). PLAT <= PROV is *forced* by refinement, so a point-estimate
   inversion there is either a bug or an under-powered estimate -- and we only count
   it as a violation when the confidence intervals are disjoint. At the 150 sessions
   these runs use, they are not: `scripts/vantage_ordering_vs_n.py` shows the
   providers-minus-platform gap is -0.020 at 150 sessions and turns positive by 210,
   because the provider observable is the highest-dimensional and the first to be
   under-sampled.
2. **Size dominates timing.** The size-only projection should beat the timing-only one
   for every model. This is the paper's central mechanism claim.
3. **Timing stays the weak channel under every model.** Timing-only advantage should
   be low across the board while size-only stays high.

A fourth claim -- that timing collapses *because* model latency adds a large,
task-independent term to every gap -- needs the inter-call gap distribution to test,
and the runs already completed did not record it. `run_live` now does, so a rerun
tests it; until then this script reports the gap where it is available and does not
assert the mechanism from token counts, which are not latency.

Usage:  python scripts/aggregate_llm_sweep.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

VANTAGES = ("network", "platform", "providers")


def load(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    r = d.get("result", d)
    if not r.get("per_seed"):
        return None
    e = r["per_seed"][0]
    llm = r.get("llm", {})
    calls = llm.get("calls") or 0
    return {
        "model": llm.get("model", path.parent.name),
        "provider": llm.get("provider", "?"),
        "advantage": {v: e["vantages"][v]["advantage"] for v in VANTAGES},
        "ci": {v: list(e["vantages"][v]["advantage_ci"]) for v in VANTAGES},
        "mi_bits": {v: e["vantages"][v]["mi_bits"] for v in VANTAGES},
        "size_only": e["projections"]["size_only"],
        "timing_only": e["projections"]["timing_only"],
        "n_traces": e["n_traces"],
        "mean_trace_length": e["mean_trace_length"],
        "llm_calls": calls,
        "prompt_tokens": llm.get("prompt_tokens"),
        "completion_tokens": llm.get("completion_tokens"),
        "mean_completion_tokens_per_call": (
            round(llm["completion_tokens"] / calls, 1)
            if calls and llm.get("completion_tokens") else None),
        "inter_call_gap_s": e.get("inter_call_gap_s"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default="outputs/llm_sweep")
    ap.add_argument("--baseline", default="outputs/live_neural/live.json",
                    help="the first neural run, included as one more model")
    ap.add_argument("--output", default="outputs/llm_sweep/summary.json")
    args = ap.parse_args()

    rows = []
    base = load(Path(args.baseline))
    if base:
        rows.append(base)
    for d in sorted(Path(args.sweep_dir).glob("*/live.json")):
        r = load(d)
        if r and r["model"] not in {x["model"] for x in rows}:
            rows.append(r)
    if not rows:
        raise SystemExit(f"no completed runs under {args.sweep_dir}")

    net_below_plat = [r for r in rows
                      if r["advantage"]["network"] < r["advantage"]["platform"]]
    prov_ge_plat = [r for r in rows
                    if r["advantage"]["providers"] >= r["advantage"]["platform"] - 1e-9]
    # A refinement "violation" only counts when the intervals do not overlap; a point
    # inversion inside overlapping intervals is an under-powered estimate, not a
    # counterexample to a statement about the population channel.
    significant_inversions = [
        r["model"] for r in rows
        if r["ci"]["providers"][1] < r["ci"]["platform"][0]
    ]
    size_beats_timing = [r for r in rows if r["size_only"] > r["timing_only"]]

    def arr(key, sub=None):
        return np.array([(r[key][sub] if sub else r[key]) for r in rows], dtype=float)

    summary = {
        "rq": "RQ8_cross_model",
        "n_models": len(rows),
        "models": [r["model"] for r in rows],
        "net_below_platform": f"{len(net_below_plat)}/{len(rows)}",
        "providers_ge_platform_pointwise": f"{len(prov_ge_plat)}/{len(rows)}",
        "significant_refinement_violations": significant_inversions,
        "size_beats_timing": f"{len(size_beats_timing)}/{len(rows)}",
        "advantage_range": {v: [float(arr('advantage', v).min()),
                                float(arr('advantage', v).max())] for v in VANTAGES},
        "advantage_mean_sd": {v: [float(arr('advantage', v).mean()),
                                  float(arr('advantage', v).std(ddof=1))
                                  if len(rows) > 1 else 0.0] for v in VANTAGES},
        "size_only_range": [float(arr("size_only").min()), float(arr("size_only").max())],
        "timing_only_range": [float(arr("timing_only").min()),
                              float(arr("timing_only").max())],
        "rows": rows,
        "note": "one seed per model, all at seed 20270214 so the task draws match. The "
                "claim under test is that size dominance is a property of LLM-driven "
                "agents rather than of one model. At 150 sessions the platform/provider "
                "step is under-powered by design (see vantage_ordering_vs_n.json); the "
                "well-powered evidence for it is the 420-session programmatic and WAN "
                "runs, so here we require only that no inversion be significant.",
    }

    print(f"{'model':<42} {'NET':>6} {'PLAT':>6} {'PROV':>6} {'size':>6} {'time':>6} {'tok/call':>9}")
    for r in rows:
        a = r["advantage"]
        print(f"{r['model']:<42} {a['network']:6.3f} {a['platform']:6.3f} "
              f"{a['providers']:6.3f} {r['size_only']:6.3f} {r['timing_only']:6.3f} "
              f"{str(r['mean_completion_tokens_per_call'] or '?'):>9}")
    print(f"\nnet < platform             : {summary['net_below_platform']}")
    print(f"providers >= platform      : {summary['providers_ge_platform_pointwise']} "
          f"pointwise; significant violations (disjoint CIs): "
          f"{significant_inversions or 'none'}")
    print(f"size beats timing          : {summary['size_beats_timing']}")
    print(f"timing-only range      : {summary['timing_only_range'][0]:.3f}"
          f"--{summary['timing_only_range'][1]:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"status": "live_captured_traffic", "result": summary,
         "provenance": {"python_version": sys.version.split()[0],
                        "platform": platform.platform(), "seed": 20270214,
                        "note": "aggregation of per-model live captures"}}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
