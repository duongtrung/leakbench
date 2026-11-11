"""E1: what does ONE tool provider learn, instead of all providers colluding?

A single provider of tool category c sees only the calls addressed to it. We assume it
can link the calls of one session (same API key or connection), and that it sees
metadata only, like every other vantage in the benchmark. Its view of a session is the
sub-trace of calls with category c:
  - request and response size of each of its calls, and how many there were;
  - the time between its own consecutive calls (the sum of the gaps in between).
It does not see the other categories, their sizes, or when the session started.
We featurize that sub-trace with the network map O_net (sizes, count, timing), which
is exactly the information a provider has about its own traffic.

Two estimates per provider:
  - "all": every session; a session that never called c is an all-zero vector. This
    gives the provider knowledge that the session exists, so it is an upper bound.
  - "called": only sessions that called c (what the provider actually receives).
    Advantage uses the empirical prior of those sessions, so it is measured against
    guessing the most common class among them.

Usage: python e1/single_provider.py OUT.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from leakbench.data.loader import iter_trace_file
from leakbench.data.schemas import Trace, TraceDataset
from leakbench.estimators.leakage import estimate_leakage
from leakbench.experiments.config import load_config
from leakbench.experiments.runners import build_dataset
from leakbench.observables.maps import NETWORK_DIM, featurize_dataset, network_map
from leakbench.taxonomy import TOOL_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RESULTS = ROOT / "results"


def provider_subtrace(t: Trace, cat: str) -> Trace:
    times = np.cumsum([c.gap for c in t.calls])
    keep = [i for i, c in enumerate(t.calls) if c.category == cat]
    calls = []
    for j, i in enumerate(keep):
        gap = 0.0 if j == 0 else float(times[i] - times[keep[j - 1]])
        calls.append(replace(t.calls[i], gap=max(gap, 0.0)))
    return Trace(calls=calls, label=t.label, family=t.family, metadata=t.metadata)


def provider_features(traces: list[Trace], cat: str) -> tuple[np.ndarray, np.ndarray]:
    rows, called = [], []
    for t in traces:
        s = provider_subtrace(t, cat)
        called.append(s.n > 0)
        rows.append(network_map(s) if s.n > 0 else np.zeros(NETWORK_DIM))
    return np.vstack(rows), np.array(called)


def est(X, y, prior, bank, bootstrap, seed, vantage):
    r = estimate_leakage(X, y, vantage=vantage, prior=prior, bank=bank,
                         bootstrap=bootstrap, seed=seed)
    return {"advantage": round(r.advantage, 4),
            "ci": [round(x, 4) for x in r.advantage_ci],
            "mi_bits": round(r.mi_lower_bound_bits, 4)}


def run(name: str, ds: TraceDataset, prior: np.ndarray, bank, bootstrap: int, seed: int):
    traces = ds.traces
    y = ds.labels()
    out = {"n_sessions": len(traces), "classes": int(len(np.unique(y))), "joint": {}, "single": {}}
    for v in ("network", "platform", "providers"):
        X, yy = featurize_dataset(ds, v)
        out["joint"][v] = est(X, yy, prior, bank, bootstrap, seed, v)
    for cat in TOOL_CATEGORIES:
        X, called = provider_features(traces, cat)
        row = {"sessions_calling": int(called.sum()),
               "coverage": round(float(called.mean()), 4)}
        if called.sum() == 0:
            out["single"][cat] = row
            continue
        row["all"] = est(X, y, prior, bank, bootstrap, seed, f"prov:{cat}")
        # Sessions that called this provider. Drop classes with fewer than 2 such
        # sessions (stratified split), and skip if too few remain to test on.
        yc, Xc = y[called], X[called]
        keep_cls = [c for c in np.unique(yc) if (yc == c).sum() >= 2]
        m = np.isin(yc, keep_cls)
        if m.sum() >= 30 and len(keep_cls) >= 2 and 0.4 * m.sum() >= len(keep_cls):
            row["called"] = est(Xc[m], yc[m], None, bank, bootstrap, seed, f"prov:{cat}|called")
            row["called"]["classes_present"] = len(keep_cls)
        out["single"][cat] = row
    advs = [r["all"]["advantage"] for r in out["single"].values() if "all" in r]
    advc = [r["called"]["advantage"] for r in out["single"].values() if "called" in r]
    out["summary"] = {
        "single_all_median": round(float(np.median(advs)), 4),
        "single_all_max": max(advs),
        "single_all_max_tool": max((r["all"]["advantage"], c) for c, r in out["single"].items() if "all" in r)[1],
        "single_called_median": round(float(np.median(advc)), 4) if advc else None,
        "single_called_max": max(advc) if advc else None,
    }
    print(name, out["summary"], {v: out["joint"][v]["advantage"] for v in out["joint"]}, flush=True)
    return out


def live_dataset(path: Path) -> TraceDataset:
    traces = list(iter_trace_file(path))
    return TraceDataset(traces=traces, taxonomy_name="live-6c", taxonomy_hash="n/a", source="live")


def main(out_path: str) -> None:
    results = {}
    # Synthetic 12-class, exactly the vantage stage's config (configs/default.yaml).
    cfg = load_config(CODE / "configs/default.yaml")
    ds, tax = build_dataset(cfg)
    results["synthetic_12c_seed20270214"] = run("synthetic", ds, tax.prior, cfg.bank, cfg.bootstrap, cfg.seed)

    uniform6 = np.full(6, 1 / 6)
    live = {
        "wan_scripted_seed20270215": (RESULTS / "wan/traces_wan_seed20270215.jsonl", 20270215, 200),
        "loopback_scripted_seed20270215": (RESULTS / "wan/traces_loopback_seed20270215.jsonl", 20270215, 200),
        "llm_local_seed20270214": (ROOT / "e2/llm_seed1_local/traces_seed20270214.jsonl", 20270214, 100),
        "llm_local_seed20270215": (ROOT / "e2/llm_seed2_local/traces_seed20270215.jsonl", 20270215, 100),
    }
    for name, (p, seed, boot) in live.items():
        results[name] = run(name, live_dataset(p), uniform6, None, boot, seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
