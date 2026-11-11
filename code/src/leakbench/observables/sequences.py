"""Raw-sequence representation of a trace --- the paper's planned experiment (iii),
"a learned sequence-model attacker over raw traces as a tighter estimator"
(App. "Additional Results").

The observation maps of :mod:`leakbench.observables.maps` **summarise**: they bin
sizes and gaps into histograms and collapse categories into uni/bigram counts. That
summary is a modelling choice, not a property of the vantage — a real on-path observer
sees packets *in order*, and a real platform sees dispatches *in order*. Histograms
throw that ordering away, so the advantage estimated from them is a looser lower bound
than it needs to be.

This module keeps the same per-vantage *information* but drops the aggregation: each
trace becomes an ordered, padded matrix of per-call feature vectors. Crucially the
vantage restriction is preserved exactly:

- **network**: per call ``[log1p(s_up + s_down), log1p(Delta)]`` — sizes and timing,
  **no categories** (as in ``O_net``, which bins ``{s_up+s_down}`` and ``{Delta}``);
- **platform**: per call ``[one-hot(category) (8), log1p(Delta)]`` — categories, order
  and timing, **no sizes** (as in ``O_plat``);
- **providers**: per call ``[one-hot(category), log1p(s_up), log1p(s_down),
  log1p(Delta)]`` — everything, matching ``O_prov``.

A trailing validity flag per position distinguishes real calls from padding, so trace
length stays recoverable without the learner having to infer it from zeros.

Because advantage and MI are **lower** bounds, an attacker that reads this richer
representation can only tighten them (paper §Estimator: report the strongest
attacker). It is therefore a legitimate improvement on the reported value, not a
different measurement — see `run_sequence_attacker`.
"""

from __future__ import annotations

import numpy as np

from leakbench.data.schemas import Trace, TraceDataset
from leakbench.taxonomy import NUM_TOOLS

# Covers ~99% of traces at the paper's generator settings (median 10, p95 21, max 33).
DEFAULT_MAX_LEN = 32

# Per-call feature width by vantage, excluding the validity flag.
_WIDTHS = {"network": 2, "platform": NUM_TOOLS + 1, "providers": NUM_TOOLS + 3}


def sequence_width(vantage: str) -> int:
    """Per-call feature width (including the validity flag)."""
    if vantage not in _WIDTHS:
        raise ValueError(f"Unknown vantage {vantage!r}; expected {list(_WIDTHS)}.")
    return _WIDTHS[vantage] + 1


def sequence_map(trace: Trace, vantage: str, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """Ordered, padded per-call matrix for one trace, flattened row-major.

    Truncates beyond ``max_len`` and zero-pads below it; the validity flag marks which
    positions are real.
    """
    width = sequence_width(vantage)
    out = np.zeros((max_len, width), dtype=float)

    cats = trace.categories()
    ups = trace.request_sizes()
    downs = trace.response_sizes()
    gaps = trace.gaps()
    n = min(trace.n, max_len)

    for i in range(n):
        row = out[i]
        if vantage == "network":
            row[0] = np.log1p(ups[i] + downs[i])
            row[1] = np.log1p(gaps[i])
        elif vantage == "platform":
            row[cats[i]] = 1.0
            row[NUM_TOOLS] = np.log1p(gaps[i])
        else:  # providers
            row[cats[i]] = 1.0
            row[NUM_TOOLS] = np.log1p(ups[i])
            row[NUM_TOOLS + 1] = np.log1p(downs[i])
            row[NUM_TOOLS + 2] = np.log1p(gaps[i])
        row[width - 1] = 1.0  # validity flag
    return out.reshape(-1)


def featurize_sequences(
    dataset: TraceDataset, vantage: str, max_len: int = DEFAULT_MAX_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`sequence_map` to every trace, returning ``(X, y)``."""
    X = np.vstack([sequence_map(t, vantage, max_len) for t in dataset.traces])
    return X, dataset.labels()


def truncation_rate(dataset: TraceDataset, max_len: int = DEFAULT_MAX_LEN) -> float:
    """Fraction of traces longer than ``max_len`` (i.e. losing tail calls).

    Reported alongside any sequence-attacker result: a high rate means the
    representation is discarding real observations and the comparison against the
    summary features is no longer like-for-like.
    """
    if not dataset.traces:
        return 0.0
    return float(np.mean([t.n > max_len for t in dataset.traces]))
