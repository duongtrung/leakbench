"""Observation maps ``O_v`` --- Contribution C1, Def. "Observation map".

Three canonical vantages (paper Table "observable-feature model"):

- **NET (network):** length (coarsened), total up/down bytes, log-spaced size
  histogram (8 buckets 256 B–64 KB), inter-arrival histogram (7 buckets 50 ms–5 s),
  dispersion statistics. Sees sizes+timing but NOT tool categories.
- **PLAT (platform):** per-category unigram counts (8), 8x8 bigram transition matrix
  (order), length, timing histogram. Sees categories+order+count+timing but NOT sizes.
- **PROV (providers):** concat(PLAT, NET) + per-category size aggregates. Refines both.

Because ``O_prov`` literally concatenates the ``O_plat`` and ``O_net`` sub-vectors,
the refinement order ``O_net ⪯ O_prov`` and ``O_plat ⪯ O_prov`` holds *by
construction* (Def. "Refinement order"). NET and PLAT are **incomparable** (sizes vs
categories), so the empirical ``NET < PLAT`` ordering is a finding, not a corollary.

Every map is a **pure function of a trace**, so any trace source (generator, measured
harness, live adapter) feeds them unchanged (paper §Impl "Observation maps").

Named feature blocks are exposed via ``feature_slices`` so the ablation (RQ2) and
projection (§Projections) views can select feature subsets by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from leakbench.data.schemas import Trace, TraceDataset
from leakbench.taxonomy import NUM_TOOLS

# Fixed histogram bucket edges (paper App. "Equation Audit / hyperparameters").
SIZE_BUCKETS = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 65536], dtype=float)
GAP_BUCKETS = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0], dtype=float)

VANTAGES: tuple[str, ...] = ("network", "platform", "providers")

# Feature-vector widths, derived from the bucket definitions rather than written out,
# so changing a bucket set cannot silently desynchronise the projection views from the
# maps. ``test_observables_defenses`` asserts these match the maps' actual output.
_N_STATS = 6                                              # _safe_stats width
NETWORK_DIM: int = 4 + len(SIZE_BUCKETS) + len(GAP_BUCKETS) + 2 * _N_STATS
PLATFORM_DIM: int = 1 + NUM_TOOLS + NUM_TOOLS * NUM_TOOLS + len(GAP_BUCKETS)
PROVIDER_DIM: int = PLATFORM_DIM + NETWORK_DIM + 2 * NUM_TOOLS

VANTAGE_DIMS: dict[str, int] = {
    "network": NETWORK_DIM,
    "platform": PLATFORM_DIM,
    "providers": PROVIDER_DIM,
}


# ------------------------------------------------------------------ helpers
def _histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Right-open bucket counts with an overflow bucket, normalized to a density.

    ``len(edges)`` buckets: bucket k counts values in [edges[k-1], edges[k]) with a
    leading (< edges[0]) and trailing (>= edges[-1]) tail folded into the ends.
    Returns a length-``len(edges)`` vector summing to 1 (or zeros for empty input).
    """
    if values.size == 0:
        return np.zeros(len(edges), dtype=float)
    # Buckets: (-inf, e0), [e0,e1), ..., [e_{m-2}, e_{m-1}), [e_{m-1}, inf)
    idx = np.searchsorted(edges, values, side="right")
    idx = np.clip(idx, 0, len(edges) - 1)
    counts = np.bincount(idx, minlength=len(edges)).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _safe_stats(values: np.ndarray) -> np.ndarray:
    """Dispersion statistics [mean, std, cv, min, max, sum-in-log] robust to empties."""
    if values.size == 0:
        return np.zeros(6, dtype=float)
    mean = float(values.mean())
    std = float(values.std())
    cv = std / mean if mean > 0 else 0.0
    return np.array(
        [mean, std, cv, float(values.min()), float(values.max()), float(np.log1p(values.sum()))],
        dtype=float,
    )


# ------------------------------------------------------------ network map O_net
def network_map(trace: Trace) -> np.ndarray:
    """``O_net(rho) = phi_net(n, {s_up+s_down}, {Delta})``.

    Deliberately the information a TLS/QUIC on-path observer can recover: length,
    total volume, binned sizes/timing, dispersion --- and NOT tool categories.
    """
    n = trace.n
    total = trace.total_sizes()
    gaps = trace.gaps()
    ups = trace.request_sizes()
    downs = trace.response_sizes()

    feats = [
        np.array([float(n), np.log1p(ups.sum()), np.log1p(downs.sum()),
                  np.log1p(total.sum())], dtype=float),   # length + volume totals
        _histogram(total, SIZE_BUCKETS),                  # size histogram (8)
        _histogram(gaps[1:] if n > 1 else np.array([]), GAP_BUCKETS),  # timing (7)
        _safe_stats(total),                               # size dispersion (6)
        _safe_stats(gaps[1:] if n > 1 else np.array([])), # timing dispersion (6)
    ]
    return np.concatenate(feats)


def network_feature_slices() -> dict[str, slice]:
    """Named blocks within ``O_net`` for the ablation / projection views."""
    n_size, n_gap = len(SIZE_BUCKETS), len(GAP_BUCKETS)
    a = 4                       # length + up/down/total byte totals
    b = a + n_size              # size histogram
    c = b + n_gap               # inter-arrival histogram
    d = c + _N_STATS            # size dispersion
    e = d + _N_STATS            # timing dispersion
    return {
        "volume": slice(0, a),
        "size_hist": slice(a, b),
        "timing_hist": slice(b, c),
        "size_stats": slice(c, d),
        "timing_stats": slice(d, e),
    }


# --------------------------------------------------------- platform map O_plat
def platform_map(trace: Trace) -> np.ndarray:
    """``O_plat(rho) = phi_plat({c_i}, {c_i->c_{i+1}}, {Delta})``.

    Semantic dispatch log: per-category unigram counts, 8x8 bigram transition matrix
    (order), length, timing histogram. NO sizes (that is what makes it incomparable
    with the network vantage).
    """
    cats = trace.categories()
    n = trace.n
    gaps = trace.gaps()

    unigram = trace.category_counts()                    # (8,)
    bigram = np.zeros((NUM_TOOLS, NUM_TOOLS), dtype=float)
    for a, b in zip(cats[:-1], cats[1:]):
        bigram[a, b] += 1.0
    bigram_flat = bigram.reshape(-1)                     # (64,)

    feats = [
        np.array([float(n)], dtype=float),
        unigram,
        bigram_flat,
        _histogram(gaps[1:] if n > 1 else np.array([]), GAP_BUCKETS),
    ]
    return np.concatenate(feats)


def platform_feature_slices() -> dict[str, slice]:
    uni_end = 1 + NUM_TOOLS
    bi_end = uni_end + NUM_TOOLS * NUM_TOOLS
    return {
        "length": slice(0, 1),
        "unigram": slice(1, uni_end),
        "bigram": slice(uni_end, bi_end),
        "timing_hist": slice(bi_end, PLATFORM_DIM),
    }


# --------------------------------------------------------- provider map O_prov
def provider_map(trace: Trace) -> np.ndarray:
    """``O_prov = (O_plat, O_net, per-category size aggregates)``.

    Concatenates both shallower maps (guaranteeing the refinement order) plus a
    per-category mean up/down size aggregate (colluding providers' per-category size
    profiles).
    """
    plat = platform_map(trace)
    net = network_map(trace)
    # Per-category size aggregates: mean up and mean down size for each category.
    cats = trace.categories()
    ups = trace.request_sizes()
    downs = trace.response_sizes()
    agg_up = np.zeros(NUM_TOOLS, dtype=float)
    agg_down = np.zeros(NUM_TOOLS, dtype=float)
    cnt = np.zeros(NUM_TOOLS, dtype=float)
    for c, u, d in zip(cats, ups, downs):
        agg_up[c] += u
        agg_down[c] += d
        cnt[c] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_up = np.where(cnt > 0, agg_up / np.maximum(cnt, 1), 0.0)
        mean_down = np.where(cnt > 0, agg_down / np.maximum(cnt, 1), 0.0)
    size_agg = np.concatenate([np.log1p(mean_up), np.log1p(mean_down)])  # (16,)
    return np.concatenate([plat, net, size_agg])


ObservationMap = Callable[[Trace], np.ndarray]

_MAPS: dict[str, ObservationMap] = {
    "network": network_map,
    "platform": platform_map,
    "providers": provider_map,
}


def get_map(vantage: str) -> ObservationMap:
    if vantage not in _MAPS:
        raise ValueError(f"Unknown vantage {vantage!r}; expected one of {VANTAGES}.")
    return _MAPS[vantage]


def featurize_dataset(dataset: TraceDataset, vantage: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``O_v`` to every trace, returning ``(X, y)``.

    ``X`` rows may have differing implicit block sizes only if traces differ in
    schema; all maps here are fixed-dimension, so ``X`` is a dense 2D array.
    """
    fn = get_map(vantage)
    X = np.vstack([fn(t) for t in dataset.traces])
    y = dataset.labels()
    return X, y


@dataclass(frozen=True)
class VantageOrder:
    """The refinement partial order over vantages (Def. "Refinement order").

    ``refines[v]`` = set of vantages that ``v`` refines (i.e. is at least as
    informative as). PROV refines both NET and PLAT; NET and PLAT are incomparable.
    """

    refines: dict[str, frozenset[str]]

    @staticmethod
    def canonical() -> "VantageOrder":
        return VantageOrder(
            refines={
                "network": frozenset({"network"}),
                "platform": frozenset({"platform"}),
                "providers": frozenset({"network", "platform", "providers"}),
            }
        )

    def dominates(self, v: str, u: str) -> bool:
        """True if vantage ``v`` refines ``u`` (so ``Adv_v >= Adv_u`` is forced)."""
        return u in self.refines.get(v, frozenset())
