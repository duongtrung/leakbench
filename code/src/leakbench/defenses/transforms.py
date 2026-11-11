"""Concrete defenses (paper Table "defense" and §Defenses).

Each maps to a website-fingerprinting ancestor. Costs (bandwidth/latency) emerge
from the transform applied to real traces, not hard-coded; the paper's reported
multipliers are an order-of-magnitude indication for its distribution.
"""

from __future__ import annotations

import numpy as np

from leakbench.data.schemas import ToolCall, Trace
from leakbench.defenses.base import Defense, register

# Size-padding buckets.
PAD_512 = 512


def _pad_up(size: float, bucket: int) -> float:
    return float(np.ceil(size / bucket) * bucket)


@register
class NoneDefense(Defense):
    """Undefended baseline."""

    name = "none"

    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        return trace


@register
class Pad512Defense(Defense):
    """Pad each request/response up to the next 512-byte bucket (WF cell padding,
    Tamaraw). Coarse buckets are nearly free and nearly useless (paper §Defenses)."""

    name = "pad-512"

    def __init__(self, bucket: int = PAD_512) -> None:
        self.bucket = bucket

    def params(self) -> dict:
        return {"bucket": self.bucket}

    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        calls = [
            ToolCall(
                category=c.category,
                request_size=_pad_up(c.request_size, self.bucket),
                response_size=_pad_up(c.response_size, self.bucket),
                gap=c.gap,
            )
            for c in trace.calls
        ]
        return _rebuild(trace, calls)


@register
class PadMaxDefense(Defense):
    """Constant-size records: pad every request/response to the dataset-wide max
    bucket (WF constant-size padding). Works but costs ~6x bandwidth (paper)."""

    name = "pad-max"

    def __init__(self, max_size: int = 65536) -> None:
        self.max_size = max_size

    def params(self) -> dict:
        return {"max_size": self.max_size}

    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        calls = [
            ToolCall(
                category=c.category,
                request_size=float(self.max_size),
                response_size=float(self.max_size),
                gap=c.gap,
            )
            for c in trace.calls
        ]
        return _rebuild(trace, calls)


@register
class ShuffleDefense(Defense):
    """Reorder calls within the trace (WF reordering/batching, Tamaraw). Removes the
    bigram/order signal at the platform. Paper finding: inert against the count-based
    platform attacker (0.68 -> 0.68) --- exactly why measured defense eval matters."""

    name = "shuffle"

    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        order = rng.permutation(len(trace.calls))
        calls = [trace.calls[i] for i in order]
        # Re-assign gaps so the first shuffled call has gap 0 (timing preserved in
        # aggregate; only order changes).
        gaps = [c.gap for c in trace.calls]
        calls = [
            ToolCall(category=c.category, request_size=c.request_size,
                     response_size=c.response_size, gap=(0.0 if i == 0 else gaps[i]))
            for i, c in enumerate(calls)
        ]
        return _rebuild(trace, calls)


@register
class ConstantRateDefense(Defense):
    """Constant-rate scheduling (Tamaraw timing): replace inter-call gaps with a
    fixed grid interval. Attacks the timing channel --- the *weaker* channel at the
    network vantage, so it helps little there (paper §Defenses "Timing regularization")."""

    name = "constant-rate"

    def __init__(self, interval: float | None = None) -> None:
        self.interval = interval

    def params(self) -> dict:
        return {"interval": self.interval}

    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        gaps = trace.gaps()
        # Grid = mean positive gap if not specified.
        pos = gaps[gaps > 0]
        interval = self.interval if self.interval is not None else (
            float(pos.mean()) if pos.size else 0.0
        )
        calls = [
            ToolCall(category=c.category, request_size=c.request_size,
                     response_size=c.response_size, gap=(0.0 if i == 0 else interval))
            for i, c in enumerate(trace.calls)
        ]
        return _rebuild(trace, calls)


@register
class CoverCallDefense(Defense):
    """Inject dummy tool calls (FRONT's front-loaded dummies / mix-net decoys).

    Obfuscates counts and the category histogram at the platform. Cover calls are
    drawn category-balanced (uniform over tool categories) so they perturb the
    per-category *count* histogram the platform relies on (paper §Defenses "Order
    hiding": only count-perturbing cover moves platform advantage)."""

    name = "cover"

    def __init__(self, rate: float = 0.9, category_balanced: bool = True) -> None:
        # rate = expected fraction of extra calls relative to trace length.
        self.rate = rate
        self.category_balanced = category_balanced

    def params(self) -> dict:
        return {"rate": self.rate, "category_balanced": self.category_balanced}

    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        from leakbench.taxonomy import TOOL_CATEGORIES

        n = len(trace.calls)
        n_cover = int(rng.poisson(self.rate * max(1, n)))
        calls = list(trace.calls)
        # Sizes/gaps for cover calls drawn from the trace's own empirical scale so
        # they blend in.
        sizes_up = trace.request_sizes()
        sizes_down = trace.response_sizes()
        gaps = trace.gaps()
        gap_scale = float(gaps[gaps > 0].mean()) if np.any(gaps > 0) else 0.1
        for _ in range(n_cover):
            cat = (TOOL_CATEGORIES[int(rng.integers(len(TOOL_CATEGORIES)))]
                   if self.category_balanced
                   else trace.calls[int(rng.integers(n))].category)
            calls.append(
                ToolCall(
                    category=cat,
                    request_size=float(np.ceil(rng.choice(sizes_up))),
                    response_size=float(np.ceil(rng.choice(sizes_down))),
                    gap=float(abs(rng.normal(gap_scale, gap_scale * 0.3))),
                )
            )
        # Shuffle so cover calls are interleaved, then fix the leading gap.
        order = rng.permutation(len(calls))
        calls = [calls[i] for i in order]
        calls = [
            ToolCall(category=c.category, request_size=c.request_size,
                     response_size=c.response_size, gap=(0.0 if i == 0 else c.gap))
            for i, c in enumerate(calls)
        ]
        return _rebuild(trace, calls)


@register
class FullDefense(Defense):
    """Combined Tamaraw-style defense: constant size + grid timing + cover calls
    (paper "full" row). Drives the network advantage lowest, dominated by the size
    component, at order-of-magnitude bandwidth cost."""

    name = "full"

    def __init__(self, max_size: int = 65536, interval: float | None = None,
                 cover_rate: float = 0.9) -> None:
        self.pad = PadMaxDefense(max_size=max_size)
        self.rate = ConstantRateDefense(interval=interval)
        self.cover = CoverCallDefense(rate=cover_rate, category_balanced=True)

    def params(self) -> dict:
        return {"components": ["pad-max", "constant-rate", "cover"]}

    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        t = self.cover.transform(trace, rng)     # inject cover first (adds calls)
        t = self.rate.transform(t, rng)          # regularize timing
        t = self.pad.transform(t, rng)           # constant-size padding
        return t


def _rebuild(trace: Trace, calls: list[ToolCall]) -> Trace:
    return Trace(
        calls=calls,
        label=trace.label,
        family=trace.family,
        metadata={**trace.metadata},
    )
