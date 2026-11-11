"""Generative shaping defense --- the Surakav analogue (paper §Defenses
"Generative shaping").

The paper states the strongest WF defenses now *generate* realistic decoy traces
with a learned model (Surakav [gong2022surakav]); the agent analogue is to shape a
trace toward a different task's signature. The paper ships only the evaluation hook
("designing one is future work"). This module implements the defense itself so that
the hook is exercised by something real rather than left empty.

Mechanism (Surakav's structure, at trace granularity):

1. **Learn** a per-class generative model of traces from the defender's own traffic:
   the empirical joint of (trace length, per-call category, request/response size,
   inter-call gap). This is the ``fit`` step; it needs the *defender's* traffic, not
   the user's current task.
2. At transform time, draw a **reference (decoy) trace** from the model for a class
   chosen **uniformly at random, independently of the true label**. Independence is
   what makes the shaped trace's schedule uninformative --- if the decoy were chosen
   as a function of the secret, the defense would leak by construction.
3. **Follow the reference**: the real payload is buffered and released on the
   reference's schedule and at the reference's sizes. A slot larger than the queued
   payload is filled with dummy bytes; a slot smaller leaves the remainder queued for
   the next slot. If payload remains once the reference is exhausted, extra slots are
   drawn from the same decoy class model (so the tail still looks like the decoy)
   until the queue drains.

Because releases follow the decoy reference, the residual leak is confined to *how
many* extra slots draining the real payload required --- i.e. to total volume, which
is exactly what Surakav's authors identify as the residual channel of a
generative shaper.

Honest scoping:

- Sizes/timing at the network vantage are fully shaped. At the **platform** vantage
  this defense corresponds to a client-side router that also batches and re-labels
  dispatches; a platform that observes true dispatches sees the real categories, so
  do not read the platform number as protection an untrusted platform would grant.
- The shaper is fitted on the same trace distribution it defends (the defender's
  own). No label is read at transform time; the estimator's train/test split is
  applied *after* shaping, so the attacker gets no oracle from the fit.

STATUS: implemented (this repository), beyond the paper's shipped artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leakbench.data.schemas import ToolCall, Trace, TraceDataset
from leakbench.defenses.base import Defense, register

MIN_SLOT_BYTES = 64.0


@dataclass
class _ClassModel:
    """The fitted per-class generative model: empirical pools to resample from."""

    lengths: np.ndarray            # observed trace lengths
    categories: list[str]          # pooled call categories
    request_sizes: np.ndarray      # pooled request sizes
    response_sizes: np.ndarray     # pooled response sizes
    gaps: np.ndarray               # pooled positive inter-call gaps

    def sample_reference(self, rng: np.random.Generator, n: int | None = None
                         ) -> list[tuple[str, float, float, float]]:
        """Draw a reference (decoy) trace: a list of (category, up, down, gap) slots."""
        if n is None:
            n = int(rng.choice(self.lengths))
        n = max(1, n)
        slots = []
        for i in range(n):
            cat = self.categories[int(rng.integers(len(self.categories)))]
            up = float(rng.choice(self.request_sizes))
            down = float(rng.choice(self.response_sizes))
            gap = 0.0 if i == 0 else float(rng.choice(self.gaps)) if self.gaps.size else 0.0
            slots.append((cat, max(MIN_SLOT_BYTES, up), max(MIN_SLOT_BYTES, down), gap))
        return slots


@register
@dataclass
class GenerativeShapingDefense(Defense):
    """Shape a trace onto a generated decoy trace (Surakav analogue).

    Parameters
    ----------
    slack:
        multiplicative headroom applied to each reference slot, so a shaper that
        follows a *slightly* larger schedule drains faster (Surakav's rate knob).
        ``1.0`` follows the decoy exactly.
    max_extra_slots:
        safety bound on the number of extra decoy slots used to drain the queue.
    """

    name: str = "gen-shaping"
    slack: float = 1.0
    max_extra_slots: int = 64
    _models: dict[int, _ClassModel] = field(default_factory=dict, init=False)
    _classes: tuple[int, ...] = field(default=(), init=False)

    def params(self) -> dict:
        return {"slack": self.slack, "max_extra_slots": self.max_extra_slots,
                "fitted_classes": len(self._models),
                "decoy_selection": "uniform over classes, independent of the true label"}

    # ------------------------------------------------------------------- fit
    def fit(self, dataset: TraceDataset, rng: np.random.Generator) -> None:
        pools: dict[int, dict[str, list]] = {}
        for t in dataset.traces:
            d = pools.setdefault(int(t.label),
                                 {"n": [], "cat": [], "up": [], "down": [], "gap": []})
            d["n"].append(t.n)
            for c in t.calls:
                d["cat"].append(c.category)
                d["up"].append(c.request_size)
                d["down"].append(c.response_size)
            g = t.gaps()[1:]
            d["gap"].extend([float(x) for x in g if x > 0])
        self._models = {}
        for label, d in pools.items():
            self._models[label] = _ClassModel(
                lengths=np.asarray(d["n"], dtype=int),
                categories=list(d["cat"]),
                request_sizes=np.asarray(d["up"], dtype=float),
                response_sizes=np.asarray(d["down"], dtype=float),
                gaps=np.asarray(d["gap"], dtype=float),
            )
        self._classes = tuple(sorted(self._models))

    # ------------------------------------------------------------- transform
    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        if not self._models:  # not fitted: degrade loudly rather than silently pass through
            raise RuntimeError(
                "GenerativeShapingDefense must be fitted before use "
                "(Defense.apply() calls fit(); call it directly for single traces)."
            )
        # The decoy class is drawn independently of the true label -- the property that
        # makes the released schedule uninformative about the secret.
        decoy = int(self._classes[int(rng.integers(len(self._classes)))])
        model = self._models[decoy]
        reference = model.sample_reference(rng)

        queue_up = float(trace.request_sizes().sum())
        queue_down = float(trace.response_sizes().sum())
        calls: list[ToolCall] = []
        extra = 0
        slot_idx = 0
        while True:
            if slot_idx < len(reference):
                cat, up, down, gap = reference[slot_idx]
            else:
                if extra >= self.max_extra_slots:
                    break
                cat, up, down, gap = model.sample_reference(rng, n=1)[0]
                gap = float(rng.choice(model.gaps)) if model.gaps.size else 0.0
                extra += 1
            up = max(MIN_SLOT_BYTES, up * self.slack)
            down = max(MIN_SLOT_BYTES, down * self.slack)
            calls.append(
                ToolCall(category=cat, request_size=up, response_size=down,
                         gap=(0.0 if not calls else gap))
            )
            queue_up = max(0.0, queue_up - up)
            queue_down = max(0.0, queue_down - down)
            slot_idx += 1
            if slot_idx >= len(reference) and queue_up <= 0 and queue_down <= 0:
                break
        if queue_up > 0 or queue_down > 0:
            # Bounded by max_extra_slots: flush the remainder rather than drop payload.
            calls.append(
                ToolCall(category=calls[-1].category if calls else "fetch",
                         request_size=max(MIN_SLOT_BYTES, queue_up),
                         response_size=max(MIN_SLOT_BYTES, queue_down),
                         gap=float(rng.choice(model.gaps)) if model.gaps.size else 0.0)
            )
        return Trace(
            calls=calls,
            label=trace.label,
            family=trace.family,
            metadata={**trace.metadata, "defense": self.name, "decoy_class": decoy,
                      "shaped_slots": len(calls), "extra_slots": extra},
        )
