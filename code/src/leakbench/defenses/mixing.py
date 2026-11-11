"""Multi-user mixing at a shared gateway --- the paper's planned experiment (iv).

The paper lists "multi-user mixing at a shared gateway" among its planned experiments
and, in §Defenses, argues that padding should be applied "at a shared gateway/MCP
router so the cost is amortized across applications". Mixing is the other half of that
argument: if several users' tool calls traverse one router, an on-path observer sees
the **union** of their streams, not one user's trace.

That makes mixing the agent-trace analogue of a mix network
[chaum1981mix, piotrowska2017loopix] --- and it is economically unlike every other
defense in the toolbox:

- padding, cover calls and shaping **manufacture** bytes, so the bandwidth they cost is
  real and is charged to the user who is being protected;
- mixing **reuses** bytes that other users were sending anyway. The observer's view
  grows, but no new traffic is created. Per-user marginal bandwidth is ~0; the cost is
  latency (the gateway may hold a call to align it with others) and the requirement
  that enough concurrent users exist.

Implementation. ``transform`` merges the target trace with ``n_users - 1`` traces drawn
from **other task classes** in the defender's own traffic, interleaved on their arrival
timeline, and re-derives inter-arrival gaps for the merged stream. The returned trace
keeps the **target's** label, so the estimator answers the deployment question: *given
the mixed stream a gateway emits, can the adversary still infer this user's task?*

The anonymity-set caveat, stated plainly: this models the observer's view of a shared
pipe. It does not model an adversary who can additionally separate flows by connection,
timing correlation, or provider-side identity --- the paper's own advice to "route calls
through per-task ephemeral identities" exists precisely because such separation is often
possible. Read the numbers as an upper bound on what mixing buys.

STATUS: implemented (this repository); the paper lists it as planned work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leakbench.data.schemas import ToolCall, Trace, TraceDataset
from leakbench.defenses.base import Defense, register


@register
@dataclass
class MixingDefense(Defense):
    """Interleave a user's calls with other users' at a shared gateway.

    Parameters
    ----------
    n_users:
        size of the mix, counting the target. ``1`` is a no-op (the observer sees one
        user); larger values mean more concurrent users behind the gateway.
    cross_class_only:
        draw the co-mixed users from *different* task classes (default). Mixing with
        another user running the same task would inflate the apparent protection,
        because the observer's job is then genuinely easier, not harder.
    """

    name: str = "mixing"
    n_users: int = 4
    cross_class_only: bool = True
    _pool: dict[int, list[Trace]] = field(default_factory=dict, init=False)
    _labels: tuple[int, ...] = field(default=(), init=False)

    def params(self) -> dict:
        return {
            "n_users": self.n_users,
            "cross_class_only": self.cross_class_only,
            "bandwidth_is_amortized": True,
            "note": "reported bandwidth is the growth of the OBSERVED stream, not new "
                    "bytes: the co-mixed traffic would have been sent anyway, so the "
                    "per-user marginal cost is ~0",
        }

    # ------------------------------------------------------------------- fit
    def fit(self, dataset: TraceDataset, rng: np.random.Generator) -> None:
        pool: dict[int, list[Trace]] = {}
        for t in dataset.traces:
            pool.setdefault(int(t.label), []).append(t)
        self._pool = pool
        self._labels = tuple(sorted(pool))

    # ------------------------------------------------------------- transform
    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        if not self._pool:
            raise RuntimeError(
                "MixingDefense must be fitted on the gateway's traffic before use "
                "(Defense.apply() calls fit())."
            )
        if self.n_users <= 1:
            return trace

        others = [lab for lab in self._labels
                  if not (self.cross_class_only and lab == trace.label)]
        if not others:
            return trace

        # (arrival_time, call) for the target and for each co-mixed user. Co-mixed
        # users start at an independent offset: they are not synchronised with the
        # target, and assuming they were would overstate the mixing.
        events: list[tuple[float, ToolCall]] = list(
            zip(np.cumsum(trace.gaps()), trace.calls)
        )
        span = float(np.sum(trace.gaps())) or 1.0
        for _ in range(self.n_users - 1):
            lab = others[int(rng.integers(len(others)))]
            candidates = self._pool[lab]
            other = candidates[int(rng.integers(len(candidates)))]
            offset = float(rng.uniform(-0.5, 0.5) * span)
            events += list(zip(np.cumsum(other.gaps()) + offset, other.calls))

        events.sort(key=lambda ev: ev[0])
        t0 = events[0][0]
        calls: list[ToolCall] = []
        prev = t0
        for i, (t, call) in enumerate(events):
            calls.append(
                ToolCall(category=call.category, request_size=call.request_size,
                         response_size=call.response_size,
                         gap=(0.0 if i == 0 else max(0.0, t - prev)))
            )
            prev = t
        return Trace(
            calls=calls,
            label=trace.label,          # the secret is still THIS user's task
            family=trace.family,
            metadata={**trace.metadata, "defense": self.name,
                      "n_users": self.n_users, "observed_calls": len(calls)},
        )
