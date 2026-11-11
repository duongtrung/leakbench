"""Differentially private protocol-level shaping --- the NetShaper analogue, and the
``epsilon`` -> advantage mapping the paper calls "the key open problem"
(paper §Defenses "Protocol-level mitigation", §Discussion).

The paper argues a DP traffic-shaping layer at the gateway/MCP router
[sabzi2024netshaper] is the most promising mitigation, and that *connecting our
advantage axis to a formal DP budget is the key open problem*. This module supplies
both halves of that connection:

**1. The mechanism.** :class:`DPShapingDefense` buffers the trace's bytes and
releases them on a fixed time grid. In each window the release size is the output of
a discrete-Laplace mechanism applied to the (quantized, clamped) queue length; the
released bytes are padded with dummies when the queue is short and the surplus stays
queued when it is long. The observable --- the sequence of released sizes on a fixed
grid --- is therefore a **post-processing of the mechanism outputs**, so the DP
guarantee carries to everything the network vantage sees, including trace length and
timing.

Privacy accounting (stated exactly, so it can be checked):

- Domain: each window's query is the queue length quantized to ``quantum`` bytes and
  clamped to ``0..L`` where ``L = cap // quantum``.
- Grid: the release schedule has a **fixed** number of windows ``W`` on a fixed
  interval, independent of the trace. This matters: if shaping stopped when the queue
  drained, the *number* of releases would itself be a data-dependent quantity outside
  the mechanism, and the guarantee would not hold for the observable the network
  vantage actually sees (we measured exactly that leak before fixing the grid).
- Adjacency, ``all_pairs`` (default): any two clamped queue values are neighbours, so
  sensitivity ``Delta = L``. This is the strong adjacency needed for a bound that
  holds between *any two tasks*.
- Adjacency, ``per_window``: neighbours differ by one quantum in one window
  (``Delta = 1``), which is NetShaper's style of guarantee. It is far cheaper, but the
  task-inference bound then scales with the *distance* between two task traces
  (``d`` quanta), i.e. ``e^{eps d}``, which is vacuous for realistic ``d``. Reporting
  both modes is the honest answer to "does a DP shaping epsilon bound task leakage?":
  only under all-pairs adjacency, and that is expensive.
- Per-window mechanism: discrete Laplace with scale ``b = Delta / eps_win`` units is
  ``eps_win``-DP. With ``W`` windows and (adaptive) sequential composition, the whole
  released sequence is ``eps = W * eps_win``-DP.

**2. The bound.** For a mechanism that is ``eps``-DP between *any two* secrets,
posterior Bayes vulnerability satisfies ``V(pi, M) <= e^eps V(pi)``
[alvim2012gleakage], hence the attacker advantage of Def. "Bayes risk and advantage"
obeys::

    Adv <= min(1, V_prior * (e^eps - 1) / R_pi),      V_prior = max_t pi(t),
                                                      R_pi    = 1 - V_prior

which for a uniform prior over ``K`` classes is ``(e^eps - 1) / (K - 1)``. See
:func:`dp_advantage_ceiling`. Sweeping ``epsilon`` and plotting the *measured*
advantage against this ceiling is the eps-to-advantage mapping; the measured value
must always sit below the ceiling, which makes the sweep a falsifiability check on
both the mechanism and the estimator.

Honest scoping:

- The bound is unconditional only under the all-pairs adjacency above. Under it, small
  ``epsilon`` costs an enormous amount of bandwidth --- which is the finding, not a
  bug: it is why NetShaper trades a weaker adjacency for affordable overhead.
- A trace whose payload does not drain inside the horizon is flushed rather than
  truncated; those traces are counted in ``overflow_traces`` and their release
  sequence is **not** covered by the stated epsilon. Raise ``windows`` or ``cap_b``
  until the count is 0 (every runner reports it).
- This defends the **network** vantage. A platform that observes true dispatches is
  unaffected; DP shaping at a gateway cannot hide what the gateway itself sees.

STATUS: implemented (this repository), beyond the paper's shipped artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from leakbench.data.schemas import ToolCall, Trace
from leakbench.defenses.base import Defense, register

DEFAULT_WINDOW_S = 0.25
DEFAULT_QUANTUM_B = 512
DEFAULT_CAP_B = 16384
DEFAULT_MAX_WINDOWS = 32


def dp_advantage_ceiling(epsilon: float, prior: np.ndarray | int,
                         distance: float = 1.0) -> float:
    """Upper bound on attacker advantage for an ``epsilon``-DP observable.

    ``prior`` is either the prior vector ``pi`` or the number of classes ``K``
    (uniform assumed). ``distance`` is the adjacency distance between the two secrets
    being distinguished, in units of the mechanism's sensitivity: ``1.0`` for
    all-pairs adjacency (any two traces are neighbours), larger under a per-window
    adjacency where two task traces differ by many quanta.

    Derivation: for an ``eps``-DP mechanism, ``P(x|t) <= e^{eps*d} P(x|t')`` for
    secrets at distance ``d``, so

        V_post = sum_x max_t pi(t) P(x|t)
              <= V_prior * e^{eps*d} * sum_x avg_t P(x|t) = V_prior * e^{eps*d},

    and ``Adv = (V_post - V_prior)/R_pi <= V_prior (e^{eps*d} - 1)/R_pi``
    [alvim2012gleakage]. The measured advantage must always fall below this, which
    makes an epsilon sweep a falsifiability check on mechanism and estimator alike.
    """
    if isinstance(prior, (int, np.integer)):
        v_prior = 1.0 / float(prior)
    else:
        v_prior = float(np.max(np.asarray(prior, dtype=float)))
    r_pi = 1.0 - v_prior
    if r_pi <= 0:
        return 0.0
    exponent = epsilon * max(0.0, distance)
    if exponent > 700:  # e^exponent overflows; the bound is vacuous long before this
        return 1.0
    return float(min(1.0, v_prior * (math.exp(exponent) - 1.0) / r_pi))


def discrete_laplace(rng: np.random.Generator, scale: float, size: int | None = None):
    """Discrete Laplace (two-sided geometric) noise with ``P(k) ~ exp(-|k|/scale)``.

    Sampled as the difference of two geometric variables, which is the standard
    construction and is ``(1/scale)``-DP per unit of sensitivity.
    """
    if scale <= 0:
        return np.zeros(size or 1, dtype=int) if size is not None else 0
    p = 1.0 - math.exp(-1.0 / scale)
    g1 = rng.geometric(p, size=size) - 1
    g2 = rng.geometric(p, size=size) - 1
    return g1 - g2


@register
@dataclass
class DPShapingDefense(Defense):
    """DP traffic shaping at a gateway / MCP router (NetShaper analogue).

    Parameters
    ----------
    epsilon:
        **total** privacy budget for one trace's released sequence, under the
        adjacency selected by ``adjacency``.
    window_s:
        shaping interval (seconds). Releases happen on this fixed grid, so the
        released timing carries no data-dependent information. ``window_s * windows``
        is the shaping horizon and is a *public deployment parameter* ("sessions are
        shaped for at most this long"), not a per-trace quantity.
    quantum_b:
        release granularity in bytes.
    cap_b:
        maximum bytes released per window; also sets the clamped query range.
    windows:
        the fixed number of shaping windows ``W``. Always emitted in full: a
        data-dependent stopping rule would leak outside the mechanism.
    adjacency:
        ``all_pairs`` (sensitivity ``L``; the ceiling of
        :func:`dp_advantage_ceiling` holds at distance 1) or ``per_window``
        (sensitivity 1 quantum; NetShaper-style, cheap, but the task-level ceiling
        must be evaluated at the real distance between task traces).
    """

    name: str = "dp-shaping"
    epsilon: float = 4.0
    window_s: float | None = None
    quantum_b: int = DEFAULT_QUANTUM_B
    cap_b: int = DEFAULT_CAP_B
    windows: int = DEFAULT_MAX_WINDOWS
    adjacency: str = "all_pairs"
    overflow_traces: int = field(default=0, init=False)
    n_transformed: int = field(default=0, init=False)
    _window_s: float = field(default=DEFAULT_WINDOW_S, init=False)
    _auto_window: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.adjacency not in ("all_pairs", "per_window"):
            raise ValueError(
                f"adjacency must be 'all_pairs' or 'per_window', got {self.adjacency!r}."
            )
        self._window_s = self.window_s if self.window_s is not None else DEFAULT_WINDOW_S
        self._auto_window = self.window_s is None

    def fit(self, dataset, rng) -> None:  # type: ignore[override]
        """Size the shaping horizon to cover the deployment's longest session.

        ``window_s * windows`` is a **public** deployment parameter ("sessions are
        shaped for at most this long"), chosen once for a deployment rather than per
        trace, so deriving it from the corpus does not make the release schedule
        data-dependent. Pass ``window_s`` explicitly to pin it.
        """
        # Counters describe one dataset pass, so a reused instance must not carry the
        # previous pass's overflow count into the next result file.
        self.overflow_traces = 0
        self.n_transformed = 0
        if not self._auto_window:
            return
        longest = max((float(t.gaps().sum()) for t in dataset.traces), default=0.0)
        self._window_s = max(1e-3, longest / max(1, self.windows))

    # ----------------------------------------------------------------- helpers
    @property
    def levels(self) -> int:
        """``L`` --- the number of quanta in the clamped query range."""
        return max(1, int(self.cap_b // self.quantum_b))

    @property
    def sensitivity_units(self) -> int:
        """``Delta`` in quanta, set by the declared adjacency."""
        return self.levels if self.adjacency == "all_pairs" else 1

    @property
    def epsilon_per_window(self) -> float:
        return self.epsilon / max(1, self.windows)

    @property
    def noise_scale_units(self) -> float:
        """``b = Delta / eps_win``."""
        eps_win = self.epsilon_per_window
        return float(self.sensitivity_units) / eps_win if eps_win > 0 else float("inf")

    @property
    def horizon_s(self) -> float:
        return self._window_s * self.windows

    def params(self) -> dict:
        return {
            "epsilon_total": self.epsilon,
            "epsilon_per_window": self.epsilon_per_window,
            "window_s": self._window_s,
            "window_s_auto_sized": self._auto_window,
            "quantum_b": self.quantum_b,
            "cap_b": self.cap_b,
            "windows_W": self.windows,
            "horizon_s": self.horizon_s,
            "levels_L": self.levels,
            "adjacency": self.adjacency,
            "sensitivity_units": self.sensitivity_units,
            "noise_scale_units": self.noise_scale_units,
            "mechanism": "discrete_laplace_on_quantized_clamped_queue",
            "overflow_traces": self.overflow_traces,
            "n_transformed": self.n_transformed,
        }

    # --------------------------------------------------------------- transform
    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        # Arrival schedule: call i's bytes become available at its arrival time.
        arrival_t = np.cumsum(trace.gaps())
        up = trace.request_sizes()
        down = trace.response_sizes()

        L = self.levels
        scale = self.noise_scale_units
        window = max(1e-9, self._window_s)
        W = max(1, self.windows)
        # Which real call falls in each window: computed once, not re-scanned per
        # window (this transform runs over every trace at every epsilon).
        window_cat = _window_categories(trace, arrival_t, W, window)

        queue = 0.0
        next_arrival = 0
        calls: list[ToolCall] = []
        for w in range(W):
            # Bytes that have arrived by the end of window w. Anything still
            # outstanding at the last window arrives there: the horizon is public, so
            # holding late payload does not make the *schedule* data-dependent.
            t_end = (w + 1) * window
            while next_arrival < len(up) and (
                arrival_t[next_arrival] <= t_end or w == W - 1
            ):
                queue += float(up[next_arrival] + down[next_arrival])
                next_arrival += 1

            q_units = int(min(L, math.ceil(queue / self.quantum_b)))
            noisy = int(q_units + discrete_laplace(rng, scale))
            release_units = int(min(L, max(0, noisy)))
            release_b = float(release_units * self.quantum_b)
            if release_b > 0:
                calls.append(
                    ToolCall(
                        category=window_cat[w],
                        request_size=max(1.0, release_b * 0.4),
                        response_size=max(1.0, release_b * 0.6),
                        gap=(0.0 if not calls else window),
                    )
                )
            queue = max(0.0, queue - release_b)

        self.n_transformed += 1
        if queue > 0:
            # The horizon did not drain the payload. Flushing it is outside the
            # mechanism, so such a trace's release sequence is NOT covered by the
            # stated epsilon; count it so the runner can report (and the deployer can
            # raise W or cap_b until the count is zero).
            self.overflow_traces += 1
            calls.append(
                ToolCall(category="fetch", request_size=max(1.0, queue * 0.4),
                         response_size=max(1.0, queue * 0.6),
                         gap=(0.0 if not calls else window))
            )
        if not calls:
            # The mechanism released nothing and there was nothing to send: emit one
            # empty grid slot so the trace stays schema-valid (>=1 call).
            calls.append(ToolCall(category="fetch", request_size=1.0,
                                  response_size=1.0, gap=0.0))
        return Trace(
            calls=calls,
            label=trace.label,
            family=trace.family,
            metadata={**trace.metadata, "defense": self.name,
                      "epsilon_total": self.epsilon, "windows": W,
                      "adjacency": self.adjacency},
        )


def _window_categories(trace: Trace, arrival_t: np.ndarray, W: int,
                       window: float) -> list[str]:
    """Label each released window with the category of a real call inside it.

    The network vantage never reads the category; keeping a plausible one only makes
    the record schema-valid. A gateway shaper does not change what the *platform*
    dispatches, which is why this defense is scored at the network vantage.
    """
    fallback = trace.calls[-1].category
    cats: list[str] = [""] * W
    slot = np.clip((arrival_t / window).astype(int), 0, W - 1)
    for i, w in enumerate(slot):
        if not cats[w]:
            cats[w] = trace.calls[i].category
    for w in range(W):
        if not cats[w]:
            cats[w] = trace.calls[min(len(trace.calls) - 1, w)].category or fallback
    return cats
