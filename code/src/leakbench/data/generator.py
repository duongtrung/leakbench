"""Calibrated trace generator ``G`` --- Stage 2(b), the high-throughput,
fully-reproducible trace source (paper §Impl "The calibrated generator", App.
"Calibrated Generator: Formalism").

Generative model (verbatim from the paper), per class ``t`` with family ``f(t)``
and signature set ``S_{f(t)}``:

    p_t ~ Dir(0.6 * 1_8),  p_t[tau] += 0.9  for tau in S_{f(t)},  renormalize
    lambda_t ~ U[6, 18]
    mu_tau ~ U[6, 9],  nu_tau ~ U[-1.2, 0.8]     (per category tau)

Trace sampling for class ``t``:

    n ~ Poisson(lambda_t),  n <- max(2, n)
    c_i ~ Cat(p_t)
    s_up_i   ~ LogNormal(0.6 * mu_{c_i}, 0.5^2)
    s_down_i ~ LogNormal(mu_{c_i},       0.6^2)
    Delta_i  ~ LogNormal(nu_{c_i},       0.5^2)

The signature up-weight (default 0.9) is the single knob that tunes ground-truth
separability (paper Fig. separability). NumPy's ``lognormal(mean, sigma)`` takes the
standard deviation of the underlying normal, so we pass ``sigma = 0.5/0.6/0.5``
(the paper writes variances ``0.5^2`` etc.) --- see math_to_code_audit.md, E9.

This generator is a *stand-in* for measured traces, calibrated to plausible orders
of magnitude, NOT a claim about any deployed system (paper §Impl, §Limitations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leakbench.data.schemas import Trace, TraceDataset
from leakbench.taxonomy import (
    NUM_TOOLS,
    TOOL_CATEGORIES,
    TOOL_INDEX,
    Taxonomy,
    reference_taxonomy,
)

# Default generator hyperparameters (App. "Calibrated Generator").
DIRICHLET_ALPHA = 0.6
SIGNATURE_UPWEIGHT = 0.9
LAMBDA_RANGE = (6.0, 18.0)
MU_RANGE = (6.0, 9.0)
NU_RANGE = (-1.2, 0.8)
SIGMA_REQUEST = 0.5          # std of underlying normal for s_up
SIGMA_RESPONSE = 0.6         # std of underlying normal for s_down
SIGMA_GAP = 0.5              # std of underlying normal for Delta
REQUEST_MU_SCALE = 0.6       # s_up uses 0.6 * mu (requests smaller than responses)
MIN_TRACE_LEN = 2


@dataclass
class ClassProfile:
    """The once-drawn parameters that define a class-conditional trace distribution."""

    tool_dist: np.ndarray            # p_t, length NUM_TOOLS, sums to 1
    length_rate: float               # lambda_t
    size_mu: np.ndarray              # mu_tau, per category (log-space size mean)
    gap_nu: np.ndarray               # nu_tau, per category (log-space gap mean)


@dataclass
class CalibratedGenerator:
    """The calibrated generator ``G``.

    Class profiles are drawn *once* (deterministically from ``seed``) and reused for
    all sampled traces, matching the paper's "draws, once" semantics. Re-drawing the
    profiles (different seed) is the generator-variance check of paper §Eval.

    Two knobs change ground-truth separability, and they are **not** interchangeable:

    - ``dirichlet_alpha`` controls how distinct the per-class tool distributions are.
      Small alpha gives sparse, sharply different profiles (separable classes); large
      alpha drives every class toward uniform (confusable). This is the knob that
      monotonically controls separability, and it is the one to sweep when validating
      that the estimator tracks ground truth.
    - ``signature_upweight`` adds mass to the *family's* signature tools. Because three
      classes share a family, raising it makes same-family classes **more alike**, so
      advantage *falls*. It controls family structure, not separability.
    """

    taxonomy: Taxonomy = field(default_factory=reference_taxonomy)
    seed: int = 20270214
    signature_upweight: float = SIGNATURE_UPWEIGHT
    dirichlet_alpha: float = DIRICHLET_ALPHA
    _profiles: list[ClassProfile] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._profiles = [self._draw_profile(c.signature_tools) for c in self.taxonomy.classes]

    # ------------------------------------------------------------- profiles
    def _draw_profile(self, signature_tools: tuple[str, ...]) -> ClassProfile:
        rng = self._rng
        p = rng.dirichlet(np.full(NUM_TOOLS, self.dirichlet_alpha))
        for tool in signature_tools:
            p[TOOL_INDEX[tool]] += self.signature_upweight
        p = p / p.sum()
        lam = rng.uniform(*LAMBDA_RANGE)
        mu = rng.uniform(MU_RANGE[0], MU_RANGE[1], size=NUM_TOOLS)
        nu = rng.uniform(NU_RANGE[0], NU_RANGE[1], size=NUM_TOOLS)
        return ClassProfile(tool_dist=p, length_rate=lam, size_mu=mu, gap_nu=nu)

    @property
    def profiles(self) -> list[ClassProfile]:
        return self._profiles

    # ------------------------------------------------------------- sampling
    def sample_trace(self, class_index: int) -> Trace:
        """Sample a single trace for a given task class."""
        rng = self._rng
        prof = self._profiles[class_index]
        n = int(rng.poisson(prof.length_rate))
        n = max(MIN_TRACE_LEN, n)
        cats = rng.choice(NUM_TOOLS, size=n, p=prof.tool_dist)

        calls = []
        for i, cat in enumerate(cats):
            mu = prof.size_mu[cat]
            nu = prof.gap_nu[cat]
            s_up = float(np.ceil(rng.lognormal(REQUEST_MU_SCALE * mu, SIGMA_REQUEST)))
            s_down = float(np.ceil(rng.lognormal(mu, SIGMA_RESPONSE)))
            # The first call has no preceding call -> gap 0 (Delta semantics).
            gap = 0.0 if i == 0 else float(rng.lognormal(nu, SIGMA_GAP))
            calls.append(
                _make_call(TOOL_CATEGORIES[cat], max(1.0, s_up), max(1.0, s_down), gap)
            )
        fam = self.taxonomy.family_of(class_index)
        return Trace(calls=calls, label=class_index, family=fam,
                     metadata={"source": "generator"})

    def sample_dataset(self, traces_per_class: int) -> TraceDataset:
        """Sample a balanced dataset with ``traces_per_class`` traces per class."""
        traces: list[Trace] = []
        for ci in range(self.taxonomy.num_classes):
            for _ in range(traces_per_class):
                traces.append(self.sample_trace(ci))
        return TraceDataset(
            traces=traces,
            taxonomy_name=self.taxonomy.name,
            taxonomy_hash=self.taxonomy.content_hash(),
            source="generator",
            metadata={
                "traces_per_class": traces_per_class,
                "seed": self.seed,
                "signature_upweight": self.signature_upweight,
            },
        )


def _make_call(category: str, s_up: float, s_down: float, gap: float):
    from leakbench.data.schemas import ToolCall

    return ToolCall(category=category, request_size=s_up, response_size=s_down, gap=gap)


@dataclass
class OtherTrafficGenerator:
    """Out-of-taxonomy "other" traffic for the open-world protocol (paper Table
    "openworld": *random tool mixes, no signature*).

    Deliberately **not** a taxonomy class: the tool distribution is drawn from a flat
    Dirichlet with no signature up-weight, lengths span a wider range, and the size /
    gap parameters are re-drawn per trace rather than fixed per class. This models
    unseen activity the attacker never trained on, which is what makes the open-world
    base-rate effect real rather than a relabeling of known classes.

    Traces are labelled :data:`leakbench.metrics.openworld.OTHER_LABEL` (-1).
    """

    seed: int = 20270214
    length_range: tuple[int, int] = (2, 30)
    dirichlet_alpha: float = DIRICHLET_ALPHA

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample_trace(self) -> Trace:
        from leakbench.metrics.openworld import OTHER_LABEL

        rng = self._rng
        p = rng.dirichlet(np.full(NUM_TOOLS, self.dirichlet_alpha))   # no signature up-weight
        n = int(rng.integers(self.length_range[0], self.length_range[1] + 1))
        mu = rng.uniform(MU_RANGE[0], MU_RANGE[1], size=NUM_TOOLS)
        nu = rng.uniform(NU_RANGE[0], NU_RANGE[1], size=NUM_TOOLS)
        cats = rng.choice(NUM_TOOLS, size=n, p=p)
        calls = []
        for i, cat in enumerate(cats):
            s_up = float(np.ceil(rng.lognormal(REQUEST_MU_SCALE * mu[cat], SIGMA_REQUEST)))
            s_down = float(np.ceil(rng.lognormal(mu[cat], SIGMA_RESPONSE)))
            gap = 0.0 if i == 0 else float(rng.lognormal(nu[cat], SIGMA_GAP))
            calls.append(_make_call(TOOL_CATEGORIES[cat], max(1.0, s_up), max(1.0, s_down), gap))
        return Trace(calls=calls, label=OTHER_LABEL, family=None,
                     metadata={"source": "generator_other", "out_of_taxonomy": True})

    def sample_traces(self, n_traces: int) -> list[Trace]:
        return [self.sample_trace() for _ in range(n_traces)]
