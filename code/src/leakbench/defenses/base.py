"""Defense interface + registry + cost accounting.

A defense is a pure trace->trace transform plus a bandwidth/latency cost accounting.
Costs are measured against the undefended baseline on the *same* traces, so the
privacy-utility frontier (paper Table "defense", Fig. "tradeoff") is reported on a
common axis. Per the paper (§Defenses "Caveat"), only the *ordering* of costs is
claimed; absolute multipliers are setting-dependent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from leakbench.data.schemas import Trace, TraceDataset


@dataclass
class DefenseResult:
    """A defended dataset plus its measured overheads vs. the undefended input."""

    dataset: TraceDataset
    bandwidth_overhead: float   # fractional extra bytes, e.g. 0.20 = +20%
    latency_overhead: float     # fractional extra wall-time (from added gaps/calls)
    defense_name: str
    params: dict = field(default_factory=dict)


class Defense(ABC):
    """Abstract trace-level defense."""

    name: str = "defense"

    @abstractmethod
    def transform(self, trace: Trace, rng: np.random.Generator) -> Trace:
        """Return a defended copy of a single trace."""

    def params(self) -> dict:
        return {}

    def fit(self, dataset: TraceDataset, rng: np.random.Generator) -> None:
        """Optional training hook for *learned* defenses (e.g. generative shaping).

        Called once by :meth:`apply` before any trace is transformed. Stateless
        defenses ignore it. A learned shaper is fitted on the **defender's own**
        traffic distribution --- the deployment assumption is that an operator knows
        its own traffic, not the user's task; the fitted model must never read the
        trace label at transform time (see ``defenses/generative.py``).
        """
        return None

    # ------------------------------------------------------------- dataset API
    def apply(self, dataset: TraceDataset, seed: int = 20270214) -> DefenseResult:
        rng = np.random.default_rng(seed)
        self.fit(dataset, np.random.default_rng(seed + 977))
        defended: list[Trace] = []
        base_bytes = 0.0
        new_bytes = 0.0
        base_time = 0.0
        new_time = 0.0
        for t in dataset.traces:
            base_bytes += float(t.total_sizes().sum())
            base_time += float(t.gaps().sum())
            dt = self.transform(t, rng)
            new_bytes += float(dt.total_sizes().sum())
            new_time += float(dt.gaps().sum())
            defended.append(dt)
        bw = (new_bytes - base_bytes) / base_bytes if base_bytes > 0 else 0.0
        lat = (new_time - base_time) / base_time if base_time > 0 else 0.0
        ds = TraceDataset(
            traces=defended,
            taxonomy_name=dataset.taxonomy_name,
            taxonomy_hash=dataset.taxonomy_hash,
            source=dataset.source,
            metadata={**dataset.metadata, "defense": self.name},
        )
        return DefenseResult(
            dataset=ds,
            bandwidth_overhead=bw,
            latency_overhead=lat,
            defense_name=self.name,
            params=self.params(),
        )


# ---------------------------------------------------------------- registry
_REGISTRY: dict[str, type[Defense]] = {}


def register(cls: type[Defense]) -> type[Defense]:
    _REGISTRY[cls.name] = cls
    return cls


def get_defense(name: str, **kwargs) -> Defense:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown defense {name!r}; available: {sorted(_REGISTRY)}.")
    return _REGISTRY[name](**kwargs)


def list_defenses() -> list[str]:
    return sorted(_REGISTRY)
