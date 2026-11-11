"""Trace data structures --- the central object of Def. "Agent trace".

A trace is a finite sequence of tool calls

    rho = < (c_i, s_up_i, s_down_i, Delta_i) >_{i=1..n}

with tool category ``c_i``, request/response byte sizes ``s_up_i, s_down_i > 0``,
and inter-call gap ``Delta_i >= 0``. Traces carry the task label ``t`` (the secret)
and metadata (family, source, seed, ...) used for subgroup analysis and provenance.

These schemas are the record format shared by *every* trace source (generator,
measured harness, live adapter), so an estimator trained on one runs unchanged on
another (paper §Design Stage 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np

from leakbench.taxonomy import NUM_TOOLS, TOOL_CATEGORIES, TOOL_INDEX


@dataclass(frozen=True)
class ToolCall:
    """One tool call = one element of a trace (Def. "Agent trace")."""

    category: str            # c_i in TOOL_CATEGORIES
    request_size: float      # s_up_i, bytes, > 0
    response_size: float     # s_down_i, bytes, > 0
    gap: float               # Delta_i, seconds, >= 0 (0 for the first call)

    def __post_init__(self) -> None:
        if self.category not in TOOL_INDEX:
            raise ValueError(
                f"Unknown tool category {self.category!r}; expected one of {TOOL_CATEGORIES}."
            )
        if not (self.request_size > 0 and self.response_size > 0):
            raise ValueError("request_size and response_size must be > 0 bytes.")
        if self.gap < 0:
            raise ValueError("gap (inter-call delay) must be >= 0.")

    @property
    def category_index(self) -> int:
        return TOOL_INDEX[self.category]

    @property
    def total_size(self) -> float:
        return self.request_size + self.response_size


@dataclass
class Trace:
    """An agent tool-use trace for a single task, plus label and metadata."""

    calls: list[ToolCall]
    label: int                              # task class index (the secret C)
    family: str | None = None               # coarse-secret / subgroup key
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.calls)

    def __iter__(self) -> Iterator[ToolCall]:
        return iter(self.calls)

    # -------------------------------------------------------------- accessors
    @property
    def n(self) -> int:
        """Trace length ``n`` (number of calls)."""
        return len(self.calls)

    def categories(self) -> np.ndarray:
        return np.array([c.category_index for c in self.calls], dtype=int)

    def request_sizes(self) -> np.ndarray:
        return np.array([c.request_size for c in self.calls], dtype=float)

    def response_sizes(self) -> np.ndarray:
        return np.array([c.response_size for c in self.calls], dtype=float)

    def total_sizes(self) -> np.ndarray:
        return np.array([c.total_size for c in self.calls], dtype=float)

    def gaps(self) -> np.ndarray:
        return np.array([c.gap for c in self.calls], dtype=float)

    # ----------------------------------------------------------------- utils
    def category_counts(self) -> np.ndarray:
        """Per-category unigram counts (length NUM_TOOLS)."""
        counts = np.zeros(NUM_TOOLS, dtype=float)
        for c in self.calls:
            counts[c.category_index] += 1
        return counts

    def to_record(self) -> dict[str, Any]:
        """Serializable record (used by IO / adapters)."""
        return {
            "label": int(self.label),
            "family": self.family,
            "metadata": dict(self.metadata),
            "calls": [
                {
                    "category": c.category,
                    "request_size": float(c.request_size),
                    "response_size": float(c.response_size),
                    "gap": float(c.gap),
                }
                for c in self.calls
            ],
        }

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "Trace":
        calls = [
            ToolCall(
                category=c["category"],
                request_size=float(c["request_size"]),
                response_size=float(c["response_size"]),
                gap=float(c["gap"]),
            )
            for c in rec["calls"]
        ]
        return cls(
            calls=calls,
            label=int(rec["label"]),
            family=rec.get("family"),
            metadata=dict(rec.get("metadata", {})),
        )


@dataclass
class TraceDataset:
    """A labeled collection of traces + the taxonomy they were drawn against."""

    traces: list[Trace]
    taxonomy_name: str
    taxonomy_hash: str
    source: str = "unknown"           # "generator" | "measured" | "live" | "adapter"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self) -> Iterator[Trace]:
        return iter(self.traces)

    def labels(self) -> np.ndarray:
        return np.array([t.label for t in self.traces], dtype=int)

    def families(self) -> list[str | None]:
        return [t.family for t in self.traces]

    def subgroup_length_band(self, split: str = "median") -> np.ndarray:
        """Binary trace-length subgroup: 0 = short, 1 = long (median split).

        Used for the disparate-vulnerability analysis (paper §Eval RQ3).
        """
        lengths = np.array([t.n for t in self.traces])
        threshold = np.median(lengths)
        return (lengths > threshold).astype(int)

    def filter_by(self, indices: Iterable[int]) -> "TraceDataset":
        idx = list(indices)
        return TraceDataset(
            traces=[self.traces[i] for i in idx],
            taxonomy_name=self.taxonomy_name,
            taxonomy_hash=self.taxonomy_hash,
            source=self.source,
            metadata=dict(self.metadata),
        )
