"""Task taxonomy T --- Stage 1 of the LeakBench harness.

Implements design principle **P1 (fix the secret distribution)** from the paper:
comparable leakage numbers require a shared task taxonomy with a declared prior,
because a leakage value is meaningless without knowing ``H(C)``.

Reference taxonomy ``tax-v1/12c-4f`` (paper Table "Reference taxonomy"): 12 classes
in 4 families, 8 tool categories, uniform prior => ``H(C) = log2(12) = 3.585`` bits.
The taxonomy is versioned by a content hash so results are keyed by (taxonomy hash),
and a later, larger taxonomy does not silently invalidate comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# The eight tool categories (paper §Impl / taxonomy table). Order is canonical and
# used everywhere feature vectors index categories.
TOOL_CATEGORIES: tuple[str, ...] = (
    "search",
    "fetch",
    "code_exec",
    "file_io",
    "calendar",
    "email",
    "db_query",
    "calc",
)
NUM_TOOLS: int = len(TOOL_CATEGORIES)
TOOL_INDEX: dict[str, int] = {name: i for i, name in enumerate(TOOL_CATEGORIES)}

# Signature tools per family (paper taxonomy table). These are up-weighted by the
# generator (App. "Calibrated Generator", default +0.9).
FAMILY_SIGNATURES: dict[str, tuple[str, ...]] = {
    "research": ("search", "fetch"),
    "coding": ("code_exec", "file_io"),
    "personal": ("calendar", "email"),
    "ops": ("db_query", "calc"),
}


@dataclass(frozen=True)
class TaskClass:
    """A single task class (a value of the secret ``C``)."""

    index: int
    name: str
    family: str

    @property
    def signature_tools(self) -> tuple[str, ...]:
        return FAMILY_SIGNATURES[self.family]


@dataclass(frozen=True)
class Taxonomy:
    """A versioned task taxonomy: the set of classes/secrets and their prior.

    Parameters
    ----------
    classes:
        ordered task classes.
    prior:
        prior ``pi`` over classes. Defaults to uniform (maximizes ``H(C)``).
    name:
        human-readable version label (e.g. ``tax-v1/12c-4f``).
    """

    classes: tuple[TaskClass, ...]
    prior: np.ndarray
    name: str = "tax-v1/12c-4f"

    def __post_init__(self) -> None:
        if len(self.classes) == 0:
            raise ValueError("Taxonomy must have at least one class.")
        if self.prior.shape != (len(self.classes),):
            raise ValueError("Prior length must equal the number of classes.")
        if not np.isclose(self.prior.sum(), 1.0):
            raise ValueError("Prior must sum to 1.")
        if np.any(self.prior < 0):
            raise ValueError("Prior must be non-negative.")

    # ------------------------------------------------------------------ sizes
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for c in self.classes:
            if c.family not in seen:
                seen.append(c.family)
        return tuple(seen)

    # ------------------------------------------------------------ information
    def entropy_bits(self) -> float:
        """Secret entropy ``H(C)`` in bits (paper §Threat: ``log2|T|`` for uniform)."""
        p = self.prior[self.prior > 0]
        return float(-np.sum(p * np.log2(p)))

    def guessing_risk(self) -> float:
        """Random-guessing risk ``R_pi = 1 - max_t pi(t)`` (Def. Bayes risk)."""
        return float(1.0 - self.prior.max())

    def family_of(self, class_index: int) -> str:
        return self.classes[class_index].family

    def family_labels(self, class_indices: Sequence[int]) -> np.ndarray:
        """Map class indices to a coarse family index (for the 4-way secret)."""
        fam_to_idx = {f: i for i, f in enumerate(self.families)}
        return np.array([fam_to_idx[self.classes[c].family] for c in class_indices])

    # --------------------------------------------------------------- hashing
    def content_hash(self) -> str:
        """Deterministic content hash so results are keyed by taxonomy version.

        Excludes the prior *values* being a numpy array by serializing a rounded
        list; two taxonomies with the same classes and (rounded) prior hash equal.
        """
        payload = {
            "classes": [(c.index, c.name, c.family) for c in self.classes],
            "prior": [round(float(x), 8) for x in self.prior],
            "tools": list(TOOL_CATEGORIES),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


# ------------------------------------------------------------- reference taxonomy
_REFERENCE_CLASSES: tuple[tuple[str, str], ...] = (
    ("literature_review", "research"),
    ("fact_check_claim", "research"),
    ("market_scan", "research"),
    ("debug_failing_test", "coding"),
    ("refactor_module", "coding"),
    ("data_wrangling", "coding"),
    ("trip_planning", "personal"),
    ("meeting_scheduling", "personal"),
    ("event_rsvp", "personal"),
    ("status_reporting", "ops"),
    ("reconciliation", "ops"),
    ("forecast", "ops"),
)


def reference_taxonomy(prior: np.ndarray | None = None) -> Taxonomy:
    """The paper's reference taxonomy ``tax-v1/12c-4f`` (12 classes, 4 families).

    ``H(C) = log2(12) ≈ 3.585`` bits under the default uniform prior.
    """
    classes = tuple(
        TaskClass(index=i, name=name, family=fam)
        for i, (name, fam) in enumerate(_REFERENCE_CLASSES)
    )
    if prior is None:
        prior = np.full(len(classes), 1.0 / len(classes))
    return Taxonomy(classes=classes, prior=prior, name="tax-v1/12c-4f")


def scaled_taxonomy(num_classes: int, prior: np.ndarray | None = None) -> Taxonomy:
    """A taxonomy with ``num_classes`` classes for the RQ7 size-sweep
    (``|T| ∈ {8,12,16,24}``).

    Classes are assigned round-robin to the four families so the signature-tool
    structure is preserved as the taxonomy scales.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1.")
    fams = tuple(FAMILY_SIGNATURES.keys())
    classes = tuple(
        TaskClass(index=i, name=f"class_{i:02d}", family=fams[i % len(fams)])
        for i in range(num_classes)
    )
    if prior is None:
        prior = np.full(num_classes, 1.0 / num_classes)
    return Taxonomy(classes=classes, prior=prior, name=f"tax-v1/{num_classes}c-4f")


def zipf_prior(num_classes: int, s: float = 1.0) -> np.ndarray:
    """A skewed (Zipf-like) prior for the non-uniform-prior sensitivity check
    (paper §Eval "Non-uniform prior")."""
    ranks = np.arange(1, num_classes + 1, dtype=float)
    w = 1.0 / np.power(ranks, s)
    return w / w.sum()
