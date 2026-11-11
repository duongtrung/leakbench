"""Prior attacks as projections of the unified observable (paper §Projections,
Table "projection"). Each deployed 2024-2026 attack reads a *sub-vector* of some
``O_v``; expressing them as projections ``pi_S(O_v)`` is what lets one benchmark
reproduce and compare them (§Eval "Case study: reproducing a deployed attack").

| Attack            | Vantage | Reads                    | Projection                 |
|-------------------|---------|--------------------------|----------------------------|
| Keylogger         | NET     | response token lengths   | pi_{s_down}(O_net)         |
| Whisper Leak      | NET     | pkt size + timing seq    | pi_{s,Delta}(O_net)        |
| AgentPrint        | NET     | burst size/count/timing  | O_net (full)               |
| Privacy side-ch.  | PLAT    | system-component signals | pi_{c,n}(O_plat)           |
| Cache side-ch.    | host    | token-position cache     | host, NOT in {O_v}         |

The cache side channel observes host micro-architectural state outside the
network/platform/provider model --- a documented reminder that the vantage
taxonomy is extensible (it is intentionally NOT implemented as an O_v projection).

Each projection returns a mask over the *full* feature vector of its base vantage,
so the estimator runs unchanged on the restricted feature set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from leakbench.observables.maps import (
    VANTAGE_DIMS,
    network_feature_slices,
    platform_feature_slices,
)


@dataclass(frozen=True)
class Projection:
    """A named feature-subset projection of a base vantage."""

    name: str
    base_vantage: str
    blocks: tuple[str, ...]   # named feature blocks of the base vantage to KEEP
    reference: str            # paper citation key

    def indices(self) -> np.ndarray:
        """Column indices (into the base vantage's feature vector) to keep."""
        slices = (
            network_feature_slices()
            if self.base_vantage == "network"
            else platform_feature_slices()
        )
        # Build over the full concatenated length.
        full_len = _base_length(self.base_vantage)
        keep = np.zeros(full_len, dtype=bool)
        for block in self.blocks:
            sl = slices[block]
            keep[sl] = True
        return np.where(keep)[0]

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.indices()]


def _base_length(vantage: str) -> int:
    """Width of the base vantage's feature vector.

    Taken from the maps' own derived dimensions, so a change to the bucket sets
    cannot silently desynchronise a projection mask from the map it slices.
    """
    if vantage not in VANTAGE_DIMS:
        raise ValueError(f"Unknown base vantage {vantage!r}; expected {list(VANTAGE_DIMS)}.")
    return VANTAGE_DIMS[vantage]


# The four in-model deployed attacks as projections (Table projection).
PROJECTIONS: dict[str, Projection] = {
    # Token-length keylogger: response sizes only. We approximate s_down with the
    # network volume + size histogram blocks (the network map does not separate
    # up/down per-call, so response-size structure lives in the size features).
    "keylogger": Projection(
        name="keylogger",
        base_vantage="network",
        blocks=("size_hist", "size_stats"),
        reference="weiss2024whatwasyourprompt",
    ),
    # Whisper Leak: size + timing series (everything except pure count/volume scalars
    # is size or timing here; we keep size and timing blocks).
    "whisper_leak": Projection(
        name="whisper_leak",
        base_vantage="network",
        blocks=("size_hist", "size_stats", "timing_hist", "timing_stats"),
        reference="microsoft2025whisperleak",
    ),
    # AgentPrint: essentially all of O_net.
    "agentprint": Projection(
        name="agentprint",
        base_vantage="network",
        blocks=("volume", "size_hist", "timing_hist", "size_stats", "timing_stats"),
        reference="zhang2025agentprint",
    ),
    # Privacy side channels: platform category + count signals (unigram + length).
    "privacy_sidechannel": Projection(
        name="privacy_sidechannel",
        base_vantage="platform",
        blocks=("length", "unigram"),
        reference="debenedetti2024sidechannels",
    ),
    # Size-only network projection (used on captured traffic in RQ8 to coincide with
    # the deployed size attacks).
    "size_only": Projection(
        name="size_only",
        base_vantage="network",
        blocks=("volume", "size_hist", "size_stats"),
        reference="microsoft2025whisperleak+weiss2024whatwasyourprompt",
    ),
    # Timing-only network projection (RQ8 contrast: size dominates timing).
    "timing_only": Projection(
        name="timing_only",
        base_vantage="network",
        blocks=("timing_hist", "timing_stats"),
        reference="—",
    ),
}


def project_features(X: np.ndarray, projection: str) -> np.ndarray:
    """Restrict a base-vantage feature matrix to a named projection's columns."""
    if projection not in PROJECTIONS:
        raise ValueError(f"Unknown projection {projection!r}; expected {list(PROJECTIONS)}.")
    return PROJECTIONS[projection].apply(X)


AblationBlock = Callable[[np.ndarray], np.ndarray]
