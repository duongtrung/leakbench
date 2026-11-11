"""Seed control (paper §Impl: 'All randomness derives from a single logged seed').

The reference seed used throughout the paper is ``20270214``.
"""

from __future__ import annotations

import os
import random

import numpy as np

REFERENCE_SEED: int = 20270214


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and hash randomization for reproducibility.

    This seeds the *global* generators. Library code should prefer an explicit
    :func:`make_rng` instance so that concurrent components do not interfere; this
    function exists for top-level scripts that want a single global seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh, independent NumPy ``Generator`` for a given seed.

    Using an explicit generator (rather than the legacy global state) is what makes
    LeakBench deterministic under a fixed seed even when several stochastic
    components run in one process.
    """
    return np.random.default_rng(seed)
