"""A tiny in-memory document corpus + SQLite fixture for the measured harness.

Everything here is generated deterministically in-process --- no downloads, no
external files. It exists only so the ``search`` and ``db_query`` tools do *real*
work (real substring scans, real SQL) and emit *true* byte sizes and timing.
"""

from __future__ import annotations

import sqlite3

import numpy as np

_WORDS = (
    "agent trace metadata privacy leakage vantage network platform provider tool "
    "call size timing padding cover shuffle taxonomy estimator advantage bayes "
    "mutual information bootstrap subgroup defense benchmark reproducible encrypted "
    "request response gap calendar email search fetch code database file report "
    "forecast reconcile schedule debug refactor wrangle market review fact check"
).split()


def build_corpus(n_docs: int = 200, seed: int = 20270214) -> list[str]:
    """Deterministic pseudo-document corpus (varied lengths so scans vary in cost)."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_docs):
        length = int(rng.integers(20, 200))
        docs.append(" ".join(rng.choice(_WORDS, size=length)))
    return docs


def build_sqlite(n_rows: int = 500, seed: int = 20270214) -> sqlite3.Connection:
    """A real in-memory SQLite table for the ``db_query`` tool."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE records (id INTEGER, category TEXT, value REAL, note TEXT)")
    rows = [
        (
            int(i),
            str(_WORDS[int(rng.integers(len(_WORDS)))]),
            float(rng.normal(100, 30)),
            " ".join(rng.choice(_WORDS, size=int(rng.integers(3, 12)))),
        )
        for i in range(n_rows)
    ]
    conn.executemany("INSERT INTO records VALUES (?,?,?,?)", rows)
    conn.commit()
    return conn
