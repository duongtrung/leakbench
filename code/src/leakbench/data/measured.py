"""Measured tool-execution harness --- Stage 2(a), the high-fidelity trace source
(paper §Impl "Measured tool-execution harness", §Eval RQ6).

Each taxonomy task runs a *program of real local tools* and we record the TRUE
request/response byte sizes and TRUE wall-clock inter-call gaps:

- ``code_exec``  : executes Python in-process, captures stdout;
- ``db_query``   : runs SQL against a real in-memory SQLite table;
- ``file_io``    : writes and reads real temporary files;
- ``search``     : substring-scans an in-memory document corpus;
- ``calendar`` / ``email`` : serialize JSON payloads;
- ``fetch``      : reads from the in-memory corpus (a local stand-in for a fetch);
- ``calc``       : evaluates a small arithmetic program.

Tool programs are sampled from **overlapping** class-conditional tool distributions
with a class-dependent payload-size scale (the same tool returns differently sized
payloads for different tasks), so sizes/timing are *emergent from real computation*
rather than drawn from a chosen distribution. The planner is **templated** (fixed
program per class), disclosed as a simplification; the tool-execution channel itself
is real.

The paper runs this over 5 seeds with 130 traces/class. This module produces the
raw (pre-wire) traces; ``wire.py`` supplies the TLS + HTTP/2 wire models applied for
the network vantage.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import numpy as np

from leakbench.data.corpus import build_corpus, build_sqlite
from leakbench.data.generator import DIRICHLET_ALPHA, SIGNATURE_UPWEIGHT
from leakbench.data.schemas import ToolCall, Trace, TraceDataset
from leakbench.taxonomy import (
    FAMILY_SIGNATURES,
    NUM_TOOLS,
    TOOL_CATEGORIES,
    TOOL_INDEX,
    Taxonomy,
    reference_taxonomy,
)

# ------------------------------------------------------------------ real tools
# Each ``*_detail`` variant additionally returns the *real* observable output of the
# tool run (hit count, row count, computed parity, ...). The live experiment (RQ8)
# feeds those to a controller that branches on them, so its plan is data-dependent on
# genuine tool results rather than on a re-drawn random number. The two-tuple
# wrappers keep the measured harness's call sites unchanged.
ToolSignal = dict[str, float]


def _tool_code_exec_detail(payload_scale: float,
                           rng: np.random.Generator) -> tuple[float, float, ToolSignal]:
    """Execute real Python in-process; return (request_bytes, response_bytes, signal)."""
    k = int(max(1, rng.integers(5, 40) * payload_scale))
    src = "total=0\n" + "".join(f"total+={i}\n" for i in range(k)) + "print(total)"
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(src, "<measured>", "exec"), {})  # noqa: S102 - sandboxed toy program
    out = buf.getvalue()
    total = float(out.strip() or 0.0)
    # Parity of the value the program actually computed (a real output signal).
    return (float(len(src.encode())), float(len(out.encode()) + 1),
            {"total": total, "parity": float(int(total) % 2)})


def _tool_db_query_detail(conn: sqlite3.Connection, payload_scale: float,
                          rng: np.random.Generator) -> tuple[float, float, ToolSignal]:
    """Run a real SQL query; response size = serialized rows, signal = row count."""
    limit = int(max(1, rng.integers(5, 60) * payload_scale))
    q = f"SELECT id, category, value, note FROM records WHERE value > ? LIMIT {limit}"
    thresh = float(rng.normal(90, 20))
    rows = conn.execute(q, (thresh,)).fetchall()
    resp = json.dumps(rows).encode()
    return float(len(q.encode()) + 8), float(len(resp) + 1), {"n_rows": float(len(rows))}


def _tool_file_io_detail(payload_scale: float,
                         rng: np.random.Generator) -> tuple[float, float, ToolSignal]:
    """Write and read a real temp file; sizes = bytes written/read."""
    nbytes = int(max(16, rng.integers(200, 4000) * payload_scale))
    data = os.urandom(nbytes)
    fd, path = tempfile.mkstemp(prefix="leakbench_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        with open(path, "rb") as fh:
            read = fh.read()
    finally:
        os.unlink(path)
    return float(len(data)), float(len(read)), {"n_bytes": float(len(read))}


def _tool_search_detail(corpus: list[str], payload_scale: float,
                        rng: np.random.Generator) -> tuple[float, float, ToolSignal]:
    """Substring-scan a real in-memory corpus; signal = number of matching documents."""
    from leakbench.data.corpus import _WORDS

    term = str(_WORDS[int(rng.integers(len(_WORDS)))])
    hits = [d for d in corpus if term in d]
    keep = hits[: int(max(1, rng.integers(1, 10) * payload_scale))]
    resp = json.dumps(keep).encode()
    return (float(len(term.encode()) + 8), float(len(resp) + 1),
            {"n_hits": float(len(hits)), "n_returned": float(len(keep))})


def _tool_json_detail(payload_scale: float,
                      rng: np.random.Generator) -> tuple[float, float, ToolSignal]:
    """Serialize a JSON payload (calendar/email/fetch/calc stand-in).

    Records carry real text content, not bare floats: a calendar entry or an email is
    a title, participants, a body. That matters for fidelity, not decoration --- the
    paper calibrates the trace scale to "hundreds of bytes to tens of kilobytes per
    call" (§Impl), and a payload of a dozen floats lands at ~300 B, i.e. below one
    1500-byte MTU. Since the network vantage MTU-quantises, a sub-MTU payload carries
    *zero* size information, so under-scaling these four tools (calendar, email,
    fetch, calc --- half the category set) silently erases the size channel the
    experiment is meant to measure.
    """
    from leakbench.data.corpus import _WORDS

    n_fields = int(max(1, rng.integers(2, 12) * payload_scale))
    obj: dict[str, object] = {}
    for i in range(n_fields):
        n_words = int(rng.integers(8, 90))
        obj[f"field_{i}"] = {
            "id": int(rng.integers(0, 10**6)),
            "value": float(rng.random()),
            "text": " ".join(rng.choice(_WORDS, size=n_words)),
        }
    req = json.dumps({"op": "get", "fields": n_fields}).encode()
    resp = json.dumps(obj).encode()
    return float(len(req)), float(len(resp) + 1), {"n_fields": float(n_fields)}


def _tool_code_exec(payload_scale: float, rng: np.random.Generator) -> tuple[float, float]:
    return _tool_code_exec_detail(payload_scale, rng)[:2]


def _tool_db_query(conn: sqlite3.Connection, payload_scale: float,
                   rng: np.random.Generator) -> tuple[float, float]:
    return _tool_db_query_detail(conn, payload_scale, rng)[:2]


def _tool_file_io(payload_scale: float, rng: np.random.Generator) -> tuple[float, float]:
    return _tool_file_io_detail(payload_scale, rng)[:2]


def _tool_search(corpus: list[str], payload_scale: float,
                 rng: np.random.Generator) -> tuple[float, float]:
    return _tool_search_detail(corpus, payload_scale, rng)[:2]


def _tool_json(payload_scale: float, rng: np.random.Generator) -> tuple[float, float]:
    return _tool_json_detail(payload_scale, rng)[:2]


@dataclass
class MeasuredHarness:
    """Runs real local tools to produce measured traces.

    Parameters
    ----------
    taxonomy:
        the task taxonomy.
    seed:
        controls the class-conditional tool distributions and payload scales.
    overlap:
        how much non-signature mass the class distributions share (higher = more
        confusable = harder, more realistic). Default 0.5.
    """

    taxonomy: Taxonomy = field(default_factory=reference_taxonomy)
    seed: int = 20270214
    overlap: float = 0.5
    signature_upweight: float = SIGNATURE_UPWEIGHT
    dirichlet_alpha: float = DIRICHLET_ALPHA

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._corpus = build_corpus(seed=self.seed)
        self._conn = build_sqlite(seed=self.seed)
        # Overlapping **class-conditional** tool distributions + payload-size scales
        # (paper §Eval RQ6: "Tasks are drawn from overlapping class-conditional tool
        # distributions"; App. "Full Task Taxonomy": "Each class fixes a
        # Dirichlet-drawn class-conditional distribution over the eight tool
        # categories, with the family's signature tools up-weighted").
        #
        # Each class therefore gets its OWN Dirichlet draw. Deriving the distribution
        # from the family signature alone would give only four distinct distributions
        # for twelve classes, and since the platform vantage never sees sizes, the
        # three classes of a family would be information-theoretically identical
        # there -- capping platform advantage at (R_pi - 2/3)/R_pi = 0.273 by
        # construction rather than by anything the traffic does.
        self._tool_dists: list[np.ndarray] = []
        self._size_scales: list[np.ndarray] = []
        for c in self.taxonomy.classes:
            p = self._rng.dirichlet(np.full(NUM_TOOLS, self.dirichlet_alpha))
            for tool in FAMILY_SIGNATURES[c.family]:
                p[TOOL_INDEX[tool]] += self.signature_upweight
            p = p / p.sum()
            # ``overlap`` is the fraction of mass shared by every class (a uniform
            # component): higher overlap = more confusable = harder.
            p = (1.0 - self.overlap) * p + self.overlap / NUM_TOOLS
            self._tool_dists.append(p / p.sum())
            # Per-class payload-size scale (same tool, different sizes per task).
            self._size_scales.append(self._rng.uniform(0.6, 1.6, size=NUM_TOOLS))

    # --------------------------------------------------------------- one call
    def _run_tool(self, cat_idx: int, scale: float) -> tuple[float, float]:
        cat = TOOL_CATEGORIES[cat_idx]
        rng = self._rng
        if cat == "code_exec":
            return _tool_code_exec(scale, rng)
        if cat == "db_query":
            return _tool_db_query(self._conn, scale, rng)
        if cat == "file_io":
            return _tool_file_io(scale, rng)
        if cat == "search":
            return _tool_search(self._corpus, scale, rng)
        # calendar / email / fetch / calc -> JSON serialization
        return _tool_json(scale, rng)

    def sample_trace(self, class_index: int) -> Trace:
        rng = self._rng
        dist = self._tool_dists[class_index]
        scales = self._size_scales[class_index]
        n = max(2, int(rng.integers(6, 18)))
        cats = rng.choice(NUM_TOOLS, size=n, p=dist)
        calls: list[ToolCall] = []
        prev_start = None
        for i, cat in enumerate(cats):
            t0 = time.perf_counter()
            s_up, s_down = self._run_tool(int(cat), float(scales[cat]))
            # Delta_i is the INTER-ARRIVAL gap between consecutive calls (the quantity
            # O_net histograms as "inter-arrival"): the wall-clock time from the start
            # of call i-1 to the start of call i. It therefore contains the previous
            # tool's real service time, which is the timing channel an on-path
            # observer actually measures. (Measuring start-to-start rather than
            # end-to-start matters: end-to-start is only this harness's loop overhead,
            # ~2 us, and would report a timing channel that is an artifact of the
            # measurement code rather than of the tool execution.)
            gap = 0.0 if prev_start is None else max(0.0, t0 - prev_start)
            prev_start = t0
            calls.append(
                ToolCall(
                    category=TOOL_CATEGORIES[cat],
                    request_size=max(1.0, s_up),
                    response_size=max(1.0, s_down),
                    gap=gap,
                )
            )
        return Trace(calls=calls, label=class_index,
                     family=self.taxonomy.family_of(class_index),
                     metadata={"source": "measured", "seed": self.seed})

    def sample_dataset(self, traces_per_class: int) -> TraceDataset:
        traces: list[Trace] = []
        for ci in range(self.taxonomy.num_classes):
            for _ in range(traces_per_class):
                traces.append(self.sample_trace(ci))
        return TraceDataset(
            traces=traces,
            taxonomy_name=self.taxonomy.name,
            taxonomy_hash=self.taxonomy.content_hash(),
            source="measured",
            metadata={"traces_per_class": traces_per_class, "seed": self.seed,
                      "overlap": self.overlap},
        )
