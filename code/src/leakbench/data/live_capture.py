"""Live experiment with captured traffic --- RQ8 (paper §Eval "Live experiment with
captured traffic").

The paper's decisive check: run a real TLS tool-server, drive it with an MCP-style
JSON-over-TLS adapter on one persistent connection per session, **capture** the
encrypted wire records with a real TCP capture proxy (so TLS record framing and TCP
coalescing emerge from the kernel stack, not from a model), and use a **stochastic,
data-dependent planner** (a ReAct-style controller that branches on real intermediate
tool outputs).

What this module actually runs, end to end:

- :class:`_ToolServer` --- a real backend server executing the genuine tools of
  :mod:`leakbench.data.measured` (in-process Python exec, in-memory SQLite, real temp
  files, corpus search, JSON payloads), wrapped in **real TLS** when certificate
  material can be provisioned locally (:mod:`leakbench.data.tlsutil`).
- :class:`_CaptureProxy` --- a real TCP proxy sitting between client and server. It
  relays bytes and records the size of every kernel-delivered chunk. Under TLS these
  are **ciphertext** sizes: the proxy is on-path and, like any on-path observer,
  cannot decrypt. These captured sizes ARE the network observable.
- :class:`StochasticReActPlanner` --- a ReAct-style controller whose branches read the
  **real** tool outputs returned by the server (documents actually matched by the
  corpus search, rows actually returned by SQLite, the parity of the value the
  executed Python actually computed), so plan length and composition are functions of
  genuine tool results, not of a re-drawn random number.
- :class:`LLMPlanner` --- the same ``(tool, scale)`` step interface for a *neural*
  controller. It requires a user-supplied completion callable (an LLM endpoint);
  without one it raises, and never silently degrades to the programmatic planner.

CAPTURE MODES (recorded verbatim in ``trace.metadata['capture_mode']`` and in the
dataset metadata --- a run never claims a mode it did not achieve):

- ``tls_captured_proxy``: real TLS handshake + records, sizes captured off the proxy
  socket. The default when cert material can be produced.
- ``tcp_captured_proxy``: TLS unavailable (no ``openssl``/``cryptography``); plaintext
  TCP captured off the proxy, with TLS record framing applied as a *model* on top of
  the captured chunk sizes.

The **link** is recorded separately as ``loopback`` or ``remote``, resolved from the
server address rather than assumed: pointing ``server_host`` at 127.0.0.1 still says
``loopback``.

REMAINING DOCUMENTED GAPS (paper §Limitations):
  * the link is **loopback**, not WAN (needs two hosts; see
    ``docs/ec2_experiment_guide.md``);
  * the default planner is **programmatic**, not a neural LLM (an LLM endpoint is an
    external dependency; :class:`LLMPlanner` is the wired extension point).

STATUS: implemented (real sockets, real TLS when available, captured wire) / gaps:
WAN link, neural planner. No default pytest opens sockets for long; a tiny smoke run
is available via ``scripts/run_live_capture.py`` and a guarded, minimal test.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import sqlite3
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator, Protocol

import numpy as np

from leakbench.data.corpus import build_corpus, build_sqlite
from leakbench.data.measured import (
    _tool_code_exec_detail,
    _tool_db_query_detail,
    _tool_file_io_detail,
    _tool_json_detail,
    _tool_search_detail,
)
from leakbench.data.schemas import ToolCall, Trace, TraceDataset
from leakbench.data.tlsutil import (
    TLSMaterial,
    client_context,
    client_context_from_ca,
    generate_self_signed,
    server_context,
)
from leakbench.data.wire import tls_frame_size
from leakbench.taxonomy import TOOL_CATEGORIES, Taxonomy, reference_taxonomy
from leakbench.utils.logging import get_logger

logger = get_logger(__name__)

# RQ8 uses 6 overlapping, multi-task classes.
LIVE_CLASSES = (
    "literature_review",
    "trip_planning",
    "debug_failing_test",
    "status_reporting",
    "market_scan",
    "meeting_scheduling",
)

# The capture mode names the transport that was captured, NOT the link; the link
# ("loopback" / "remote") is recorded separately, because claiming "loopback" on a
# two-host run (or the reverse) would misdescribe the measurement.
CAPTURE_MODE_TLS = "tls_captured_proxy"
CAPTURE_MODE_TCP = "tcp_captured_proxy"

_RECV_CHUNK = 65536
_DRAIN_TIMEOUT_S = 0.25
# Per-recv deadline inside a session. Generous enough for a WAN round trip, short
# enough that a hung server fails the run instead of blocking it forever.
_SESSION_TIMEOUT_S = 30.0


# ---------------------------------------------------------------- capture proxy
@dataclass
class _CaptureRecord:
    """One captured wire event: direction and byte size of a kernel-delivered chunk."""

    direction: str   # "up" (client->server) or "down" (server->client)
    size: int
    t: float


class _CaptureProxy:
    """A real TCP proxy that relays bytes and records every chunk it carries.

    This is the paper's "capture proxy": it is on-path between the MCP-style client
    and the tool server. Under TLS it sees only ciphertext, exactly like the network
    adversary it stands in for. Sizes are what the kernel delivered, so TLS record
    framing and TCP coalescing are *observed*, not modeled.
    """

    def __init__(self, upstream_host: str, upstream_port: int,
                 bind_host: str = "127.0.0.1", bind_port: int = 0) -> None:
        self._upstream = (upstream_host, upstream_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind_host, bind_port))
        self._sock.listen(8)
        self.host: str = bind_host
        self.port: int = self._sock.getsockname()[1]
        self._logs: dict[tuple[str, int], list[_CaptureRecord]] = {}
        self._lock = threading.Lock()
        self._running = True
        self._threads: list[threading.Thread] = []
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._accept_thread.start()

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        for th in self._threads:
            th.join(timeout=1.0)

    # ------------------------------------------------------------------ capture
    def log_for(self, client_addr: tuple[str, int], timeout: float = 5.0) -> list[_CaptureRecord]:
        """Return the (live, append-only) capture log for one client connection."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                log = self._logs.get(client_addr)
            if log is not None:
                return log
            time.sleep(0.001)
        raise TimeoutError(f"capture proxy never saw a connection from {client_addr}")

    def snapshot(self, log: list[_CaptureRecord], start: int) -> list[_CaptureRecord]:
        with self._lock:
            return list(log[start:])

    def size(self, log: list[_CaptureRecord]) -> int:
        with self._lock:
            return len(log)

    # ------------------------------------------------------------------ plumbing
    def _accept_loop(self) -> None:
        while self._running:
            try:
                client, addr = self._sock.accept()
            except OSError:
                break
            try:
                upstream = socket.create_connection(self._upstream, timeout=10)
            except OSError:
                client.close()
                continue
            # Clear the connect timeout: it would otherwise persist as a *read*
            # timeout on the relay, so any quiet period longer than it tears the
            # session down. A programmatic planner never pauses that long, but a
            # neural controller thinks for seconds between calls and a WAN link adds
            # more, so leaving it set makes exactly the two runs we care about fail
            # with a bare "connection reset".
            upstream.settimeout(None)
            client.settimeout(None)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log: list[_CaptureRecord] = []
            with self._lock:
                self._logs[addr] = log
            for src, dst, direction in ((client, upstream, "up"), (upstream, client, "down")):
                th = threading.Thread(
                    target=self._relay, args=(src, dst, direction, log), daemon=True
                )
                th.start()
                self._threads.append(th)

    def _relay(self, src: socket.socket, dst: socket.socket, direction: str,
               log: list[_CaptureRecord]) -> None:
        try:
            while True:
                data = src.recv(_RECV_CHUNK)
                if not data:
                    break
                # Record BEFORE forwarding, so a chunk is always logged no later than
                # the moment the peer can observe it.
                with self._lock:
                    log.append(_CaptureRecord(direction, len(data), time.perf_counter()))
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass


# ------------------------------------------------------------------- tool server
class _ToolServer:
    """A real backend server that executes tools, optionally over real TLS.

    Protocol: newline-delimited JSON. Request ``{"tool": name, "scale": s}`` ->
    response JSON carrying the tool's *real* output signals plus a payload whose
    serialized length is the true response size.
    """

    def __init__(self, seed: int = 20270214, tls: TLSMaterial | None = None,
                 bind_host: str = "127.0.0.1", bind_port: int = 0) -> None:
        self._seed = seed
        # Backends (corpus, SQLite, RNG) are created *inside* the serving thread in
        # _serve(), because a SQLite connection may not be shared across threads.
        self._corpus: list[str] = []
        self._conn: sqlite3.Connection | None = None
        self._rng = np.random.default_rng(seed)
        self._tls_ctx: ssl.SSLContext | None = server_context(tls) if tls else None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind_host, bind_port))
        self._sock.listen(8)
        self.host: str = bind_host
        self.port: int = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._running = True

    def start(self) -> None:
        self._thread.start()

    # ------------------------------------------------------------------- tools
    def _run_tool(self, tool: str, scale: float) -> bytes:
        """Execute one real tool and serialize a response of its true size."""
        rng = self._rng
        if tool == "code_exec":
            _, resp, signal = _tool_code_exec_detail(scale, rng)
        elif tool == "db_query":
            assert self._conn is not None
            _, resp, signal = _tool_db_query_detail(self._conn, scale, rng)
        elif tool == "file_io":
            _, resp, signal = _tool_file_io_detail(scale, rng)
        elif tool == "search":
            _, resp, signal = _tool_search_detail(self._corpus, scale, rng)
        else:
            _, resp, signal = _tool_json_detail(scale, rng)
        body = {"tool": tool, "signal": signal}
        # Materialize an actual payload so the response's true byte size is the
        # tool's real output size.
        overhead = len(json.dumps({**body, "payload": ""}).encode())
        pad = max(0, int(resp) - overhead)
        body["payload"] = "x" * pad
        return json.dumps(body).encode()

    def _serve(self) -> None:
        # Build thread-local backends here so the SQLite handle lives on this thread.
        self._corpus = build_corpus(seed=self._seed)
        self._conn = build_sqlite(seed=self._seed)
        while self._running:
            try:
                raw, _ = self._sock.accept()
            except OSError:
                break
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn: socket.socket
            if self._tls_ctx is not None:
                try:
                    conn = self._tls_ctx.wrap_socket(raw, server_side=True)
                except (ssl.SSLError, OSError):
                    raw.close()
                    continue
            else:
                conn = raw
            self._serve_connection(conn)

    def _serve_connection(self, conn: socket.socket) -> None:
        buf = b""
        try:
            with conn:
                while self._running:
                    chunk = conn.recv(_RECV_CHUNK)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        req = json.loads(line)
                        if req.get("tool") == "__close__":
                            return
                        resp = self._run_tool(req["tool"], float(req.get("scale", 1.0)))
                        conn.sendall(resp + b"\n")
        except (OSError, ssl.SSLError, json.JSONDecodeError):
            pass

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


# ----------------------------------------------------------------------- planner
class EmptySessionError(RuntimeError):
    """One planner draw produced no tool calls, so there is no trace to record.

    Distinct from a transport failure on purpose: a stochastic planner may decline a
    task on a given draw, and the caller should redraw rather than abandon the capture.
    """


@dataclass
class StepObservation:
    """What the controller learns after a step: the tool's *real* output signals."""

    tool: str
    scale: float
    signal: dict[str, float]
    response_bytes: int


Step = tuple[str, float]
_Program = Generator[Step, StepObservation, None]


class Planner(Protocol):
    """The ``(tool, scale)`` step interface every controller implements.

    ``program(task_name)`` returns a generator that yields one ``(tool, scale)`` step
    at a time and receives the resulting :class:`StepObservation` via ``send`` --- so
    a controller may branch on what the tools actually returned. A neural LLM
    controller plugs in here unchanged (see :class:`LLMPlanner`).
    """

    def program(self, task_name: str) -> _Program:  # pragma: no cover - protocol
        ...


class StochasticReActPlanner:
    """A ReAct-style controller that branches on **real** tool outputs (paper RQ8).

    Branch conditions read the signals the server returns from genuine tool runs:
    the number of corpus documents the search actually matched, the number of rows
    SQLite actually returned, and the parity of the integer the executed Python
    actually computed. Plan length and composition therefore vary with real results,
    and (with probability 0.25) a second task is interleaved in the same session.
    """

    def __init__(self, seed: int = 20270214) -> None:
        self._rng = np.random.default_rng(seed)

    # The program is written as a coroutine: ``yield`` emits a step, and the value
    # sent back is the real observation for that step.
    def program(self, task_name: str) -> _Program:
        rng = self._rng
        yield from self._backbone(task_name, rng)
        # With prob 0.25 interleave a second task in the same session (multi-task).
        if rng.random() < 0.25:
            other = LIVE_CLASSES[int(rng.integers(len(LIVE_CLASSES)))]
            yield from self._short_backbone(other, rng)

    def _backbone(self, task_name: str, rng: np.random.Generator) -> _Program:
        if task_name in ("literature_review", "market_scan"):
            docs = 0.0
            tries = 0
            # Loop searches until the corpus has actually returned enough documents.
            while docs < 8 and tries < 5:
                obs = yield ("search", float(rng.uniform(0.5, 1.5)))
                docs += obs.signal.get("n_returned", 0.0)
                tries += 1
            n_fetch = int(min(3, max(1, docs // 4)))  # result-dependent fetches
            for _ in range(n_fetch):
                yield ("fetch", float(rng.uniform(0.5, 1.5)))
        elif task_name == "trip_planning":
            obs = yield ("search", float(rng.uniform(0.5, 1.5)))
            n_fetch = int(min(3, max(1, obs.signal.get("n_returned", 1.0) // 3)))
            for _ in range(n_fetch):
                yield ("fetch", float(rng.uniform(0.5, 1.5)))
            yield ("calendar", 1.0)
            yield ("email", 1.0)
        elif task_name == "debug_failing_test":
            parity = 1.0
            attempts = 0
            while parity == 1.0 and attempts < 4:   # retry while the run is 'failing'
                obs = yield ("code_exec", float(rng.uniform(0.5, 1.5)))
                parity = obs.signal.get("parity", 0.0)
                yield ("file_io", float(rng.uniform(0.5, 1.5)))
                attempts += 1
        elif task_name == "status_reporting":
            rows = 99.0
            queries = 0
            while rows > 10 and queries < 5:        # re-query when many rows return
                obs = yield ("db_query", float(rng.uniform(0.5, 1.5)))
                rows = obs.signal.get("n_rows", 0.0)
                queries += 1
            yield ("calc", 1.0)
        elif task_name == "meeting_scheduling":
            yield ("calendar", 1.0)
            yield ("email", 1.0)
            if rng.random() < 0.5:
                yield ("search", 1.0)
        else:
            yield ("search", 1.0)
            yield ("fetch", 1.0)

    def _short_backbone(self, task_name: str, rng: np.random.Generator) -> _Program:
        if task_name in ("literature_review", "market_scan", "trip_planning"):
            pair = (("search", 1.0), ("fetch", 1.0))
        elif task_name == "debug_failing_test":
            pair = (("code_exec", 1.0), ("file_io", 1.0))
        elif task_name == "status_reporting":
            pair = (("db_query", 1.0), ("calc", 1.0))
        else:
            pair = (("calendar", 1.0), ("email", 1.0))
        for step in pair:
            yield step


class LLMPlanner:
    """Neural controller behind the same ``(tool, scale)`` step interface (RQ8 gap).

    The paper's stated limitation is that its live controller is programmatic rather
    than a neural LLM. Closing that gap needs an LLM endpoint, which is an external
    dependency this artifact does not bundle, download, or hold credentials for. This
    class is the wired extension point: supply ``complete``, a callable that takes a
    prompt string and returns the model's completion, and the live harness drives a
    real model with no other change.

    The expected completion is a single JSON object per step::

        {"tool": "search", "scale": 1.0}     # or {"done": true}

    A malformed completion ends the session (a planner that silently substitutes a
    hand-written step would make the trace a fiction).
    """

    def __init__(self, complete: Callable[[str], str] | None = None,
                 max_steps: int = 20, parse_retries: int = 2) -> None:
        self.complete = complete
        self.max_steps = max_steps
        # A real ReAct loop does not abandon a session because one completion came back
        # as prose instead of JSON -- it asks again. Neither should this: reasoning
        # models routinely spend a whole budget thinking and answer on the retry, and
        # without this a 150-session capture dies on session three. The retry prompt is
        # firmer, and the retries cost only LLM calls, which the network observer never
        # sees; they change no observable.
        self.parse_retries = parse_retries
        self.parse_failures = 0

    def program(self, task_name: str) -> _Program:
        if self.complete is None:
            raise NotImplementedError(
                "LLMPlanner needs a `complete(prompt) -> str` callable bound to a real "
                "LLM endpoint (credentials are the user's to provide). Without one, use "
                "StochasticReActPlanner and report the trace as programmatically planned."
            )
        history: list[StepObservation] = []
        last_reply = ""
        for _ in range(self.max_steps):
            step, finished = None, False
            for attempt in range(self.parse_retries + 1):
                prompt = self._prompt(task_name, history)
                if attempt:
                    prompt += ("\n\nYour previous reply could not be parsed. Reply with "
                               "the JSON object ONLY, no reasoning and no prose.")
                last_reply = self.complete(prompt)
                step = self._parse(last_reply)
                if step is not None:
                    break
                if self._is_done(last_reply):
                    finished = True
                    break
                self.parse_failures += 1
            if step is None:
                if finished:
                    return
                if not history:
                    # A model that never emits a usable call would otherwise surface far
                    # away as "planner produced no steps", with no clue why. Say what it
                    # actually replied.
                    raise ValueError(
                        f"the model produced no usable tool call for task {task_name!r} "
                        f"in {self.parse_retries + 1} attempts; last reply was "
                        f"{last_reply.strip()[:200]!r}. If it reads like reasoning rather "
                        f"than an answer, the model is spending its whole budget "
                        f"thinking: raise --max-tokens."
                    )
                return
            obs = yield step
            history.append(obs)

    @staticmethod
    def _prompt(task_name: str, history: list[StepObservation]) -> str:
        # Offering `{"done": true}` before any tool has run invites a model to decline
        # the task outright, which yields a session with no calls -- no trace at all,
        # for a harness whose entire purpose is to record tool-use traces. These tasks
        # require tools by construction, so the first turn says so and offers only the
        # call form. Later turns keep the stop option, because when to stop is a real
        # planner decision and it shapes the trace-length distribution we measure.
        lines = [
            "You are the controller of a tool-using agent. Decide the NEXT tool call.",
            "Reply with ONE JSON object and nothing else:",
        ]
        if history:
            lines.append('{"tool": "<name>", "scale": <0.5-1.5>}  or  {"done": true}')
        else:
            lines.append('{"tool": "<name>", "scale": <0.5-1.5>}')
        lines += [
            f"Available tools: {', '.join(TOOL_CATEGORIES)}",
            f"Task: {task_name.replace('_', ' ')}",
        ]
        if history:
            lines.append("Results so far:")
            for i, obs in enumerate(history):
                lines.append(f"  {i + 1}. {obs.tool} -> {json.dumps(obs.signal)}")
        else:
            lines.append("No tools called yet. This task requires at least one tool "
                         "call, so choose one now.")
        return "\n".join(lines)

    @staticmethod
    def _parse(reply: str) -> Step | None:
        """Tolerant parse: models wrap JSON in prose or markdown fences.

        Shares the framework adapters' parser so the neural planner and the neural
        adapter cannot drift apart in what they accept.
        """
        from leakbench.data.adapters import _parse_step

        return _parse_step(reply)

    @staticmethod
    def _is_done(reply: str) -> bool:
        """Did the model say it had finished, as opposed to failing to answer?

        The distinction matters once the planner retries: retrying a model that
        correctly announced it was done wastes two calls at the end of every session
        and misreads a success as a failure.

        This has to **parse**, not substring-match. The prompt documents
        ``{"done": true}`` as the stop format, so a reasoning model deliberating about
        whether to stop writes the word in its chain of thought -- and a substring test
        then reads deliberation as a decision and ends the session before it makes a
        single tool call. That is exactly how gpt-oss-20b produced zero steps.
        """
        from leakbench.data.adapters import json_objects

        text = reply.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        spans = json_objects(text)
        if not spans:
            return False
        try:                              # the last object is the answer, not the aside
            obj = json.loads(spans[-1])
        except json.JSONDecodeError:
            return False
        return isinstance(obj, dict) and bool(obj.get("done"))


# --------------------------------------------------------------------- harness
@dataclass
class LiveCaptureHarness:
    """Drive real tools over a captured (TLS) loopback socket with a live planner.

    Parameters
    ----------
    taxonomy:
        used only to resolve class names/families; the live study relabels to a
        compact ``0..len(classes)-1`` space.
    seed:
        seeds the planner and the server's tool-parameter draws.
    use_tls:
        attempt a real TLS handshake (default). If certificate material cannot be
        provisioned locally, the run falls back to plaintext TCP and records
        ``capture_mode = tcp_captured_proxy``.
    planner:
        any object implementing :class:`Planner`. Defaults to
        :class:`StochasticReActPlanner`.
    server_host, server_port:
        point at a tool server running on **another host** (``scripts/run_tool_server.py``)
        instead of starting one in this process. The capture proxy then still runs
        locally, so it sits on the client's uplink and captures the bytes as they
        cross the real link — which is what turns the loopback experiment into the WAN
        experiment (``docs/ec2_experiment_guide.md`` §5a). The recorded ``link``
        metadata becomes ``wan`` and the "loopback not WAN" gap disappears.
    server_ca_file:
        path to the remote server's certificate, needed to verify TLS across a real
        link (copy the file the server script writes).
    """

    taxonomy: Taxonomy = field(default_factory=reference_taxonomy)
    seed: int = 20270214
    use_tls: bool = True
    planner: Planner | None = None
    server_host: str | None = None
    server_port: int | None = None
    server_ca_file: str | None = None

    def _class_index(self, task_name: str) -> int:
        for c in self.taxonomy.classes:
            if c.name == task_name:
                return c.index
        # scaled taxonomy fallback: map by order
        return LIVE_CLASSES.index(task_name)

    # --------------------------------------------------------------- one session
    def run_session(self, task_name: str, planner: Planner, proxy: _CaptureProxy,
                    client: socket.socket, log: list[_CaptureRecord],
                    capture_mode: str) -> Trace:
        """Run one task session over an already-open connection, capturing the wire.

        The per-step network observable is the sum of the **captured** chunk sizes the
        proxy carried between the request being written and the response being fully
        received --- ciphertext bytes when TLS is active.
        """
        program = planner.program(task_name)
        calls: list[ToolCall] = []
        prev_start: float | None = None
        pending: StepObservation | None = None
        modeled_tls = capture_mode == CAPTURE_MODE_TCP

        while True:
            try:
                step = program.send(pending) if pending is not None else next(program)
            except StopIteration:
                break
            tool, scale = step
            req = (json.dumps({"tool": tool, "scale": scale}) + "\n").encode()

            mark = proxy.size(log)
            t0 = time.perf_counter()
            client.sendall(req)
            resp = b""
            while b"\n" not in resp:
                chunk = client.recv(_RECV_CHUNK)
                if not chunk:
                    break
                resp += chunk

            # Wait for the proxy to have logged at least the plaintext response size
            # (ciphertext is never smaller), then read the captured chunk sizes.
            up_size, down_size = self._captured_sizes(
                proxy, log, mark, min_down=len(resp), min_up=len(req)
            )
            if modeled_tls:
                # No real TLS available: apply TLS record framing as a MODEL on top of
                # the captured TCP chunk sizes (recorded in capture_mode).
                up_size = tls_frame_size(up_size)
                down_size = tls_frame_size(down_size)

            # Delta_i = inter-arrival gap between consecutive requests on the wire
            # (start of call i-1 -> start of call i), which contains the previous
            # call's real round-trip + tool service time. This is the quantity an
            # on-path observer times between bursts; see measured.py for why
            # end-to-start would instead measure only the client's loop overhead.
            gap = 0.0 if prev_start is None else max(0.0, t0 - prev_start)
            prev_start = t0
            calls.append(
                ToolCall(
                    category=tool if tool in TOOL_CATEGORIES else "fetch",
                    request_size=float(max(1, up_size)),
                    response_size=float(max(1, down_size)),
                    gap=gap,
                )
            )
            pending = StepObservation(
                tool=tool, scale=scale,
                signal=_parse_signal(resp), response_bytes=len(resp),
            )

        if not calls:
            # A model may still decline the task on a given draw. That is one degenerate
            # session, not a reason to lose the other 149 -- the planner is stochastic,
            # so the caller can simply draw again. Raise a distinct type so the caller
            # can tell "this draw produced nothing" from a real transport failure.
            raise EmptySessionError(
                f"planner produced no tool calls for task {task_name!r}")
        label = self._class_index(task_name)
        fam = self.taxonomy.classes[label].family if label < self.taxonomy.num_classes else None
        return Trace(calls=calls, label=label, family=fam,
                     metadata={"source": "live", "capture_mode": capture_mode,
                               "task": task_name})

    def _captured_sizes(self, proxy: _CaptureProxy, log: list[_CaptureRecord], mark: int,
                        *, min_down: int, min_up: int) -> tuple[float, float]:
        deadline = time.perf_counter() + _DRAIN_TIMEOUT_S
        while True:
            events = proxy.snapshot(log, mark)
            up = sum(e.size for e in events if e.direction == "up")
            down = sum(e.size for e in events if e.direction == "down")
            if (up >= min_up and down >= min_down) or time.perf_counter() > deadline:
                return float(up), float(down)
            time.sleep(0.0005)

    # -------------------------------------------------------------- one dataset
    def capture_dataset(self, traces_per_class: int,
                        classes: tuple[str, ...] = LIVE_CLASSES) -> TraceDataset:
        """Capture a live dataset: one persistent connection per session, through a
        real capture proxy, over real TLS when available.

        With ``server_host`` set, the tool server is assumed to be running on another
        host and only the client + capture proxy run here, so the captured bytes are
        the ones that crossed the real link.
        """
        remote = self.server_host is not None
        upstream: tuple[str, int]
        material: TLSMaterial | None = None
        server: _ToolServer | None = None
        ssl_ctx: ssl.SSLContext | None = None

        if remote:
            if self.server_port is None:
                raise ValueError("server_port is required when server_host is set.")
            upstream = (str(self.server_host), int(self.server_port))
            if self.use_tls:
                if not self.server_ca_file:
                    raise ValueError(
                        "server_ca_file is required to verify TLS against a remote tool "
                        "server (copy the certificate the server printed), or pass "
                        "use_tls=False to capture plaintext TCP."
                    )
                ssl_ctx = client_context_from_ca(self.server_ca_file)
        else:
            material = generate_self_signed() if self.use_tls else None
            if self.use_tls and material is None:
                logger.warning(
                    "No TLS material could be provisioned (no cryptography package and "
                    "no openssl binary); falling back to captured plaintext TCP with "
                    "MODELED TLS framing. capture_mode=%s", CAPTURE_MODE_TCP
                )
            server = _ToolServer(seed=self.seed, tls=material)
            server.start()
            upstream = ("127.0.0.1", server.port)
            ssl_ctx = client_context(material) if material is not None else None

        tls_active = ssl_ctx is not None
        capture_mode = CAPTURE_MODE_TLS if tls_active else CAPTURE_MODE_TCP
        proxy = _CaptureProxy(upstream[0], upstream[1])
        proxy.start()
        planner = self.planner or StochasticReActPlanner(seed=self.seed)

        traces: list[Trace] = []
        handshake_bytes: list[int] = []
        empty_sessions = 0
        # A neural capture takes tens of minutes and used to print nothing until it
        # finished, so "slow" and "stuck" looked identical from outside and the only
        # way to find out was to wait. Report per class with a rate and an estimate.
        started = time.time()
        total_wanted = len(classes) * traces_per_class
        try:
            for task in classes:
                collected = 0
                attempts = 0
                # Redraw a declined session instead of losing the capture. The budget is
                # generous but finite: a model that declines *every* draw is a real
                # configuration problem and should surface as one, not spin forever.
                max_attempts = traces_per_class * 3 + 10
                while collected < traces_per_class and attempts < max_attempts:
                    attempts += 1
                    raw = socket.create_connection(("127.0.0.1", proxy.port), timeout=10)
                    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    local_addr = raw.getsockname()
                    client: socket.socket = raw
                    try:
                        if ssl_ctx is not None:
                            client = ssl_ctx.wrap_socket(raw, server_hostname="localhost")
                        client.settimeout(_SESSION_TIMEOUT_S)
                        log = proxy.log_for(local_addr)
                        # Everything captured so far is handshake, not agent traffic.
                        handshake_bytes.append(
                            sum(e.size for e in proxy.snapshot(log, 0))
                        )
                        try:
                            traces.append(self.run_session(task, planner, proxy, client,
                                                           log, capture_mode))
                            collected += 1
                        except EmptySessionError:
                            empty_sessions += 1
                            handshake_bytes.pop()   # that handshake bought no trace
                            logger.debug("empty session for %s; redrawing", task)
                    finally:
                        try:
                            client.sendall((json.dumps({"tool": "__close__"}) + "\n").encode())
                        except OSError:
                            pass
                        try:
                            client.close()
                        except OSError:
                            pass
                done = len(traces)
                rate = done / max(1e-9, time.time() - started)
                remaining = (total_wanted - done) / rate if rate > 0 else float("inf")
                logger.info(
                    "%s: %d/%d sessions (%d/%d total, %.1f/min, ~%.0f min left"
                    "%s)", task, collected, traces_per_class, done, total_wanted,
                    rate * 60, remaining / 60,
                    f", {empty_sessions} declined" if empty_sessions else "")
                if collected < traces_per_class:
                    logger.warning(
                        "%s: only %d/%d sessions after %d attempts -- the planner "
                        "declined the rest, so this class is under-sampled",
                        task, collected, traces_per_class, attempts)
        finally:
            proxy.close()
            if server is not None:
                server.close()
            if material is not None:
                material.cleanup()

        # Re-label to a compact 0..len(classes)-1 space for the live study.
        name_to_new = {name: i for i, name in enumerate(classes)}
        for t in traces:
            task = t.metadata["task"]
            t.label = name_to_new[task]
        link = _link_kind(self.server_host)
        gaps = ([] if link == "remote" else ["loopback not WAN"])
        if not isinstance(planner, LLMPlanner):
            gaps.append("programmatic not neural planner")
        if not tls_active:
            gaps.append("TLS framing modeled over captured TCP")
        return TraceDataset(
            traces=traces,
            taxonomy_name=f"live/{len(classes)}c",
            taxonomy_hash=self.taxonomy.content_hash(),
            source="live",
            metadata={
                "traces_per_class": traces_per_class,
                "seed": self.seed,
                "classes": list(classes),
                "capture_mode": capture_mode,
                "capture": "captured_at_tcp_proxy",
                "tls": ("real" if tls_active else "modeled_over_captured_tcp"),
                "tls_generator": (material.generator if material is not None
                                  else ("provided_ca" if tls_active else None)),
                "planner": type(planner).__name__,
                "planner_kind": ("neural_llm" if isinstance(planner, LLMPlanner)
                                 else "programmatic_stochastic"),
                "link": link,
                "link_note": (
                    "captured on the client's uplink across a real link; record the "
                    "host topology (AZ/region pair) alongside this run"
                    if link == "remote" else
                    "client, proxy and server share one host"
                ),
                "server": (f"{self.server_host}:{self.server_port}" if remote
                           else "in-process (loopback)"),
                "mean_handshake_bytes": (float(np.mean(handshake_bytes))
                                         if handshake_bytes else 0.0),
                # How often the planner declined a task outright. Redrawn rather than
                # recorded (an empty session has no observable), but reported, because
                # a high count means the model is refusing the task and the traces that
                # did get collected are a filtered sample.
                "empty_sessions_redrawn": empty_sessions,
                "gaps": gaps,
            },
        )


def _link_kind(server_host: str | None) -> str:
    """``loopback`` or ``remote``, resolved from the address actually used.

    Derived rather than assumed: a two-host *flag* pointed back at 127.0.0.1 is still
    a loopback measurement, and recording it as a WAN link would misdescribe the run.
    """
    if server_host is None:
        return "loopback"
    try:
        resolved = socket.gethostbyname(server_host)
        return "loopback" if ipaddress.ip_address(resolved).is_loopback else "remote"
    except (OSError, ValueError):
        return "remote"


def _parse_signal(resp: bytes) -> dict[str, float]:
    """Extract the tool's real output signals from a server response."""
    try:
        obj: Any = json.loads(resp.decode().strip())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    sig = obj.get("signal", {}) if isinstance(obj, dict) else {}
    return {str(k): float(v) for k, v in sig.items()} if isinstance(sig, dict) else {}


def iter_live_classes() -> Iterator[str]:
    return iter(LIVE_CLASSES)
