"""Framework adapters --- the high-fidelity trace source that instruments *real*
agent stacks (paper §Design Stage 2(a)).

The paper instruments a ReAct-style loop, an MCP router, and a LangChain-style
pipeline to emit Def. "Agent trace" records as the agent runs a taxonomy task.

What ships here:

- :class:`ReActAdapter` --- a **real** plan-act loop. A real LLM decides each next
  tool call; the call runs against the genuine local backends of
  :mod:`leakbench.data.measured`; the tool's real output signals go back into the next
  prompt. Needs a ``complete(prompt) -> str`` callable bound to an LLM endpoint
  (:mod:`leakbench.data.llm` builds one for OpenRouter / Groq / OpenAI / any
  OpenAI-compatible URL).
- :class:`MCPRouterAdapter` --- the same loop dispatched over a real socket to the
  MCP-style tool server, so sizes are what a router actually put on the wire.
- :class:`LangChainAdapter` --- still a documented stub: LangChain is a heavyweight
  SDK with a moving callback API, and inventing it would be worse than leaving it
  open.
- :class:`DummyAdapter` --- deterministic, tests only, never a real baseline.

Every LLM-backed path **raises** without a real endpoint rather than falling back to
a scripted plan: a scripted plan reported as an LLM trace would be a fiction.

STATUS: react/mcp_router implemented (credential-gated); langchain stub. See the
traceability CSV.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from leakbench.data.schemas import ToolCall, Trace, TraceDataset
from leakbench.taxonomy import TOOL_CATEGORIES, Taxonomy, reference_taxonomy


@dataclass
class ToolCallEvent:
    """The event schema a real framework adapter must emit per tool call.

    A framework integration is responsible for populating these fields from its own
    instrumentation (e.g. MCP request/response hooks, LangChain callbacks). Sizes are
    *bytes on the encrypted channel*; ``gap`` is seconds since the previous call.
    """

    category: str          # must be one of TOOL_CATEGORIES
    request_size: float
    response_size: float
    gap: float


class FrameworkAdapter(ABC):
    """Abstract adapter: run one taxonomy task, yield a stream of ToolCallEvents.

    Implement :meth:`run_task` against your own stack and
    the rest of LeakBench (observation maps, estimator) works unchanged.
    """

    name: str = "adapter"

    @abstractmethod
    def run_task(self, class_index: int) -> Iterable[ToolCallEvent]:
        """Execute the agent on one task class and yield per-call events."""

    def _events_to_trace(self, class_index: int, events: Iterable[ToolCallEvent],
                         taxonomy: Taxonomy) -> Trace:
        calls = [
            ToolCall(category=e.category, request_size=max(1.0, e.request_size),
                     response_size=max(1.0, e.response_size), gap=max(0.0, e.gap))
            for e in events
        ]
        if not calls:
            raise ValueError("Adapter produced an empty trace.")
        return Trace(calls=calls, label=class_index,
                     family=taxonomy.family_of(class_index),
                     metadata={"source": f"adapter:{self.name}"})

    def sample_dataset(self, taxonomy: Taxonomy, traces_per_class: int) -> TraceDataset:
        traces: list[Trace] = []
        for ci in range(taxonomy.num_classes):
            for _ in range(traces_per_class):
                traces.append(self._events_to_trace(ci, self.run_task(ci), taxonomy))
        return TraceDataset(
            traces=traces,
            taxonomy_name=taxonomy.name,
            taxonomy_hash=taxonomy.content_hash(),
            source=f"adapter:{self.name}",
            metadata={"traces_per_class": traces_per_class},
        )


# ------------------------------------------------------- LLM-driven adapters
@dataclass
class ReActAdapter(FrameworkAdapter):
    """A **real** ReAct-style plan-act loop: a real LLM drives real tools.

    The model is asked, step by step, which tool to call next; each step is executed
    against the genuine local backends of :mod:`leakbench.data.measured` (in-process
    Python exec, in-memory SQLite, real temp-file I/O, corpus search, JSON payloads),
    and the tool's **real** output signals are fed back into the next prompt. The
    emitted :class:`ToolCallEvent` sizes and inter-arrival gaps are therefore measured,
    not drawn.

    This closes the "templated planner" simplification the paper discloses: the plan
    is whatever the model decides, given what the tools actually returned.

    Parameters
    ----------
    complete:
        ``complete(prompt) -> str``, e.g. from
        :func:`leakbench.data.llm.make_completion_fn`. **Required** --- without a real
        endpoint this adapter raises rather than substituting a scripted plan, because
        a scripted plan reported as an LLM trace would be a fiction.
    taxonomy:
        the task taxonomy whose class names are put to the model.
    max_steps:
        hard cap on tool calls per task (also caps the token bill).

    Cost note: one task ≈ ``max_steps`` short completions. 12 classes × 100 traces at
    8 steps is ~10k small calls; start with a few traces per class and check
    :meth:`usage` on your client.
    """

    name: str = "react"
    complete: Callable[[str], str] | None = None
    taxonomy: Taxonomy | None = None
    max_steps: int = 8
    seed: int = 20270214

    def __post_init__(self) -> None:
        if self.taxonomy is None:
            self.taxonomy = reference_taxonomy()
        self._runtime = _ToolRuntime(seed=self.seed)

    def run_task(self, class_index: int) -> Iterable[ToolCallEvent]:
        if self.complete is None:
            raise NotImplementedError(
                "ReActAdapter needs a `complete(prompt) -> str` callable bound to a "
                "real LLM endpoint: "
                "`from leakbench.data.llm import make_completion_fn; "
                "fn, client = make_completion_fn('openrouter')`. "
                "See docs/baseline_adapters.md."
            )
        assert self.taxonomy is not None
        task = self.taxonomy.classes[class_index]
        history: list[tuple[str, dict[str, float]]] = []
        events: list[ToolCallEvent] = []
        prev_start: float | None = None

        for _ in range(self.max_steps):
            reply = self.complete(self._prompt(task.name, history))
            step = _parse_step(reply)
            if step is None:
                break
            tool, scale = step
            t0 = time.perf_counter()
            req_bytes, resp_bytes, signal = self._runtime.run(tool, scale)
            gap = 0.0 if prev_start is None else max(0.0, t0 - prev_start)
            prev_start = t0
            events.append(
                ToolCallEvent(category=tool, request_size=req_bytes,
                              response_size=resp_bytes, gap=gap)
            )
            history.append((tool, signal))

        if not events:
            raise ValueError(
                f"the model produced no usable tool call for class {class_index} "
                f"({task.name}); check the model id and that it can emit JSON."
            )
        return events

    def _prompt(self, task_name: str, history: list[tuple[str, dict[str, float]]]) -> str:
        lines = [
            "You are the controller of a tool-using agent. Decide the NEXT tool call.",
            "Reply with ONE JSON object and nothing else:",
            '{"tool": "<name>", "scale": <0.5-1.5>}  or  {"done": true}',
            f"Available tools: {', '.join(TOOL_CATEGORIES)}",
            f"Task: {task_name.replace('_', ' ')}",
        ]
        if history:
            lines.append("Results so far:")
            for i, (tool, signal) in enumerate(history):
                lines.append(f"  {i + 1}. {tool} -> {json.dumps(signal)}")
        else:
            lines.append("No tools called yet.")
        return "\n".join(lines)


@dataclass
class MCPRouterAdapter(ReActAdapter):
    """The same loop dispatched through an **MCP-style router** over a real socket.

    Where :class:`ReActAdapter` calls the tool backends in-process, this variant sends
    each call as newline-delimited JSON over a real (TLS, when available) connection to
    the tool server of :mod:`leakbench.data.live_capture` --- the paper's MCP-style
    adapter path. Sizes are then what a router actually put on the wire.

    Requires the same ``complete`` callable. Because it stands up a server per
    dataset, prefer :meth:`sample_dataset` over calling :meth:`run_task` in a loop.
    """

    name: str = "mcp_router"

    def sample_dataset(self, taxonomy: Taxonomy, traces_per_class: int) -> TraceDataset:
        if self.complete is None:
            raise NotImplementedError(
                "MCPRouterAdapter needs a `complete(prompt) -> str` callable bound to a "
                "real LLM endpoint. See docs/baseline_adapters.md."
            )
        from leakbench.data.live_capture import LiveCaptureHarness, LLMPlanner

        planner = LLMPlanner(complete=self.complete, max_steps=self.max_steps)
        harness = LiveCaptureHarness(taxonomy=taxonomy, seed=self.seed, planner=planner)
        classes = tuple(c.name for c in taxonomy.classes)
        ds = harness.capture_dataset(traces_per_class, classes=classes)
        ds.metadata["adapter"] = self.name
        return ds


class LangChainAdapter(FrameworkAdapter):
    """TODO: instrument a LangChain-style pipeline via callbacks.

    Left as a stub deliberately: LangChain is a heavyweight external SDK whose
    callback API changes between releases, and inventing it would be worse than
    leaving it open. If you use LangChain, subclass :class:`FrameworkAdapter` and emit
    a :class:`ToolCallEvent` from an ``on_tool_end`` callback --- everything
    downstream then works unchanged. See docs/baseline_adapters.md.
    """

    name = "langchain"

    def run_task(self, class_index: int) -> Iterable[ToolCallEvent]:  # pragma: no cover
        raise NotImplementedError(
            "LangChainAdapter is a stub. Use LangChain tool callbacks to emit "
            "ToolCallEvent per tool call. See docs/baseline_adapters.md."
        )


class _ToolRuntime:
    """Executes the genuine local tool backends for the in-process adapters."""

    def __init__(self, seed: int = 20270214) -> None:
        from leakbench.data.corpus import build_corpus, build_sqlite

        self._rng = np.random.default_rng(seed)
        self._corpus = build_corpus(seed=seed)
        self._conn = build_sqlite(seed=seed)

    def run(self, tool: str, scale: float) -> tuple[float, float, dict[str, float]]:
        from leakbench.data.measured import (
            _tool_code_exec_detail,
            _tool_db_query_detail,
            _tool_file_io_detail,
            _tool_json_detail,
            _tool_search_detail,
        )

        if tool == "code_exec":
            return _tool_code_exec_detail(scale, self._rng)
        if tool == "db_query":
            return _tool_db_query_detail(self._conn, scale, self._rng)
        if tool == "file_io":
            return _tool_file_io_detail(scale, self._rng)
        if tool == "search":
            return _tool_search_detail(self._corpus, scale, self._rng)
        return _tool_json_detail(scale, self._rng)


def json_objects(text: str) -> list[str]:
    """Every balanced top-level ``{...}`` span in ``text``, in order of appearance.

    Taking ``text[first_brace : last_brace]`` is the obvious thing to do and it is
    wrong for exactly the models we most want to support. A reasoning model narrates
    before it answers, and the narration quotes JSON:

        <reasoning>I could reply {"done": true}, but no tool has run yet...</reasoning>
        {"tool": "search", "scale": 1.0}

    First-to-last spans from the brace inside the narration to the brace closing the
    real answer, which parses as nothing at all. Scanning for balanced spans finds
    both objects, and the caller can prefer the last one -- the answer, not the
    deliberation. Braces inside strings are skipped so a quoted ``{`` cannot unbalance
    the scan.
    """
    spans: list[str] = []
    depth = start = 0
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    return spans


def _parse_step(reply: str) -> tuple[str, float] | None:
    """Read one ``{"tool": ..., "scale": ...}`` object out of a model reply.

    Candidates are tried **last first**: when a model narrates before answering, the
    answer is the final object, and an earlier one is something it was thinking about.
    """
    text = reply.strip()
    if text.startswith("```"):                      # strip a markdown fence
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    for span in reversed(json_objects(text)):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("done") or "tool" not in obj:
            continue
        tool = str(obj["tool"]).strip()
        if tool not in TOOL_CATEGORIES:
            continue
        try:
            scale = float(obj.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        return tool, float(min(1.5, max(0.5, scale)))
    return None


# ---------------------------------------------------------------- dummy (tests)
@dataclass
class DummyAdapter(FrameworkAdapter):
    """A deterministic dummy adapter for tests ONLY. Not a real agent stack; it does
    not run any model or tool. It merely emits schema-valid events so the
    adapter->trace->observable->estimator path can be exercised end-to-end."""

    name: str = "dummy"
    taxonomy: Taxonomy = None  # type: ignore[assignment]
    seed: int = 20270214

    def __post_init__(self) -> None:
        if self.taxonomy is None:
            self.taxonomy = reference_taxonomy()
        self._rng = np.random.default_rng(self.seed)

    def run_task(self, class_index: int) -> Iterable[ToolCallEvent]:
        rng = self._rng
        sig = self.taxonomy.classes[class_index].signature_tools
        n = int(rng.integers(4, 10))
        events = []
        for i in range(n):
            cat = (sig[int(rng.integers(len(sig)))] if rng.random() < 0.7
                   else TOOL_CATEGORIES[int(rng.integers(len(TOOL_CATEGORIES)))])
            events.append(
                ToolCallEvent(
                    category=cat,
                    request_size=float(rng.integers(64, 2048)),
                    response_size=float(rng.integers(128, 16384)),
                    gap=0.0 if i == 0 else float(abs(rng.normal(0.3, 0.1))),
                )
            )
        return events
