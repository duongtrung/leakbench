# Baselines and adapters

This document distinguishes three things that must never be confused: **real
baselines**, **adapter stubs** (waiting for an external dependency), and **dummy
baselines** (for tests only).

Credentials for the LLM-backed paths go in a gitignored `.env` (see `.env.example`);
they are read from the environment, never committed, and never written into a result
file — only the provider/model/call counts are.

## 1. Deployed attacks as projections (real comparators)

The paper does not re-run external attack code; it re-expresses each deployed attack
as a **projection** of the unified observable and measures it on common traces
(§Projections, §Eval "Case study"). These are implemented in
`observables/projections.py`:

| Attack | Vantage | Projection | Code key |
|---|---|---|---|
| Token-length keylogger | NET | response sizes | `keylogger` |
| Whisper Leak | NET | size + timing | `whisper_leak` |
| AgentPrint | NET | full `O_net` | `agentprint` |
| Privacy side-channels | PLAT | category + count | `privacy_sidechannel` |
| Size-only (RQ8) | NET | volume + size | `size_only` |
| Timing-only (RQ8) | NET | timing | `timing_only` |

Use `estimators.leakage.estimate_leakage_projection(X_network, y, "whisper_leak")`.
The **cache side channel** observes host micro-architectural state outside the
network/platform/provider model and is intentionally **not** an `O_v` projection.

Measure them all on common traces with the `projections` stage:

```bash
python scripts/run_experiment.py --config configs/measured.yaml --stage projections
```

which reports each projection alongside the full network observable and the platform
vantage — the comparison the deployed papers cannot make against one another. These
are the paper's comparators, **not** re-runs of the original attack code; running the
published implementations would require obtaining them, and their APIs are not
invented here.

## 2. Defense baselines (real, fully implemented)

`defenses/transforms.py`: `none`, `pad-512`, `shuffle`, `constant-rate`, `cover`,
`pad-max`, `full`. Each is directly specified by the paper and implemented with real
bandwidth/latency accounting.

Two further defenses go beyond the paper's shipped artifact, on the same axis:

| Defense | Analogue | Module | Scope |
|---|---|---|---|
| `gen-shaping` | Surakav | `defenses/generative.py` | learned per-class trace model; decoy reference drawn independently of the true label |
| `dp-shaping` | NetShaper | `defenses/dp_shaping.py` | discrete-Laplace releases on a fixed grid; `epsilon` maps to an advantage ceiling |

The paper calls the first future work and the second its key open problem; read
`audits/implementation_gaps.md` §4–5 for exactly what each does and does not
guarantee.

## 3. Framework adapters (real LLM + real tools)

The high-fidelity trace source instruments a **real** agent stack. Two of the three
adapters are implemented and only need an API key:

| Adapter | What it does | Needs |
|---|---|---|
| `ReActAdapter` | a real LLM decides each next tool call; the call runs against the genuine local backends; the tool's real output feeds the next prompt | an LLM endpoint |
| `MCPRouterAdapter` | the same loop dispatched over a real (TLS) socket to the MCP-style tool server, so sizes are what a router put on the wire | an LLM endpoint |
| `LangChainAdapter` | still a stub — LangChain is a heavyweight SDK with a moving callback API, and inventing it would be worse than leaving it open | the SDK + your pipeline |

```bash
cp .env.example .env          # add OPENROUTER_API_KEY or GROQ_API_KEY
python scripts/run_adapter.py --adapter react --traces-per-class 1 --max-steps 4
python scripts/run_adapter.py --adapter react --dry-run    # cost estimate, spends nothing
```

```python
from leakbench.data.adapters import ReActAdapter
from leakbench.data.llm import make_completion_fn

complete, client = make_completion_fn("openrouter")        # or "groq", "openai", "custom"
ds = ReActAdapter(complete=complete, max_steps=8).sample_dataset(taxonomy, 5)
client.usage()        # {'provider': ..., 'calls': 40, 'prompt_tokens': ...}
```

Both raise `NotImplementedError` without a `complete` callable rather than falling
back to a scripted plan — a scripted plan reported as an LLM trace would be a fiction.
The same callable drives the RQ8 neural planner
(`run_live_capture.py --planner llm`).

To write your own adapter for a stack we do not cover:

```python
from leakbench.data.adapters import FrameworkAdapter, ToolCallEvent

class MyAdapter(FrameworkAdapter):
    name = "my_stack"
    def run_task(self, class_index: int):
        # Drive your real agent on the task; for each tool call, yield:
        yield ToolCallEvent(category="search", request_size=..., response_size=..., gap=...)

ds = MyAdapter().sample_dataset(taxonomy, traces_per_class=200)
# ds now flows through observation maps + estimator unchanged.
```

**Expected input/output schema:** input is a task class index; output is an ordered
stream of `ToolCallEvent` (see `docs/data_format.md`). Sizes are bytes on the
encrypted channel; `gap` is the **inter-arrival** time since the previous call
started (it therefore includes that call's round trip — see
`docs/implementation_notes.md`).

You do not have to keep the adapter in the loop: dump its dataset once and every
stage can read it back through the ordinary real-data path.

```python
from leakbench.data.loader import dump_trace_dataset

dump_trace_dataset(ds, "/data/leakbench/traces/train/class_all.jsonl")
# then: source: real  +  data.root: /data/leakbench/traces
```

**STATUS:** `react` and `mcp_router` implemented (credential-gated); `langchain` is a
documented stub; the ingestion path is implemented and tested.

### Neural planner for the live experiment

`data/live_capture.py::LLMPlanner` is the same kind of extension point for RQ8: give
it a `complete(prompt) -> str` callable bound to your LLM endpoint and the live
harness drives a real model unchanged. Without one it raises — a stub must never be
reported as a controller. See `docs/ec2_experiment_guide.md` §5b.

## 4. Dummy baseline (tests only — never a real baseline)

`data/adapters.py::DummyAdapter` emits schema-valid events deterministically. It runs
**no model and no tool**; it exists solely to exercise the
adapter→trace→observable→estimator path in tests. It is never presented as, and must
never be reported as, a real agent stack or a real baseline.

## 5. Attacker bank (estimators, not "baselines")

`estimators/attackers.py`: RF (200 trees) + kNN (k=7) core bank; HGB + logistic
added for the robustness check (RQ6). kNN is the F-BLEAU-style Bayes-risk **floor**,
never inverted into a privacy guarantee.
