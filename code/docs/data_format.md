# Data format

LeakBench works on **traces**. This document defines the record schema and the
on-disk layout expected when you supply real data via `data.root`. LeakBench never
downloads data; you place it manually (see `ec2_experiment_guide.md`).

## The trace record (canonical schema)

A trace is a finite sequence of tool calls plus a task label. In code:
`leakbench.data.schemas.Trace` / `ToolCall`. Serialized (JSON) form, one trace:

```json
{
  "label": 6,
  "family": "personal",
  "metadata": {"source": "adapter:mcp_router", "session_id": "..."},
  "calls": [
    {"category": "calendar", "request_size": 812.0, "response_size": 3140.0, "gap": 0.0},
    {"category": "email",    "request_size": 640.0, "response_size": 1890.0, "gap": 0.42}
  ]
}
```

### Required fields per call

| Field | Type | Meaning | Constraint |
|---|---|---|---|
| `category` | string | tool category `c_i` | one of the 8 tool categories (below) |
| `request_size` | float | request bytes `s_up_i` | `> 0` |
| `response_size` | float | response bytes `s_down_i` | `> 0` |
| `gap` | float | inter-call gap `Delta_i` (seconds) | `>= 0`; first call = 0 |

### Required fields per trace

| Field | Type | Meaning |
|---|---|---|
| `label` | int | task class index (the secret `C`), `0 .. |T|-1` |
| `family` | string / null | coarse-secret family (for the 4-way secret + subgroups) |
| `metadata` | object | free-form provenance (source, seed, session id, …) |
| `calls` | array | the tool-call sequence (≥ 1; generator enforces ≥ 2) |

### Tool categories (8, canonical order)

`search, fetch, code_exec, file_io, calendar, email, db_query, calc`

### Sizes are on the **encrypted channel**

`request_size`/`response_size` are bytes as they appear on the wire the vantage
observes, **after** encryption. Argument/return plaintext is out of scope (assumed
encrypted). For the network vantage on measured data, apply `data/wire.py`.

## On-disk layout for `data.root`

```
<data.root>/
├── taxonomy.json           # optional: {name, classes:[{index,name,family}], prior:[...]}
├── train/
│   ├── class_00.jsonl       # one JSON trace record per line
│   ├── class_01.jsonl
│   └── ...
├── test/                    # optional; else LeakBench does its own stratified split
│   └── class_00.jsonl
└── splits.json              # optional: {"open_world_other": [...class indices...]}
```

- **`*.jsonl`**: newline-delimited trace records (schema above). Filenames are a
  convention; the loader reads `label` from each record, not the filename. `.ndjson`
  is accepted too, and sub-directories are searched recursively.
- **`taxonomy.json`**: if absent, the reference taxonomy is assumed and labels must be
  in `0..11`. Families must be one of `research`, `coding`, `personal`, `ops` (the
  signature-tool structure and the coarse 4-way secret are defined per family).
- **`splits.json`**: for the open-world protocol, lists which classes are folded into
  the unmonitored "other" bucket (label `-1`).

### Reading it

```python
from leakbench.data.loader import load_real_dataset

bundle = load_real_dataset("/data/leakbench/traces")
bundle.taxonomy          # from taxonomy.json, else the reference taxonomy
bundle.train, bundle.test    # test is None unless you shipped a canonical split
bundle.combined()        # train+test, for LeakBench's own stratified split
```

or, from any config, `source: real` + `data.root:` — every stage then runs on your
traces unchanged.

Validation is **strict and loud**: an unknown tool category, a non-positive size, a
negative gap, a label outside the taxonomy, or malformed JSON raises
`TraceFormatError` naming the file and line. Nothing is silently skipped, because a
dropped trace would move a leakage number with no record of why.

To write this format (e.g. to persist adapter output), use
`leakbench.data.loader.dump_trace_dataset(dataset, path)`.

## Adapter output (for framework integrations)

A framework adapter (`data/adapters.py`) instead emits a **stream of
`ToolCallEvent`** per call, which LeakBench converts to a `Trace`:

```python
ToolCallEvent(category="search", request_size=812.0, response_size=3140.0, gap=0.0)
```

See `docs/baseline_adapters.md`.

## Datasets requiring manual provisioning

The paper's high-fidelity paths depend on external components that this artifact does
**not** bundle or download:

- a real LLM planner endpoint (for a neural, non-templated planner);
- a production MCP router / tool server, or a real ReAct/LangChain deployment;
- any captured WAN traffic (the bundled live experiment is loopback only).

Provision these yourself on EC2 and expose them through an adapter. Missing pieces are
tracked in `audits/implementation_gaps.md` and `TODO_IMPLEMENTATION.md`.
