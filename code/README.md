# LeakBench

**A reproducible benchmark for task leakage from LLM agent tool-call metadata.**

The benchmark harness behind the paper *"What the Tools See: Measuring Task Leakage
from LLM Agent Tool-Call Metadata at Three Web Vantages"*. The code covers more
experiments than the 4-page paper reports (for example defenses). LeakBench measures how much an LLM
agent's tool-use **trace metadata** (which tools, in what order, how many times,
with what timing, at what message sizes) leaks the user's **task**, even when every
payload is encrypted.

> ⚠️ **No paper-level experimental results are claimed by this repository unless
> real benchmark outputs are generated and their provenance is recorded.** All demo
> numbers are computed from a calibrated *generator* or *local* tools and are
> labeled "synthetic smoke-test output, not a paper result." Absolute magnitudes are
> calibrated-illustrative; the paper's claims are the *orderings and mechanisms*.

> ⚠️ **This repository does not automatically download datasets or run paper-level
> experiments.** Real experiments are intended to be executed on EC2 after the
> required datasets, benchmarks, models, and credentials are manually provisioned
> (see `docs/ec2_experiment_guide.md`).

---

## What the paper is (and is not)

It is a **measurement paper**, not a new attack. It argues agent-trace privacy needs
the treatment website fingerprinting received a decade ago: a shared threat model, a
common observable, and a principled leakage estimator. Its four contributions map to
this codebase:

- **C1 — Unified observable-feature model** (`leakbench.observables`): three
  adversarial *vantages* — **network (NET)**, **platform (PLAT)**, **providers
  (PROV)** — and a vantage-indexed observation map `O_v`.
- **C2 — Reproducible benchmark + harness** (`leakbench.taxonomy`, `leakbench.data`):
  a fixed 12-class task taxonomy, three interchangeable trace sources (calibrated
  generator, measured local-tool harness, live captured-traffic adapter), and a
  closed-/open-world protocol borrowed from WF.
- **C3 — Standardized leakage estimator** (`leakbench.estimators`,
  `leakbench.metrics`): Bayes-security advantage + residual Bayes risk + MI
  lower bound, with bootstrap CIs and subgroup/vantage resolution, grounded in
  quantitative information flow.
- **C4 — Defense study** (`leakbench.defenses`): the WF defense toolbox (padding,
  cover calls, constant-rate scheduling, shuffling) on a common privacy–utility axis.

## Paper → code map

| Paper object | Code |
|---|---|
| Task taxonomy `T` (`tax-v1/12c-4f`, `H(C)=3.585` bits) | `leakbench/taxonomy.py` |
| Agent trace `rho = <(c_i, s_up_i, s_down_i, Delta_i)>` | `leakbench/data/schemas.py` |
| Calibrated generator `G` (Dirichlet/Poisson/log-normal) | `leakbench/data/generator.py` |
| Observation maps `O_net`, `O_plat`, `O_prov` + refinement order | `leakbench/observables/maps.py` |
| Prior attacks as projections (Whisper Leak, keylogger, …) | `leakbench/observables/projections.py` |
| Bayes risk / advantage / MI (E3, E4, E7) | `leakbench/estimators/leakage.py`, `leakbench/metrics/information.py` |
| g-leakage view (E5) | `leakbench/metrics/information.py::g_leakage` |
| Estimation algorithm (RF + kNN, bootstrap, subgroups) | `leakbench/estimators/leakage.py::estimate_leakage` |
| Defenses (pad-512, shuffle, constant-rate, cover, pad-max, full) | `leakbench/defenses/transforms.py` |
| Measured tool-execution harness (RQ6) | `leakbench/data/measured.py`, `leakbench/data/wire.py` |
| Ground-truth validation channel (E8, RQ5) | `leakbench/validation/ground_truth.py` |
| Live captured-traffic experiment (RQ8) | `leakbench/data/live_capture.py`, `leakbench/data/tlsutil.py` |
| Framework adapters (ReAct / MCP router) | `leakbench/data/adapters.py` + `leakbench/data/llm.py` |
| Real-trace ingestion (`data.root`) | `leakbench/data/loader.py` |
| Generative shaping (Surakav analogue) | `leakbench/defenses/generative.py` |
| DP protocol shaping + ε→advantage bound (NetShaper) | `leakbench/defenses/dp_shaping.py` |

The full mapping is in [`audits/paper_to_code_traceability.csv`](audits/paper_to_code_traceability.csv).

## Installation

Python 3.11+. No GPU, no internet access required; all tests run on synthetic/local data.

```bash
pip install -e .            # or: make install
pip install -e ".[dev]"     # + pytest, ruff, mypy
```

## Quick start — synthetic demo

```bash
python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml
# or: make demo
```

This runs taxonomy → generator → 3 observation maps → estimator bank →
per-vantage leakage profile, plus the feature ablation (RQ2), subgroups (RQ3),
defense frontier (RQ4), and estimator validation (RQ5). Outputs (JSON + PNG,
**all labeled synthetic**) land in `outputs/synthetic_demo/`.

Inspect the harness:

```bash
python -m leakbench.cli info        # taxonomy summary, H(C), vantages
python -m leakbench.cli validate    # estimator vs. the analytic channel
```

## Running experiments

The **main evaluation belongs on a server** — see `docs/ec2_experiment_guide.md` for
instance sizing (`c7i.2xlarge` is ample; no inbound ports; the study is minutes of
CPU) and the bootstrap. One command runs everything at the paper's sizes:

```bash
python scripts/run_scale_study.py --quick --output outputs/scale_quick   # ~1 min check
python scripts/run_scale_study.py --output outputs/scale_study           # paper sizes
```

It writes one JSON per stage (with provenance and wall-clock time), plus
`summary.json` and a readable `report.md`.

Individual stages:

```bash
# generator study (RQ1-RQ5), one stage, several, or the core set:
python scripts/run_experiment.py --config configs/default.yaml --stage vantage
python scripts/run_experiment.py --config configs/default.yaml --stage all

# measured harness with real local tools (RQ6) + its robustness sweeps:
python scripts/run_experiment.py --config configs/measured.yaml --stage measured_multiseed
python scripts/run_experiment.py --config configs/measured.yaml --stage wire_sweep,robustness

# open-world protocol and the DP-shaping epsilon sweep:
python scripts/run_experiment.py --config configs/default.yaml --stage openworld,dp_sweep

# taxonomy-size sweep (RQ7):
python scripts/run_taxonomy_sweep.py --sizes 8 12 16 24 --traces-per-class 600

# live experiment with captured TLS loopback traffic (RQ8):
python scripts/run_live_capture.py --traces-per-class 70 --seeds 2

# regenerate plots from result files:
python scripts/make_plots.py --input outputs/synthetic_demo --output outputs/synthetic_demo/plots
```

Stages: `vantage` (RQ1), `ablation` (RQ2), `subgroup` (RQ3), `defenses` (RQ4),
`validation` (RQ5), `measured` / `measured_multiseed` / `wire_sweep` / `robustness`
(RQ6), `projections` (deployed-attack case study), `openworld`, `dp_sweep`,
`learning_curve`. `--stage all` runs the RQ1–RQ6 core; `--stage everything` runs all
of them; a comma-separated list runs exactly those.

## Using a real LLM (optional, credential-gated)

Two paths take a real model. Both need an API key for any OpenAI-compatible endpoint
(OpenRouter, Groq, OpenAI, or your own); put it in a **gitignored** `.env`
(`cp .env.example .env`). Nothing else in the repository ever leaves the machine.

```bash
# a real LLM decides each tool call; the calls run against the genuine local tools
python scripts/run_adapter.py --adapter react --traces-per-class 1 --max-steps 4
python scripts/run_adapter.py --adapter react --dry-run   # cost estimate, spends nothing

# RQ8 with a neural controller instead of the programmatic one
python scripts/run_live_capture.py --planner llm --llm-provider openrouter
```

Result files record the provider, model, and call/token counts — never the key. Both
paths **raise** without an endpoint rather than falling back to a scripted plan.

## Real-data usage

LeakBench **never downloads data**. To run on real agent traces, either (a) implement
a framework adapter for your stack (see `docs/baseline_adapters.md`) so it emits the
`Trace` schema, or (b) place trace files on disk in the layout of
`docs/data_format.md` and point `data.root` at them:

```yaml
source: real
data:
  root: /data/leakbench/traces     # train/*.jsonl (+ optional test/, taxonomy.json, splits.json)
```

Every stage then runs on your traces unchanged. The loader validates each record
against the trace schema and fails with the offending file and line rather than
skipping it silently. Full EC2 instructions are in `docs/ec2_experiment_guide.md`.

## Folder structure

```
LeakBench/
├── src/leakbench/
│   ├── taxonomy.py              # Stage 1: task taxonomy T (the secret)
│   ├── data/                    # Stage 2: trace sources + schema + wire models
│   │   ├── schemas.py           #   Trace / ToolCall / TraceDataset
│   │   ├── generator.py         #   calibrated generator G (RQ1-5, 7) + open-world "other"
│   │   ├── measured.py          #   real local-tool harness (RQ6)
│   │   ├── wire.py              #   TLS framing + HTTP/2 wire model
│   │   ├── live_capture.py      #   live captured-traffic experiment (RQ8)
│   │   ├── tlsutil.py           #   throwaway self-signed TLS material for RQ8
│   │   ├── loader.py            #   real-trace ingestion from data.root
│   │   ├── adapters.py          #   framework adapter interface + stubs + dummy
│   │   └── corpus.py            #   in-memory corpus + SQLite fixtures
│   ├── observables/             # Stage 3: C1 observation maps + projections
│   ├── defenses/                # Stage 3: C4 defense transforms
│   │   ├── transforms.py        #   WF toolbox (pad / shuffle / constant-rate / cover)
│   │   ├── generative.py        #   Surakav-analogue learned shaper
│   │   └── dp_shaping.py        #   NetShaper-analogue DP shaping + eps ceiling
│   ├── estimators/              # Stage 4: C3 attacker bank + leakage algorithm
│   ├── metrics/                 #   MI, g-leakage, open-world scoring
│   ├── validation/              #   analytic ground-truth channel (RQ5)
│   ├── experiments/             # Stage 5: RQ runners, demo, plotting, config
│   ├── utils/                   #   seeding, logging, IO, provenance
│   └── cli.py
├── configs/                     # default / synthetic_demo / measured / template
├── scripts/                     # run_scale_study, run_experiment, run_live_capture, ...
├── tests/                       # pytest suite (synthetic/local only)
├── examples/                    # synthetic example data + config
├── docs/                        # implementation notes, data format, adapters, repro, EC2
├── audits/                      # spec, traceability, math audit, gaps, integrity, QA
├── outputs/                     # experiment outputs (gitignored)
└── TODO_IMPLEMENTATION.md
```

## Implemented algorithms

- The **standardized leakage-estimation algorithm** (stratified split → estimator
  bank → min-error Bayes risk → advantage → confusion-matrix MI → bootstrap CI →
  per-subgroup profile).
- **Observation maps** `O_net`, `O_plat`, `O_prov` and the refinement partial order
  (PROV refines both; NET/PLAT incomparable), with a **falsifiability guard** that
  surfaces (never hides) any `Adv_prov < max(Adv_net, Adv_plat)` violation.
- **Calibrated generator** `G` and the **analytic symmetric validation channel**
  (closed-form MI and Bayes risk).
- **WF defense toolbox** with per-defense bandwidth/latency accounting, plus a
  **generative shaper** (Surakav analogue) and a **DP shaping layer** (NetShaper
  analogue) whose `epsilon` maps onto the advantage axis via
  `Adv <= V_prior (e^{ε·d} - 1)/R_π`, checked against the measured value at every budget.
- **Prior attacks as projections** (keylogger, Whisper Leak, AgentPrint, privacy
  side-channels) reproduced on common traces.
- **Open-world protocol** with an out-of-taxonomy "other" (random tool mixes, no
  signature) at configurable monitored base rates.

## Metrics

Attacker advantage `Adv_v ∈ [0,1]` (+ bootstrap CI), residual Bayes risk `R*_v`, MI
lower bound `Î_v` (bits), multiplicative g-leakage (identity + family gains), open-world
precision/recall/F1, and disparate-vulnerability subgroup profiles.

**Honesty invariants** (enforced in code and docs): advantage and MI are leakage
**lower bounds** ("at least this much leaks"), never an upper bound on privacy; the
*strongest* attacker is reported; a single attacker's accuracy is never inverted into
a privacy guarantee; the net-vs-plat step is reported empirically, never enforced.

## Baselines

- **Defense baselines:** implemented directly (Table "defense").
- **Deployed attacks:** reproduced as feature-subset *projections* of the unified
  observable (the paper's comparators), not external code.
- **Attacker floor:** kNN (F-BLEAU-style) as a conservative Bayes-risk estimator;
  RF/HGB/LR as strong learners.
- **Framework adapters:** `react` and `mcp_router` run a **real LLM driving real
  tools** (credential-gated; they raise without an endpoint rather than substituting a
  scripted plan). `langchain` stays a documented stub. The deterministic `DummyAdapter`
  is tests-only and is **never** presented as a real baseline.

## Configuration guide

Configs are YAML (`configs/`). Key fields: `source` (`generator|measured|live`),
`num_classes`, `traces_per_class`, `seed`, `signature_upweight` (generator
separability knob), `overlap` (measured-harness confusability), `bank`, `bootstrap`,
`test_size`, `vantages`, `defenses`, `prior` (`uniform|zipf`), `output_dir`, and
`data.root` (real data path; never auto-downloaded). See
`configs/experiment_template.yaml`.

## Reproducibility

All randomness derives from a single logged seed (reference `20270214`). Every result
file carries run provenance (version, platform, seed, taxonomy hash, config) and is
keyed by taxonomy hash so results are diff-able across runs. See
`docs/reproducibility.md`.

## Limitations & known gaps

- The generator is a **stand-in** for measured traces (plausible orders of magnitude,
  not a specific deployed system).
- The measured harness uses a **templated planner** (fixed program per class).
- The live experiment runs over **real TLS** with the wire **captured** at a real TCP
  proxy, and its planner branches on **real tool outputs**. The link is **loopback**
  unless you run the two-host variant (`scripts/run_tool_server.py` + `--server-host`),
  and the planner is **programmatic**, not a neural LLM (`LLMPlanner` is the wired
  extension point). If no certificate material can be provisioned locally, the run
  falls back to captured plaintext TCP with modeled TLS framing and **records that
  mode** in every trace.
- MI is only ever a **lower bound**.
- The LLM-driven adapters and the neural RQ8 planner are implemented but
  **credential-gated**, and the tools they drive are LeakBench's local backends, not a
  production tool server (disclosed in every run). Generative shaping and DP protocol
  shaping are implemented, with their scope and the negative result about
  ε-vs-advantage written out in `audits/implementation_gaps.md`.
- No paper-scale numbers have been produced yet: the study is meant to run on a
  server, and only a reduced-size end-to-end check has been executed
  (`audits/qa_results.md`).

## Tests

```bash
pytest -q          # 102 tests, synthetic/local data only, no network beyond loopback
make test
```

QA (dev extras); results recorded in `audits/qa_results.md`:

```bash
ruff check .       # All checks passed!
mypy src           # Success: no issues found in 40 source files
```

## Reproduce the demo outputs

```bash
make demo          # writes outputs/synthetic_demo/*.json + plots (labeled synthetic)
```

## License

MIT (re-implementation). The paper and its original artifact belong to their authors;
this repository reconstructs the described methods from the paper text.
