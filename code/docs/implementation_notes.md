# Implementation notes

How the paper was translated into code. Read alongside
`audits/paper_implementation_spec.md` (what) and `audits/math_to_code_audit.md`
(why the math is implemented the way it is).

## Design: the five-stage harness

The package mirrors the paper's Fig. "architecture" pipeline:

```
taxonomy (T)  →  trace source  →  observation (O_v) [+ defense]  →  estimator bank  →  report
   Stage 1         Stage 2              Stage 3                        Stage 4          Stage 5
```

Each stage is an independent module behind a small interface (paper §Impl: "adding a
vantage, a defense, or an estimator is local").

- **Stage 1 — `taxonomy.py`.** Fixes the secret distribution (principle P1). The
  reference taxonomy is content-hashed so results are keyed by version.
- **Stage 2 — `data/`.** All trace sources emit the same `Trace` record
  (`schemas.py`), so an estimator trained on one runs unchanged on another. Sources:
  `generator.py` (calibrated `G`), `measured.py` (real local tools),
  `live_capture.py` (real sockets), `adapters.py` (framework hooks; stubs).
- **Stage 3 — `observables/` + `defenses/`.** Observation maps are pure functions of
  a trace; defenses are pure trace→trace transforms applied *before* observation, so
  the same estimator scores defended and undefended traces on one axis.
- **Stage 4 — `estimators/`.** The attacker bank (RF, kNN, +HGB/LR for robustness)
  and the standardized leakage-estimation algorithm.
- **Stage 5 — `experiments/`.** RQ runners produce a *profile* (per vantage, per
  subgroup, with CIs), written to JSON with provenance; plots are built from those files.

## Separating observable from estimator (principle P2)

The observation map is a property of the **vantage**; the attacker is a property of
the **estimator**. Conflating them ("accuracy of our CNN") is what makes prior
numbers incomparable. In code this is a hard boundary: `observables/maps.py` never
imports an estimator, and `estimators/leakage.py` takes an already-featurized `(X,y)`.

## The refinement order and the falsifiability guard

`provider_map` literally concatenates `platform_map` then `network_map` (verified by
`test_provider_map_contains_shallower_maps`), so `O_net ⪯ O_prov` and
`O_plat ⪯ O_prov` hold **by construction**. `run_vantage` then checks
`Adv_prov >= max(Adv_net, Adv_plat)` and emits a WARNING if violated — it does *not*
reorder. A violation means an estimator/feature bug (or, as we observed at very small
sample sizes in the live study, insufficient data for the high-dimensional provider
space to converge — the paper flags this too). The net-vs-plat step is **incomparable**
and reported empirically, never enforced.

## Estimator honesty

- Report the **strongest** attacker (min test error = tightest lower bound).
- Advantage clamped to `[0,1]`; a finite-sample classifier worse than guessing yields
  advantage 0 (paper's prescription).
- MI is the confusion-matrix `I(C;Ĉ)`, a genuine **lower bound** on `I(C;O_v)`;
  reported as "at least this much leaks", never inverted into a privacy upper bound.
- Bootstrap CIs use the point estimate's own estimator across resamples (not re-argmin
  per resample, which would bias the interval upward) — see `math_to_code_audit.md`.

## Wire models (measured harness)

Real tools emit *true* byte sizes and timing. For the **network** vantage we apply
(1) TLS-1.3 record framing and (2) an HTTP/2 wire model (record coalescing prob `p`,
cross-traffic noise `σ`, MTU quantization) because a real on-path observer sees
multiplexed bursts, not clean per-call boundaries. Platform/provider vantages see the
pre-wire trace (they sit above the transport). This is a **model**, disclosed as such;
the live experiment (RQ8) instead captures real socket bytes.

## Inter-call gap semantics (`Delta_i`)

`Delta_i` is the **inter-arrival** gap: wall-clock time from the *start* of call `i-1`
to the *start* of call `i`. That is the quantity `O_net`'s "inter-arrival histogram"
is named after, and it is what an on-path observer can time between consecutive
request bursts, so it contains the previous call's real round trip and tool service
time. Measuring end-of-previous-call → start-of-this-call instead would capture only
the harness's own Python loop overhead (~2 µs here) and would report a timing channel
that is an artifact of the measurement code. See `audits/math_to_code_audit.md`.

## The live experiment (RQ8): what is captured vs. modeled

- **Captured:** the network observable. A real TCP **capture proxy** sits between the
  MCP-style client and the tool server, relays bytes, and records the size of every
  kernel-delivered chunk. TLS record framing and TCP coalescing therefore emerge from
  the kernel stack.
- **Real TLS:** the server runs a real TLS handshake using a throwaway self-signed
  certificate minted locally (`data/tlsutil.py`, via `cryptography` or the `openssl`
  binary). The proxy sees ciphertext only — like the adversary it stands in for. The
  handshake bytes precede the first step marker and are excluded from the trace; their
  mean is recorded in the dataset metadata.
- **Real branching:** the controller reads the tools' actual outputs (documents the
  corpus search matched, rows SQLite returned, the parity of the integer the executed
  Python computed) and branches on them, so plan length and composition are functions
  of genuine results.
- **Recorded, not assumed:** if no certificate material can be provisioned, the run
  captures plaintext TCP, applies TLS framing as a model, and records
  `capture_mode = tcp_captured_proxy`. The mode always reflects what ran.
- **Link:** loopback by default. A two-host run (`scripts/run_tool_server.py` on the
  server, `--server-host/--server-port/--server-ca` on the client) puts the capture
  proxy on the client's uplink so the captured bytes are the ones that crossed the real
  link. The recorded `link` is resolved from the address, never from the flag.
- **Still limited:** the default planner is programmatic, not a neural LLM
  (`LLMPlanner` is the extension point).

## Defenses beyond the paper's artifact

`gen-shaping` (Surakav analogue) and `dp-shaping` (NetShaper analogue) implement two
things the paper names as future work and as its key open problem. Both are scored by
the ordinary defense path, so they sit on the same privacy–utility axis as the WF
toolbox. `dp_advantage_ceiling` turns a DP budget into an advantage bound
(`Adv <= V_prior (e^{ε·d} - 1)/R_π`), and `run_dp_sweep` checks the measured value
against it at every budget — the sweep is a falsifiability check, not a decoration.
The honest headline is negative: the bound is unconditional only under an all-pairs
adjacency, which is expensive; under NetShaper's cheaper per-window adjacency the
task-level ceiling is vacuous. Scope and caveats: `audits/implementation_gaps.md` §4–5.

## What is deliberately synthetic / stubbed

- The calibrated generator is a stand-in for measured traces.
- Framework adapters against production stacks are stubs (`NotImplementedError`) with a
  deterministic dummy for tests. Their *output* format is ingestible: dump traces to
  `data.root` and every stage reads them via `source: real` (`data/loader.py`).
- The live planner is programmatic by default (with `LLMPlanner` as the wired
  extension point), and the link is loopback unless you run the two-host variant.

All of these are recorded in `audits/implementation_gaps.md` and `TODO_IMPLEMENTATION.md`.

## Why magnitudes differ from the paper's tables

The demo uses small trace counts for speed, and the generator/measured distributions
are reconstructed from the paper's described parameters, not its exact private draw.
The paper itself states magnitudes are "calibrated-illustrative" and the **orderings
and mechanisms** are the robust claims. The repository reproduces those orderings
(NET < PLAT < PROV; size dominates timing; personal most identifiable; only size
padding neutralizes the channel, at order-of-magnitude cost).
