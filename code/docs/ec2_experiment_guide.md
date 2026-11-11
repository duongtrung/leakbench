# EC2 experiment guide

This repository is **EC2-ready but does not itself download data**. This guide is the
runbook for executing the full paper-scale study on a server, and for the two
experiments that need hardware this repository cannot provide on its own (a WAN link,
and a real LLM endpoint).

> The local/laptop environment is for building the repo, writing adapters, config,
> unit tests, and lightweight smoke tests. **The main evaluation belongs on the
> server**: `scripts/run_scale_study.py` is the single entry point.

---

## 0. Instance sizing

The workload is CPU-bound scikit-learn (random forests over ≤ 130-dimensional feature
matrices of ≤ 36k rows), plus a few thousand real local tool executions and a few
hundred loopback TLS sessions. It is **not** GPU work and **not** memory-bound — and
it is **small**: the paper itself reports the reference study running "end-to-end in
under three minutes on a laptop CPU", and our per-stage measurements agree (see
"Measured costs" below). Do not over-provision.

**The decision, if you just want to be told:** launch **one `c7i.2xlarge`, Ubuntu
24.04, 40 GB gp3, on-demand, in your nearest region**, run the study, archive the
output, terminate. Expect well under an hour of instance time end to end, so roughly
US$0.36 at us-east-1 on-demand pricing. Spot would save ~60% but is not worth the
interruption handling for a job this short.

| | Instance | vCPU | RAM | Storage | Notes |
|---|---|---|---|---|---|
| **Recommended** | `c7i.2xlarge` | 8 | 16 GiB | 40 GB gp3 | comfortable headroom; the full study is minutes, not hours |
| Cheapest sensible | `t3.large` | 2 | 8 GiB | 30 GB gp3 | fine — a few-minute job does not exhaust burst credits |
| ARM alternative | `c7g.2xlarge` | 8 | 16 GiB | 40 GB gp3 | Graviton3, cheaper; manylinux aarch64 wheels exist for numpy/scipy/scikit-learn |
| Only if you scale up | `c7i.4xlarge` | 16 | 32 GiB | 50 GB gp3 | for much larger sweeps (many more seeds, `\|T\|` ≫ 24, a 10× open-world) |

Two optional add-ons have a different shape and are costed separately:

| Add-on | What it needs | Cost / time |
|---|---|---|
| WAN variant of RQ8 (§5a) | **no extra instance** — one SG rule, and a client machine far from the region | free; the capture takes ~20 min at WAN RTT |
| Neural planner for RQ8 (§5b) | an LLM API key, **not** more CPU | latency- and rate-limit-bound, not compute-bound — see §5b for the sizing that keeps it under an hour |

### Measured costs (Apple M-series laptop, per stage, at paper sizes)

| Work | Cost |
|---|---|
| generate 7,200 traces (600/class) | 0.3 s |
| featurize + estimate one vantage (B=200) | 0.8 s |
| measured harness, 130 traces/class incl. real tool execution | 1.0 s per seed |
| open-world at 24,000 traces (generate + featurize + RF fit) | 1.9 s per base rate |
| DP shaping, one ε config (transform + estimate) | 1.1 s |
| generative shaping, one pass over 7,200 traces | 1.9 s |
| live capture, 6 classes × 70 sessions over real TLS | ~30 s per seed |

Summing the plan gives single-digit minutes for the whole study. Budget an hour of
instance time and you have a large margin.

Configuration details:

- **AMI:** Ubuntu Server 24.04 LTS (x86_64, or arm64 for `c7g`). Amazon Linux 2023
  also works (Python 3.11). Python **3.11+** is required.
- **Storage:** 50 GB gp3 (default 3000 IOPS / 125 MB/s is ample). Results are < 100 MB;
  the measured harness writes only small short-lived temp files under `/tmp`.
- **Networking:** **no inbound ports are required** for the single-host study — the
  live experiment (RQ8) binds `127.0.0.1` only. (The WAN variant in §5a is the
  exception: one port, open to the client's IP alone.) Outbound HTTPS is needed once,
  for `pip install`. Prefer SSM Session Manager; if you use SSH, restrict port 22 to
  your own IP.
- **IAM:** none needed (add `AmazonSSMManagedInstanceCore` if you use Session Manager).
- **Spot:** fine — the study checkpoints per stage (one JSON per stage), and
  `--only` resumes the stages you still need.
- **Cost:** a full run is minutes of on-demand time on any instance in the table.

Sizing rationale: peak resident set is well under a GB (largest feature matrix is
24,000 × 127 float64 ≈ 25 MB; the random forests dominate at a few hundred MB per
fit). Extra vCPUs help a little because `RandomForestClassifier` runs with
`n_jobs=-1`, but at these sizes the fits are seconds either way, so cores buy less
than the table above might suggest.

The **WAN experiment** (§5a) needs **no second instance** — see §5a for why two hosts
in one region are the wrong topology. It needs one inbound security-group rule on the
instance you already have, and a client machine that is genuinely far from the region.

## 1. Bootstrap

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-dev git openssl
git clone <your copy of this repo> leakbench && cd leakbench
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
pytest -q                      # sanity: the whole suite runs on synthetic/local data
python -m leakbench.cli info
```

`openssl` matters: the live experiment (RQ8) uses it to mint a throwaway self-signed
certificate so the capture runs over **real TLS**. Without it (and without the
optional `cryptography` package) the run falls back to plaintext TCP and records
`capture_mode = tcp_captured_proxy` — the result is still real, but the TLS
framing is then a model rather than a capture, and every output says so.

## 2. The full study (one command)

```bash
# check the whole pipeline end-to-end in well under a minute first
python scripts/run_scale_study.py --quick --output outputs/scale_quick

# then the real thing
nohup python scripts/run_scale_study.py --output outputs/scale_study > scale.log 2>&1 &
tail -f scale.log
```

Outputs land in `outputs/scale_study/`: one JSON per stage (each with full provenance
and its wall-clock time), plus `summary.json` and a human-readable `report.md`.

Useful flags: `--only vantage,live`, `--skip dp_sweep`, `--seeds 5`,
`--live-per-class 70`, `--openworld-total 24000`, `--draws 8`, `--no-tls`.

### Coverage: every result in the paper → the stage that produces it

| Paper object | Stage |
|---|---|
| Tbl. vantage, Fig. vantage (RQ1) | `vantage` |
| Tbl. coarse (4-way family vs 12-way class) | `coarse` |
| Fig. ablation (RQ2) | `ablation` |
| RQ2 feature-importance corroboration (§Eval prose) | `feature_importance` |
| Tbl. subgroup, Fig. subgroup (RQ3) — length bands **and** within-family 3-way | `subgroup` |
| Tbl. defense, Fig. tradeoff, main_tradeoff_plot (RQ4) | `defenses` |
| Fig. validation (RQ5) | `validation` |
| Fig. separability + App. "Separability sweep (numbers)" | `separability` |
| Fig. scalability, cost_or_scalability_plot | `scalability` |
| §Eval case study (deployed attacks on common traces) | `projections` |
| Tbl. measured, Fig. measured (RQ6) + paired *t*-test | `measured_multiseed` |
| RQ6 "Robust to the wire model" (3×3 sweep) | `wire_sweep` |
| Tbl. robustness (four attacker families) | `robustness` |
| Tbl. openworld + within-taxonomy F1 | `openworld` |
| Tbl. taxablation (RQ7) | `taxonomy_sweep` |
| Tbl. live (RQ8) | `live` |
| §Eval "Variance across generator draws" (8 redraws) | `generator_variance` |
| §Eval "Non-uniform prior" (Zipf vs uniform) | `prior_sensitivity` |
| App. "Estimator agreement" | `estimator_agreement` |
| §Discussion open problem: ε → advantage | `dp_sweep` *(beyond the paper)* |
| Estimator convergence (F-BLEAU style) | `learning_curve` *(beyond the paper)* |

Descriptive tables (`t_taxonomy`, `t_observables`, `t_wfmap`, `t_related`, `t_novelty`,
`claim_audit`, `citation_audit`, `evaluation_plan`, `ablation_plan`,
`baseline_comparison`) state the design rather than a measurement, so they have no
runner; `t_projection` is a mapping table whose *numbers* come from `projections`.

### Reading the result against the paper

Three stages report a **flag rather than an assumption**, because they test claims the
paper makes in prose:

- `separability.advantage_monotone` / `mi_monotone` — the paper claims both curves rise
  with the signature up-weight.
- `estimator_agreement.agreement` — the paper claims RF and k-NN agree on the
  *ordering* of vantages and defenses.
- `measured_multiseed.ordering_steps[*].holds_in_seeds` — the paper claims 5/5.

If any comes back false at full scale, that is a finding about the paper, not a bug in
the harness. Check it before writing the number up.

## 3. Individual stages

```bash
python scripts/run_experiment.py --config configs/default.yaml  --stage all
python scripts/run_experiment.py --config configs/measured.yaml --stage measured_multiseed
python scripts/run_experiment.py --config configs/measured.yaml --stage wire_sweep,robustness
python scripts/run_experiment.py --config configs/default.yaml  --stage openworld,dp_sweep
python scripts/run_taxonomy_sweep.py --sizes 8 12 16 24 --traces-per-class 600
python scripts/run_live_capture.py --traces-per-class 70 --seeds 2
```

`--stage all` runs the RQ1–RQ6 core; `--stage everything` adds the heavier sweeps.

## 4. Real agent traces (framework-adapter path) — **requires provisioning**

To measure a real stack instead of the generator:

1. **Stand up the stack** you want to measure (a ReAct loop, an MCP router + tool
   server, or a LangChain pipeline) with a real LLM endpoint. Credentials/API keys are
   **your** responsibility and must never be committed.
2. **Implement an adapter** (`docs/baseline_adapters.md`) that yields a
   `ToolCallEvent` per tool call, or dump traces to disk in the
   `docs/data_format.md` layout.
3. Point the config at them and run any stage:

```yaml
# configs/my_real.yaml
source: real
data:
  root: /data/leakbench/traces      # train/*.jsonl (+ optional test/, taxonomy.json)
```

```bash
python scripts/run_experiment.py --config configs/my_real.yaml --stage vantage,ablation,subgroup
```

The loader (`leakbench/data/loader.py`) validates every record against the trace
schema and fails loudly with file and line on any violation — a silently dropped trace
would change a leakage number with no record of why. **Never auto-download** models or
benchmarks; place them yourself and point the config at them.

## 5. The two variants that need hardware this repo cannot bundle

### 5a. WAN link for RQ8

The bundled live experiment is a real capture over a real TLS connection, but by
default the link is **loopback**. The code path for a remote server is implemented and
tested; what follows is how to get a link worth measuring.

**Do not use two instances in the same region.** Two EC2 hosts in different AZs of one
region sit about **1 ms** apart — barely distinguishable from the ~0.05 ms loopback
this is meant to improve on, so the experiment would cost money and change nothing.
Put the **client on a machine that is genuinely far from the server**: a laptop on a
normal connection is ideal. Measured from a developer laptop to `us-east-1`:

| link | RTT |
|---|---|
| loopback | ~0.05 ms |
| two EC2 hosts, different AZ, same region | ~1 ms |
| laptop in Vietnam → `us-east-1` (**measured**) | **249 ms** |

That is nearly four orders of magnitude of RTT, which is the point of the experiment.
The 249 ms figure is from the packet capture of the run reported in the paper: median
over 2,932 request intervals, of which the server's own service time is 1.0 ms.

**Server side (the EC2 instance you already have):**

```bash
python scripts/run_tool_server.py --bind 0.0.0.0 --port 8443
#   -> prints the listening address and writes leakbench_server_cert.pem
```

Open **one** inbound rule on the instance's security group: Custom TCP, port 8443,
source `<your client's public IP>/32`. Never `0.0.0.0/0`.

**Client side (your machine):**

```bash
scp -i <key.pem> ubuntu@<server>:~/leakbench_server_cert.pem .
python scripts/run_live_capture.py --traces-per-class 70 --seeds 2 \
    --server-host <server public IP> --server-port 8443 \
    --server-ca leakbench_server_cert.pem
```

The capture proxy runs on the **client**, so it sits on the client's uplink and records
the bytes as they cross the real link — precisely the network vantage the paper models.
Everything else (tools, protocol, planner, estimator) is unchanged; only the RTT
distribution and the segmentation change, which is what the WAN test measures.

#### If you cannot open an inbound port

You may not control the security group, or policy may forbid opening anything but SSH.
The run in the paper was done this way, so the fallback is documented rather than
hypothetical: tunnel the connection over the SSH port you already have.

```bash
# server (cloud): bind loopback only -- no inbound rule needed
python scripts/run_tool_server.py --bind 127.0.0.1 --port 8443 \
    --cert-out /tmp/leakbench_ca.pem

# client (your machine): one dedicated tunnel, then the capture
scp -i <key.pem> ubuntu@<server>:/tmp/leakbench_ca.pem .
ssh -i <key.pem> -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -L 18443:127.0.0.1:8443 ubuntu@<server> &
python scripts/run_wan_capture.py --traces-per-class 70 \
    --server-port 18443 --server-ca leakbench_ca.pem
```

**What this gives you, and what it does not.** Every call genuinely crosses the
internet, so propagation delay, jitter and congestion are in the timing channel, and
TLS is end-to-end between client and server so the records the proxy captures are
real. What is *not* raw is TCP segmentation: the stream rides inside SSH, so on-wire
packet boundaries are the tunnel's framing rather than the TLS stream's own. Say so
when you report it.

To show the link really is a wide-area link rather than a relabelled loopback run,
capture on the server's interface and summarize it:

```bash
# server: pin the filter to the tunnel's own connection, not all of :22
ss -tn state established sport = :22          # find the client port that persists
sudo tcpdump -i <iface> -s 96 -w /tmp/wan.pcap "tcp and host <client IP> and port <port>"

# afterwards, on either host
python scripts/analyze_wan_pcap.py /tmp/wan.pcap --client-ip <client IP>
```

The summary reports the two things that settle it: the largest packet on the path
(1452 B against a 9001 B interface MTU, i.e. path-MTU discovery clamped it to a 1500 B
internet path) and the round trip measured from the server's NIC (249 ms, against
1.0 ms of server service time).

The run records `link: remote` and drops "loopback not WAN" from its gap list — but
only when the address really is remote: pointing `--server-host` at 127.0.0.1 still
records `link: loopback`, because the metadata describes the measurement, not the flag.
Record the client location / server region alongside the output.

Expect the capture itself to be slow: at ~0.5 s RTT and ~6 calls per session, 420
sessions take roughly 20 minutes. That is the experiment working, not hanging.

### 5b. Neural LLM planner for RQ8

**Sizing decision.** This variant is bound by API latency and rate limits, not by CPU,
so it does not scale the way the rest of the study does. Run it as a *second*,
smaller RQ8 rather than as a replacement:

| Run | Command | Calls | Wall clock |
|---|---|---|---|
| RQ8 as the paper states it (programmatic planner, full size) | `--traces-per-class 70 --seeds 2` | 0 | ~1 min |
| RQ8 gap-closing addition (neural planner) | `--planner llm --traces-per-class 20 --seeds 1 --max-steps 8` | ≤ 960 | ~10 min on a paid endpoint; ~35 min on a free tier at 30 req/min |

Reporting both is stronger than either alone: the programmatic run gives the
full-size numbers on the same footing as the rest of the study, and the neural run
shows the laws survive a real controller — including a **timing channel three orders
of magnitude more realistic** (median inter-arrival gap ~0.33 s with a real model
versus ~0.00004 s with a scripted planner, because a real controller thinks between
calls; see `audits/qa_results.md`).

Budget note: on a free tier the client's 429 backoff will do the pacing for you, so
the run is slow but safe. Check `client.usage()` in the result file before scaling up.


`leakbench.data.live_capture.LLMPlanner` implements the same `(tool, scale)` step
interface as the programmatic controller and drives the live harness unchanged; it
needs a `complete(prompt) -> str` callable bound to your LLM endpoint:

```python
from leakbench.data.live_capture import LiveCaptureHarness, LLMPlanner

def complete(prompt: str) -> str:
    ...  # your endpoint; returns e.g. '{"tool": "search", "scale": 1.0}'

ds = LiveCaptureHarness(planner=LLMPlanner(complete=complete)).capture_dataset(70)
```

The dataset metadata then records `planner_kind = neural_llm`, and the
"programmatic not neural planner" gap disappears from the recorded gap list. Without a
`complete` callable the planner **raises** rather than silently falling back — a stub
must never be reported as a real controller.

## 6. Datasets requiring manual download / authentication / registration

LeakBench's own harness needs **no** external dataset. If you integrate one, record
its license, the manual download/authentication steps, its directory layout, and the
adapter that maps it to the `Trace` schema, and track it in
`audits/implementation_gaps.md` and `TODO_IMPLEMENTATION.md`.

## 7. Provenance and archiving

Every result file embeds LeakBench version, Python version, platform, seed, taxonomy
name + hash, and the full config. Archive the whole output directory next to the
config:

```bash
tar czf leakbench-$(date -u +%Y%m%dT%H%M%SZ).tar.gz outputs/scale_study configs scale.log
aws s3 cp leakbench-*.tar.gz s3://<your-bucket>/leakbench/
```
