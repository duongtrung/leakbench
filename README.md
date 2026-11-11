# What the Tools See: artifact

Code, stored traces and results for the WWW 2027 short paper *What the Tools See: Measuring Task
Leakage from LLM Agent Tool-Call Metadata at Three Web Vantages*.

| Folder | Contents |
|---|---|
| `code/` | The LeakBench harness: trace sources, observation maps, estimator, live TLS capture, tests |
| `results/` | Stored outputs of the original runs: generator study, live captures, wide-area traces, hosted-model sweep |
| `e0/` | Recomputation of the paper's numbers from code and stored traces, and the merged Table 1 |
| `e1/` | Single-provider vantage (PROV-1) |
| `e2/` | Live LLM captures with a self-hosted 4-bit llama-3.3-70b (two seeds, traces included) |

## Setup

Python 3.11 or later.

```bash
cd code && python -m venv .venv && . .venv/bin/activate && pip install -e . -r requirements.txt && pytest -q
```

Run everything below from the artifact root, with that environment active.

## Where each number in the paper comes from

| Paper | File | Command |
|---|---|---|
| Table 1, generator and measured rows; validation (0.018 bits); subgroups; open world | `e0/scale_study/*.json` | `python code/scripts/run_scale_study.py --only vantage,subgroup,validation,measured_multiseed,openworld,live --output e0/scale_study` (about 1 min) |
| 0.018 bits at 3,000 traces per class | `e0/scale_study/validation_n3000.json` | `python code/scripts/rerun_validation.py` with 3,000 traces per class |
| Table 1, wide-area and loopback rows, seed B | `e0/wan_rescored.json` | `python e0/rescore_wan.py results/wan e0/wan_rescored.json` |
| Size-only / timing-only, seed B | `e0/projections_seedB.json` | `python e0/projections_seedB.py` |
| Table 1 merged | `e0/table1_merged.csv` | `python e0/merge_table.py` |
| PROV-1 column, Section 4 single-provider numbers | `e1/single_provider.json`, `e1/table_single_provider.csv` | `python e1/single_provider.py e1/single_provider.json && python e1/summarize.py` (about 30 s) |
| Hosted LLM rows (llama-3.3-70b and four other models) | `results/live_neural/live.json`, `results/llm_sweep/*/live.json` | Stored outputs; rerunning needs an API key for a hosted model (`code/scripts/run_live_capture.py --planner llm`) |
| Self-hosted rows | `e2/llm_seed{1,2}_local/` | `python e2/run_llm_seed.py --seed 20270214 --output e2/llm_seed1_local --base-url http://localhost:11434/v1 --model llama3.3:70b` against a local Ollama server (about 40 min per seed); set `LLM_API_KEY` to any value |

Seeds: A = 20270214, B = 20270215. Captured timing and chunk sizes vary from run to run by design,
so a re-captured live run reproduces within its confidence intervals, not to the last digit.
`e0/compare.py stored_dir new_dir stages tol` diffs a recomputed stage against `results/scale_study/`.

## Notes

- The code covers more experiments than the 4-page paper reports, such as defenses and taxonomy
  sweeps. Only the results listed above are claimed in the paper.
- No API keys or credentials are included. `code/.env.example` lists the variable names.
