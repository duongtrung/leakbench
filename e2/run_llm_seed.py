"""Second L-LLM seed: same protocol as scripts/run_live_capture.py --planner llm, but the
captured traces are also saved (the original script keeps only the result JSON).

    LLM_API_KEY=unused python e2/run_llm_seed.py --seed 20270215 --output e2/llm_seed2 \
        --base-url http://localhost:11434/v1 --model llama3.3:70b \
        --served-via "local Ollama 0.40.15, llama3.3:70b Q4_K_M, one workstation"
"""
import argparse
from pathlib import Path

import leakbench.data.live_capture as lc
from leakbench.data.live_capture import LIVE_CLASSES, LLMPlanner
from leakbench.data.llm import make_completion_fn
from leakbench.data.loader import dump_trace_dataset
from leakbench.experiments.config import ExperimentConfig
from leakbench.experiments.runners import run_live
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.io import dump_json
from leakbench.utils.provenance import capture_provenance
from leakbench.utils.seeding import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--traces-per-class", type=int, default=25)
ap.add_argument("--base-url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--max-steps", type=int, default=12)
ap.add_argument("--output", required=True)
ap.add_argument("--served-via", required=True, help="recorded in the result, e.g. 'local Ollama, Q4_K_M'")
args = ap.parse_args()
out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)

# Keep every captured dataset on disk, then hand it back to run_live unchanged.
_capture = lc.LiveCaptureHarness.capture_dataset


def capture_and_save(self, *a, **kw):
    ds = _capture(self, *a, **kw)
    dump_trace_dataset(ds, out_dir / f"traces_seed{args.seed}.jsonl")
    return ds


lc.LiveCaptureHarness.capture_dataset = capture_and_save

cfg = ExperimentConfig(seed=args.seed, bootstrap=100)  # as run_live_capture.py without --config
seed_everything(args.seed)
complete, client = make_completion_fn(provider="custom", model=args.model,
                                      base_url=args.base_url, max_tokens=args.max_tokens)
print(f"up to {len(LIVE_CLASSES) * args.traces_per_class * args.max_steps} completions", flush=True)
result = run_live(cfg, traces_per_class=args.traces_per_class, seeds=[args.seed], use_tls=True,
                  planner_factory=lambda _s: LLMPlanner(complete=complete, max_steps=args.max_steps))
result["llm"] = client.usage()
result["llm"]["served_via"] = args.served_via
tax = reference_taxonomy()
prov = capture_provenance(args.seed, tax.content_hash(), tax.name,
                          {"traces_per_class": args.traces_per_class, "seeds": 1, "use_tls": True,
                           "server_host": None, "server_port": None})
dump_json({"provenance": prov.to_dict(), "result": result,
           "status": f"live_captured_traffic ({result['capture_mode']})", "gaps": result["gaps"]},
          out_dir / "live.json")
s = result["per_seed"][0]
print({v: round(d["advantage"], 3) for v, d in s["vantages"].items()}, s["projections"], result["llm"])
