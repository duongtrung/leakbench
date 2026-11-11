#!/usr/bin/env python3
"""Capture traces from a **real LLM driving real tools** (framework-adapter path).

    # cheapest sanity check: 1 trace per class, 4 steps max
    python scripts/run_adapter.py --adapter react --traces-per-class 1 --max-steps 4

    # over a real socket to the MCP-style tool server
    python scripts/run_adapter.py --adapter mcp_router --traces-per-class 5

    # a different provider / model
    python scripts/run_adapter.py --llm-provider groq --llm-model llama-3.3-70b-versatile

The model decides each next tool call; the call executes against the genuine local
backends; the tool's real output goes back into the next prompt. Sizes and
inter-arrival gaps are measured, so this closes the paper's "templated planner"
simplification.

Requires an API key (see `.env.example`). **This is the only script that leaves the
machine**, and it does so only because you asked it to; it estimates leakage on the
captured traces and writes the usual result file, plus the model's token usage. The
key itself is never written to any output.

Cost control: the bill is roughly `classes x traces_per_class x max_steps` short
completions. Start with `--traces-per-class 1` and read the printed usage before
scaling up; `--dry-run` prints the plan and the call estimate without spending
anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from leakbench.data.adapters import MCPRouterAdapter, ReActAdapter
from leakbench.data.llm import make_completion_fn
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset
from leakbench.taxonomy import reference_taxonomy, scaled_taxonomy
from leakbench.utils.io import dump_json
from leakbench.utils.provenance import capture_provenance
from leakbench.utils.seeding import seed_everything

ADAPTERS = {"react": ReActAdapter, "mcp_router": MCPRouterAdapter}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), default="react")
    ap.add_argument("--traces-per-class", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--llm-provider", default="openrouter",
                    help="openrouter | groq | openai | custom")
    ap.add_argument("--llm-model", default=None, help="model id (provider default if unset)")
    ap.add_argument("--llm-base-url", default=None, help="required for --llm-provider custom")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the call estimate; contact no endpoint")
    ap.add_argument("--output", default="outputs/adapter")
    args = ap.parse_args()

    tax = (reference_taxonomy() if args.num_classes == 12
           else scaled_taxonomy(args.num_classes))
    est_calls = tax.num_classes * args.traces_per_class * args.max_steps
    print(f"adapter={args.adapter}  classes={tax.num_classes}  "
          f"traces/class={args.traces_per_class}  max_steps={args.max_steps}")
    print(f"  -> at most {est_calls} short completions to "
          f"{args.llm_provider}/{args.llm_model or 'default model'}")
    if args.dry_run:
        print("dry run: no endpoint contacted, nothing spent.")
        return 0

    seed_everything(args.seed)
    complete, client = make_completion_fn(
        provider=args.llm_provider, model=args.llm_model,
        base_url=args.llm_base_url, temperature=args.temperature,
    )
    adapter = ADAPTERS[args.adapter](
        complete=complete, taxonomy=tax, max_steps=args.max_steps, seed=args.seed,
    )
    ds = adapter.sample_dataset(tax, args.traces_per_class)

    result: dict = {
        "rq": "framework_adapter",
        "adapter": args.adapter,
        "n_traces": len(ds),
        "mean_trace_length": float(np.mean([t.n for t in ds.traces])),
        "llm": client.usage(),               # provider/model/call counts, never the key
        "advantage_by_vantage": {},
    }
    for v in ("network", "platform", "providers"):
        X, y = featurize_dataset(ds, v)
        try:
            leak = estimate_leakage(X, y, vantage=v, prior=tax.prior,
                                    bootstrap=100, seed=args.seed)
        except ValueError as exc:                 # too few traces per class to split
            result["advantage_by_vantage"][v] = None
            result.setdefault("notes", []).append(f"{v}: {exc}")
            continue
        result["advantage_by_vantage"][v] = leak.advantage

    prov = capture_provenance(args.seed, tax.content_hash(), tax.name, vars(args))
    payload = {"provenance": prov.to_dict(), "result": result,
               "status": f"real_llm_driving_real_local_tools ({args.adapter})",
               "gaps": ["tools are local backends, not a production tool server"]}
    out = dump_json(payload, Path(args.output) / f"{args.adapter}.json")
    print(f"\nCaptured {len(ds)} traces -> {out}")
    print(f"  LLM usage: {client.usage()}")
    for v, a in result["advantage_by_vantage"].items():
        print(f"  {v:10s} advantage = {a if a is None else round(a, 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
