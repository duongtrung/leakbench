#!/usr/bin/env python3
"""RQ8 --- live experiment with captured (loopback) traffic.

    python scripts/run_live_capture.py --traces-per-class 70 --seeds 2 --output outputs/live

Stands up a REAL tool-server over real TLS (a throwaway self-signed certificate,
minted locally), drives it with a ReAct-style planner that branches on genuine tool
outputs, relays the connection through a REAL TCP capture proxy, and takes the network
observable from the captured chunk sizes --- ciphertext bytes, captured rather than
modeled. Estimates leakage per vantage and writes a result file.

Two variants close the paper's stated RQ8 limitations:

    # neural controller instead of the programmatic one (needs an API key; see .env.example)
    python scripts/run_live_capture.py --planner llm --llm-provider openrouter

    # real link instead of loopback (needs scripts/run_tool_server.py on another host)
    python scripts/run_live_capture.py --server-host <ip> --server-port 8443 --server-ca cert.pem

Whatever is *not* closed stays in the result file's `gaps` list, and `link` is
resolved from the server address rather than from the flag. If no certificate material
can be provisioned, the run falls back to plaintext TCP with modeled TLS framing and
records that mode.

By default this opens sockets on 127.0.0.1 only and contacts no external service. It
is intentionally NOT part of the default pytest run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leakbench.data.live_capture import LIVE_CLASSES
from leakbench.experiments.config import ExperimentConfig, load_config
from leakbench.experiments.runners import run_live
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.io import dump_json
from leakbench.utils.provenance import capture_provenance
from leakbench.utils.seeding import seed_everything


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-per-class", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1, help="number of capture seeds")
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--config", default=None, help="optional YAML config (bank, bootstrap)")
    ap.add_argument("--no-tls", action="store_true",
                    help="capture plaintext TCP instead (the mode is recorded)")
    ap.add_argument("--server-host", default=None,
                    help="tool server on ANOTHER host (WAN variant); see "
                         "scripts/run_tool_server.py and docs/ec2_experiment_guide.md")
    ap.add_argument("--server-port", type=int, default=None)
    ap.add_argument("--server-ca", default=None,
                    help="certificate file copied from the remote tool server")
    ap.add_argument("--planner", choices=("programmatic", "llm"), default="programmatic",
                    help="'llm' drives the session with a real model (needs an API key)")
    ap.add_argument("--llm-provider", default="openrouter",
                    help="openrouter | groq | openai | custom")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="per-completion token budget. Reasoning models spend it on a "
                         "hidden chain of thought before answering, so they need far "
                         "more than a direct model does (1024 is a workable floor)")
    ap.add_argument("--max-steps", type=int, default=12,
                    help="cap on tool calls per session when --planner llm")
    ap.add_argument("--output", default="outputs/live")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else ExperimentConfig(seed=args.seed,
                                                                       bootstrap=100)
    seed_everything(args.seed)

    planner_factory = None
    client = None
    if args.planner == "llm":
        from leakbench.data.live_capture import LLMPlanner
        from leakbench.data.llm import make_completion_fn

        complete, client = make_completion_fn(
            provider=args.llm_provider, model=args.llm_model,
            base_url=args.llm_base_url, max_tokens=args.max_tokens,
        )
        n_sessions = len(LIVE_CLASSES) * args.traces_per_class * args.seeds
        print(f"neural planner: up to {n_sessions * args.max_steps} completions to "
              f"{args.llm_provider}/{args.llm_model or 'default model'}")
        def planner_factory(_seed: int) -> LLMPlanner:
            return LLMPlanner(complete=complete, max_steps=args.max_steps)

    result = run_live(
        cfg,
        traces_per_class=args.traces_per_class,
        seeds=[args.seed + i for i in range(args.seeds)],
        use_tls=not args.no_tls,
        server_host=args.server_host,
        server_port=args.server_port,
        server_ca_file=args.server_ca,
        planner_factory=planner_factory,
    )
    if client is not None:
        result["llm"] = client.usage()
    tax = reference_taxonomy()
    prov = capture_provenance(args.seed, tax.content_hash(), tax.name,
                              {"traces_per_class": args.traces_per_class,
                               "seeds": args.seeds, "use_tls": not args.no_tls,
                               "server_host": args.server_host,
                               "server_port": args.server_port})
    payload = {
        "provenance": prov.to_dict(),
        "result": result,
        "status": f"live_captured_traffic ({result['capture_mode']})",
        "gaps": result["gaps"],
    }
    out = dump_json(payload, Path(args.output) / "live.json")
    print(f"Live capture complete ({result['capture_mode']}, link={result['link']}). "
          f"Result: {out}")
    for entry in result["per_seed"]:
        adv = {v: round(d["advantage"], 3) for v, d in entry["vantages"].items()}
        print(f"  seed {entry['seed']}: {adv}  "
              f"size-only={entry['projections']['size_only']:.3f}  "
              f"timing-only={entry['projections']['timing_only']:.3f}")
    print(f"  network < platform in {result['net_below_platform_in_seeds']}/"
          f"{len(result['per_seed'])} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
