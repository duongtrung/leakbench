#!/usr/bin/env python3
"""Run the cross-model live sweep as one sequential job.

Written after an inline shell queue deadlocked: each waiting wrapper matched its own
command line with `pgrep -f run_live_capture`, so the wrappers counted each other and
none of them ever started. One process running the list in order has no such failure
mode, and it is also harder to kill by accident.

Each model gets its own capture, all at the same seed so the task draws match. A model
that fails is recorded and the sweep continues -- one uncooperative model should not
cost the other five.

Usage:
    python scripts/run_model_sweep.py                     # the paper's set
    python scripts/run_model_sweep.py --only gpt-oss-20b-bedrock
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# (tag, provider, model id, max_tokens)
#
# max_tokens is per-model on purpose: reasoning models spend the whole budget on a
# hidden chain of thought before answering, so 128 (fine for a direct model) leaves
# them no room to emit the JSON at all.
# Ordered cheapest-first, using measured per-call latency rather than guesswork: at
# twelve planner calls per session and 150 sessions, a model's seconds-per-call maps
# almost linearly to the wall-clock of its capture. Measured on this link:
#
#     qwen3-30b-a3b   0.5 s   ~16 min      gemma-3-27b        1.5 s   ~46 min
#     mistral-nemo    0.8 s   ~23 min      llama-3.3-70b      1.7 s   ~52 min
#     gpt-oss-20b     1.0 s   ~29 min      mistral-small-24b  2.0 s   ~60 min
#     qwen3-32b       8.4 s  ~251 min  <-- reasoning; an order of magnitude, not a bug
#
# Run these one at a time. Three concurrent captures made each of them ~10x slower
# than its single-call latency predicts, which is how a 30-minute job looked like a
# four-hour hang.
SWEEP: list[tuple[str, str, str, int]] = [
    ("qwen3-30b-a3b", "openrouter", "qwen/qwen3-30b-a3b-instruct-2507", 128),
    ("mistral-nemo", "openrouter", "mistralai/mistral-nemo", 128),
    ("gpt-oss-20b-openrouter", "openrouter", "openai/gpt-oss-20b", 1024),
    # Serving-infrastructure control: same weights as the row above, different stack.
    ("gpt-oss-20b-bedrock", "bedrock", "openai.gpt-oss-20b-1:0", 1024),
    # Reasoning models, last: they answer an order of magnitude slower, so they are
    # the ones to drop or shrink if time runs out rather than the ones to wait on.
    ("qwen3-32b", "openrouter", "qwen/qwen3-32b", 1024),
    ("deepseek-v4-flash", "openrouter", "deepseek/deepseek-v4-flash", 2048),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--outdir", default="outputs/llm_sweep")
    ap.add_argument("--only", nargs="*", help="run only these tags")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    # Create it up front. Writing the per-model failure log used to be the first thing
    # that touched this directory, so a first-model failure raised FileNotFoundError
    # *while reporting the failure* and took the whole sweep down -- defeating the one
    # property this script exists for, that one bad model costs only itself.
    outdir.mkdir(parents=True, exist_ok=True)
    by_tag = {row[0]: row for row in SWEEP}
    if args.only:
        # Honour the order the caller asked for. Filtering SWEEP in place instead
        # silently ignores it, which matters when the point of naming two models is to
        # run the riskier one first so a failure in the other does not cost both.
        unknown = [t for t in args.only if t not in by_tag]
        if unknown:
            raise SystemExit(f"unknown tag(s): {unknown}; known: {sorted(by_tag)}")
        todo = [by_tag[t] for t in args.only]
    else:
        todo = list(SWEEP)
    status: dict[str, str] = {}

    for tag, provider, model, max_tokens in todo:
        dest = outdir / tag
        if args.skip_existing and (dest / "live.json").exists():
            print(f"[skip] {tag}: already captured", flush=True)
            status[tag] = "already-present"
            continue
        cmd = [sys.executable, "scripts/run_live_capture.py",
               "--planner", "llm", "--llm-provider", provider, "--llm-model", model,
               "--max-tokens", str(max_tokens),
               "--traces-per-class", str(args.traces_per_class),
               "--seeds", "1", "--seed", str(args.seed), "--output", str(dest)]
        print(f"[run ] {tag}: {provider}/{model} (max_tokens={max_tokens})", flush=True)
        started = time.time()
        # Stream the child's output instead of capturing it. `capture_output=True`
        # buffers everything until the process exits, which for a run that takes tens
        # of minutes means the per-class progress the harness emits is invisible for
        # exactly as long as it would have been useful.
        lines: list[str] = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    lines.append(line)
                    print(f"       {line}", flush=True)
            returncode = proc.wait()
        elapsed = time.time() - started
        if returncode == 0 and (dest / "live.json").exists():
            status[tag] = "ok"
            print(f"[ok  ] {tag} in {elapsed / 60:.1f} min", flush=True)
        else:
            reason = lines[-1][:200] if lines else f"exit {returncode}"
            status[tag] = f"failed: {reason}"
            print(f"[fail] {tag} after {elapsed / 60:.1f} min: {reason}", flush=True)
            try:
                (outdir / f"{tag}.log").write_text("\n".join(lines))
            except OSError as exc:                # never let reporting a failure fail
                print(f"       (could not write {tag}.log: {exc})", flush=True)

    print("\n=== sweep summary ===")
    for tag, st in status.items():
        print(f"  {tag:<26} {st}")
    (outdir / "sweep_status.json").write_text(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
