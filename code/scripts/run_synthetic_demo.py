#!/usr/bin/env python3
"""Run the LeakBench synthetic smoke-test demo.

    python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml

Produces result JSON + plots under outputs/synthetic_demo/, all labeled
"synthetic smoke-test output, not a paper result".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leakbench.experiments.config import ExperimentConfig, load_config
from leakbench.experiments.demo import run_synthetic_demo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/synthetic_demo.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else ExperimentConfig(
        name="synthetic_demo", traces_per_class=40, bootstrap=50,
        output_dir="outputs/synthetic_demo",
    )
    out = run_synthetic_demo(cfg)
    print(f"Done. Synthetic (non-paper) outputs in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
