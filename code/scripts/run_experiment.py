#!/usr/bin/env python3
"""Run a LeakBench experiment stage on the generator or measured harness.

    python scripts/run_experiment.py --config configs/default.yaml --stage vantage
    python scripts/run_experiment.py --config configs/measured.yaml --stage measured
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage all

Stages: vantage (RQ1), ablation (RQ2), subgroup (RQ3), defenses (RQ4),
validation (RQ5), measured (RQ6), or 'all'.

This script runs only what is configured; for *real* datasets set `data.root` in the
config and see docs/ec2_experiment_guide.md. It never downloads data.
"""

from __future__ import annotations

import argparse

from leakbench.cli import _cmd_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--stage", default="all", help="stage name or 'all'")
    args = ap.parse_args()
    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
