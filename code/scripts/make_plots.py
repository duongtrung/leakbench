#!/usr/bin/env python3
"""Regenerate plots from existing result JSON files.

    python scripts/make_plots.py --input outputs/synthetic_demo --output outputs/synthetic_demo/plots

Reads the stage result files written by the demo / experiment runner and rebuilds
the figures. Plots are generated FROM the result files, never from hard-coded
numbers. If a result came from synthetic data (status contains 'synthetic'), the
figure is stamped accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leakbench.experiments.plotting import (
    plot_ablation,
    plot_defenses,
    plot_validation,
    plot_vantage,
)
from leakbench.utils.io import load_json


def _is_synthetic(payload: dict) -> bool:
    return "synthetic" in str(payload.get("status", "")).lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="directory with stage JSON files")
    ap.add_argument("--output", required=True, help="directory for plots")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    made = []
    mapping = {
        "vantage.json": (plot_vantage, "vantage.png"),
        "ablation.json": (plot_ablation, "ablation.png"),
        "defenses.json": (plot_defenses, "defenses.png"),
        "validation.json": (plot_validation, "validation.png"),
    }
    for fname, (fn, out_name) in mapping.items():
        fpath = in_dir / fname
        if not fpath.exists():
            continue
        payload = load_json(fpath)
        result = payload.get("result", payload)
        synthetic = _is_synthetic(payload)
        made.append(str(fn(result, out_dir / out_name, synthetic=synthetic)))
    if not made:
        print(f"No recognized result files found in {in_dir}.")
        return 1
    print("Wrote:\n  " + "\n  ".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
