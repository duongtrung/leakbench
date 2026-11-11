#!/usr/bin/env python3
"""RQ7 --- generalization across taxonomy size.

    python scripts/run_taxonomy_sweep.py --sizes 8 12 16 24 --traces-per-class 600

Sweeps |T| and reports normalized advantage (expected to dip) and absolute MI in
bits (expected to rise monotonically) at the platform vantage on the generator.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from leakbench.experiments.config import ExperimentConfig, load_config
from leakbench.experiments.runners import run_taxonomy_sweep
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.io import dump_json
from leakbench.utils.provenance import capture_provenance
from leakbench.utils.seeding import seed_everything


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24])
    ap.add_argument("--traces-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--config", default=None, help="optional YAML config (bank, bootstrap)")
    ap.add_argument("--output", default="outputs/taxsweep")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    cfg = replace(cfg, seed=args.seed, traces_per_class=args.traces_per_class)
    result = run_taxonomy_sweep(cfg, sizes=tuple(args.sizes))
    for row in result["rows"]:
        print(f"|T|={row['num_classes']:2d}  adv={row['advantage']:.3f}  "
              f"MI={row['mi_bits']:.3f} bits  (H={row['entropy_bits']:.2f})")

    tax = reference_taxonomy()
    prov = capture_provenance(args.seed, tax.content_hash(), tax.name, cfg.to_dict())
    payload = {"provenance": prov.to_dict(), "result": result,
               "status": "real_computation_on_synthetic_data"}
    out = dump_json(payload, Path(args.output) / "taxsweep.json")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
