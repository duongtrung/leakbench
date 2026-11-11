#!/usr/bin/env python3
"""Regenerate the example JSONL trace sample (one trace per class).

    python examples/synthetic_data/generate_example_data.py

Writes traces_sample.jsonl next to this script. Output is
"synthetic smoke-test output, not a paper result."
"""

from __future__ import annotations

import json
from pathlib import Path

from leakbench.data.generator import CalibratedGenerator
from leakbench.taxonomy import reference_taxonomy


def main() -> int:
    tax = reference_taxonomy()
    gen = CalibratedGenerator(taxonomy=tax, seed=20270214)
    out = Path(__file__).parent / "traces_sample.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for ci in range(tax.num_classes):
            trace = gen.sample_trace(ci)
            fh.write(json.dumps(trace.to_record()) + "\n")
    print(f"Wrote {out} (synthetic smoke-test data, not a paper result).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
