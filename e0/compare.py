"""Compare recomputed stage results against the stored results, leaf by leaf."""
import json
import sys
from pathlib import Path


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


stored_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
stages = sys.argv[3].split(",")
tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
for stage in stages:
    a = dict(leaves(json.load(open(stored_dir / f"{stage}.json"))["result"]))
    b = dict(leaves(json.load(open(new_dir / f"{stage}.json"))["result"]))
    common = sorted(set(a) & set(b))
    diffs = [(k, a[k], b[k]) for k in common if abs(a[k] - b[k]) > tol]
    exact = sum(1 for k in common if a[k] == b[k])
    print(f"## {stage}: {len(common)} values, {exact} identical, {len(diffs)} differ by > {tol}"
          f" (only stored: {len(set(a) - set(b))}, only new: {len(set(b) - set(a))})")
    for k, x, y in diffs[:25]:
        print(f"   {k}: stored {x:.4f}  new {y:.4f}  diff {y - x:+.4f}")
