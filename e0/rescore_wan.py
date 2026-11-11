"""Re-estimate the stored WAN and loopback traces (seed 20270215) without re-capturing."""
import json
import sys
from pathlib import Path

import numpy as np

from leakbench.data.loader import iter_trace_file
from leakbench.data.schemas import TraceDataset
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset, platform_feature_slices

wan_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
seed = 20270215
rows = {}
for cond in ("loopback", "wan"):
    traces = list(iter_trace_file(wan_dir / f"traces_{cond}_seed{seed}.jsonl"))
    ds = TraceDataset(traces=traces, taxonomy_name="live-6c", taxonomy_hash="n/a", source="live")
    k = len(set(t.label for t in traces))
    prior = np.full(k, 1.0 / k)
    rows[cond] = {"n_traces": len(traces)}
    for v in ("network", "platform", "providers"):
        X, y = featurize_dataset(ds, v)
        r = estimate_leakage(X, y, vantage=v, prior=prior, bootstrap=200, seed=seed)
        rows[cond][v] = {"advantage": r.advantage, "ci": list(r.advantage_ci), "mi_bits": r.mi_lower_bound_bits}
json.dump(rows, open(out, "w"), indent=2)
for cond, r in rows.items():
    print(cond, r["n_traces"], {v: round(r[v]["advantage"], 4) for v in ("network", "platform", "providers")})
