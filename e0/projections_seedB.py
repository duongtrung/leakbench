"""Size-only / timing-only advantage for stored traces (same code path as runners.run_live)."""
import json, sys
from pathlib import Path
import numpy as np
from leakbench.data.loader import iter_trace_file
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import network_map
from leakbench.observables.projections import project_features

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "results/wan"
sets = {
    "loopback_seed20270215": (W / "traces_loopback_seed20270215.jsonl", 20270215),
    "wan_seed20270215": (W / "traces_wan_seed20270215.jsonl", 20270215),
    "check_llm_local_seed20270214": (ROOT / "e2/llm_seed1_local/traces_seed20270214.jsonl", 20270214),
}
out = {}
for name, (p, seed) in sets.items():
    tr = list(iter_trace_file(p))
    X = np.vstack([network_map(t) for t in tr]); y = np.array([t.label for t in tr])
    out[name] = {proj: round(estimate_leakage(project_features(X, proj), y, vantage=proj,
                 prior=np.full(6, 1/6), bootstrap=100, seed=seed).advantage, 4)
                 for proj in ("size_only", "timing_only")}
    print(name, out[name], flush=True)
json.dump(out, open(ROOT / "e0/projections_seedB.json", "w"), indent=2)
