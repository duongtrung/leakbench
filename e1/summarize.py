"""Summarize e1/single_provider.json into e1/table_single_provider.csv and print a markdown table."""
import csv
import json
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "e1/single_provider.json")
R = json.load(open(src))
LABELS = {
    "synthetic_12c_seed20270214": ("Synthetic (12 cl.)", "20270214"),
    "loopback_scripted_seed20270215": ("L-prog loopback (6 cl.)", "20270215"),
    "wan_scripted_seed20270215": ("WAN, scripted (6 cl.)", "20270215"),
    "llm_local_seed20270214": ("L-LLM local Q4 (6 cl.)", "20270214"),
    "llm_local_seed20270215": ("L-LLM local Q4 (6 cl.)", "20270215"),
}
fmt = lambda d: f"{d['advantage']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"
rows = []
for key, (label, seed) in LABELS.items():
    r = R[key]
    single = {c: s for c, s in r["single"].items() if "all" in s}
    best = max(single, key=lambda c: single[c]["all"]["advantage"])
    rows.append({
        "level": label, "seed": seed, "sessions": r["n_sessions"],
        "network": fmt(r["joint"]["network"]),
        "platform": fmt(r["joint"]["platform"]),
        "providers_colluding": fmt(r["joint"]["providers"]),
        "single_provider_median": f"{r['summary']['single_all_median']:.2f}",
        "single_provider_max": f"{fmt(single[best]['all'])} ({best}, in {single[best]['coverage']:.0%} of sessions)",
    })
out = src.parent / "table_single_provider.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("| Level | Seed | Network | Platform | Providers, colluding | One provider, median | One provider, max |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['level']} | {r['seed']} | {r['network']} | {r['platform']} | {r['providers_colluding']} | "
          f"{r['single_provider_median']} | {r['single_provider_max']} |")
