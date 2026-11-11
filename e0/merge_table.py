"""Build the merged Table 1 (vantage x realism level) from E0 recomputations and stored results."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORED = ROOT / "results"
E0 = ROOT / "e0"
V = ("network", "platform", "providers")


def load(p):
    return json.load(open(p))["result"]


def from_vantages(vs, ci_key="advantage_ci"):
    return {v: (vs[v]["advantage"], *(vs[v].get(ci_key) or (None, None))) for v in V}


rows = []


def add(level, setting, classes, sessions, seeds, provenance, adv, size=None, timing=None):
    row = {"level": level, "setting": setting, "classes": classes, "sessions": sessions,
           "seeds": seeds, "provenance": provenance, "size_only": size, "timing_only": timing}
    for v in V:
        a, lo, hi = adv[v]
        row[f"{v}_adv"], row[f"{v}_lo"], row[f"{v}_hi"] = a, lo, hi
    rows.append(row)


r = load(E0 / "scale_study/vantage.json")
add("synthetic", "calibrated generator", 12, 7200, "20270214", "recomputed on Mac = stored",
    from_vantages(r["vantages"]))

r = load(E0 / "scale_study/measured_multiseed.json")
add("measured", "real local tools, mean of 5 seeds (lo/hi = min/max over seeds)", 12,
    r.get("traces_per_class", 130) * 12, "5 seeds", "recomputed on Mac",
    {v: (r["summary"][v]["mean"], r["summary"][v]["min"], r["summary"][v]["max"]) for v in V})

for prov, path in (("stored (original run, Linux)", STORED / "scale_study/live.json"),
                   ("recomputed on Mac (macOS loopback)", E0 / "scale_study/live.json")):
    s = load(path)["per_seed"][0]
    add("L-prog", "scripted planner, live TLS, loopback", 6, s.get("n_traces", 420), "20270214",
        prov, from_vantages(s["vantages"]), s["projections"]["size_only"], s["projections"]["timing_only"])

s = load(STORED / "live_neural/live.json")["per_seed"][0]
add("L-LLM", "llama-3.3-70b planner, live TLS, loopback", 6, s["n_traces"], "20270214",
    "stored (original run), re-read", from_vantages(s["vantages"]),
    s["projections"]["size_only"], s["projections"]["timing_only"])

for sub, seed in (("llm_seed1_local", "20270214"), ("llm_seed2_local", "20270215")):
    s = load(ROOT / f"e2/{sub}/live.json")["per_seed"][0]
    add("L-LLM", "llama3.3:70b Q4_K_M, local Ollama, live TLS, loopback; every session hit the 12-step cap", 6,
        s["n_traces"], seed, "new run on Mac (E2)", from_vantages(s["vantages"]),
        s["projections"]["size_only"], s["projections"]["timing_only"])

r = load(STORED / "wan/live_wan.json")
add("WAN", "scripted planner, live TLS, AWS us-east-1", 6, r["n_traces"], "20270214",
    "stored (original run), re-read", from_vantages(r["vantages"]),
    r["projections"]["size_only"], r["projections"]["timing_only"])

w = json.load(open(E0 / "wan_rescored.json"))["wan"]
add("WAN", "scripted planner, live TLS, AWS us-east-1", 6, w["n_traces"], "20270215",
    "recomputed on Mac from stored traces = stored", from_vantages(w, "ci"))

with open(E0 / "table1_merged.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    for row in rows:
        wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def cell(row, v):
    a, lo, hi = row[f"{v}_adv"], row[f"{v}_lo"], row[f"{v}_hi"]
    return f"{a:.2f}" + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")


for row in rows:
    extra = "" if row["size_only"] is None else f" | size {row['size_only']:.2f} time {row['timing_only']:.2f}"
    print(f"{row['level']:9s} {row['seeds']:9s} " + " ".join(f"{cell(row, v):18s}" for v in V)
          + f"{extra}  ({row['provenance']})")
