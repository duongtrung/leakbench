"""Cross-stage identities: the same quantity, computed by different stages, must agree.

Bit-reproducible stages must agree exactly. Stages built on the measured harness are
allowed a tolerance, because that harness records real wall-clock timing and is
reproducible only up to it (see scripts/measured_repeatability.py); the tolerance here
is the measured repeat range, not a fudge factor.
"""
import json
import sys


def R(n): return json.load(open(f"outputs/scale_study/{n}.json"))["result"]

MEASURED_TOL = 0.03   # observed repeat range at the network vantage, rounded up
v, dfn = R("vantage")["vantages"], {r["defense"]: r for r in R("defenses")["rows"]}
mix = {r["n_users"]: r for r in R("mixing")["rows"]}
ea, co, sub = R("estimator_agreement"), R("coarse")["rows"], R("subgroup")
bank = R("bank_comparison")["rows"]
ov = {r["overlap"]: r for r in R("overlap_sweep")["rows"]}
mm, proj, ws = R("measured_multiseed"), R("projections")["rows"], R("wire_sweep")
net = v["network"]["advantage"]

exact = [
    ("network adv: vantage vs defenses[none]",      net, dfn["none"]["advantage"]),
    ("network adv: vantage vs mixing[n=1]",         net, mix[1]["advantage"]),
    ("network adv: vantage vs estimator_agreement", net, ea["by_vantage"]["network"]["random_forest"]),
    ("network adv: vantage vs coarse[class]",       net, co["network"]["class_advantage"]),
    ("network adv: vantage vs bank[core]",          net, bank["network"]["core"]["advantage"]),
    ("platform adv: vantage vs subgroup overall",   v["platform"]["advantage"], sub["overall_advantage"]),
    ("platform adv: vantage vs coarse[class]",      v["platform"]["advantage"], co["platform"]["class_advantage"]),
    ("network MI: vantage vs defenses[none]",       v["network"]["mi_lower_bound_bits"], dfn["none"]["mi_bits"]),
]
tol = [
    ("measured platform: multiseed vs overlap[0.15]", mm["per_seed_advantage"][0]["platform"], ov[0.15]["platform"]),
    ("measured platform: multiseed vs projections",   mm["per_seed_advantage"][0]["platform"], proj["platform_full"]["advantage"]),
    ("measured platform: multiseed vs wire_sweep",    mm["per_seed_advantage"][0]["platform"], ws["platform_advantage"]),
    ("measured network:  multiseed vs projections",   mm["per_seed_advantage"][0]["network"], proj["network_full"]["advantage"]),
]
bad = 0
for name, a, b in exact:
    ok = abs(a - b) < 1e-9
    bad += not ok
    print(f"  {'ok      ' if ok else 'MISMATCH'} {name:<48} {a:.6f} vs {b:.6f}")
for name, a, b in tol:
    ok = abs(a - b) <= MEASURED_TOL
    bad += not ok
    print(f"  {'ok(tol) ' if ok else 'MISMATCH'} {name:<48} {a:.6f} vs {b:.6f}  (d={abs(a-b):.4f})")
print(f"\n{len(exact)} exact + {len(tol)} tolerance identities, {bad} mismatch(es)")
sys.exit(1 if bad else 0)
