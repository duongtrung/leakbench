"""Defense sweep at the PLATFORM vantage.

The paper's defense section makes platform-vantage claims (order hiding is inert
because the platform's power is the category *count* histogram, not the order),
but the study only measured the network vantage. Measure it.
"""
import json

from leakbench.data.generator import CalibratedGenerator
from leakbench.defenses import get_defense
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset
from leakbench.taxonomy import reference_taxonomy

SEED = 20270214
DEFENSES = ["pad-512", "shuffle", "constant-rate", "cover", "pad-max", "full",
            "gen-shaping", "dp-shaping", "mixing"]

tax = reference_taxonomy()
ds = CalibratedGenerator(taxonomy=tax, seed=SEED).sample_dataset(600)


def score(dataset, vantage):
    X, y = featurize_dataset(dataset, vantage)
    r = estimate_leakage(X, y, vantage=vantage, prior=tax.prior, bootstrap=200, seed=SEED)
    return {"advantage": r.advantage, "mi_bits": r.mi_lower_bound_bits,
            "advantage_ci": list(r.advantage_ci)}


rows = {"none": {v: score(ds, v) for v in ("platform", "providers")}}
for name in DEFENSES:
    res = get_defense(name).apply(ds, seed=SEED)
    rows[name] = {v: score(res.dataset, v) for v in ("platform", "providers")}
    rows[name]["bandwidth_overhead"] = res.bandwidth_overhead
    print(name, "platform=%.3f" % rows[name]["platform"]["advantage"],
          "providers=%.3f" % rows[name]["providers"]["advantage"], flush=True)

out = {"status": "real_computation_at_paper_scale",
       "result": {"rq": "RQ4b_defenses_operator_vantages", "rows": rows,
                  "traces_per_class": 600, "seed": SEED,
                  "note": "the paper's order-hiding argument is a platform-vantage claim; "
                          "measure it there rather than inferring it from the network run"},
       "provenance": json.load(open("outputs/scale_study/defenses.json"))["provenance"]}
json.dump(out, open("outputs/scale_study/defenses_operator.json", "w"), indent=2)
