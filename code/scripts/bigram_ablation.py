"""Does the 8x8 bigram transition block earn its place in the platform observable?

Advantage is a LOWER bound, so dropping a block can only be justified if it costs
nothing. We measure the cost on all three trace sources, because the answer differs:
an i.i.d. generator has no order structure to find, while a real planner does.
"""
import json

import numpy as np

from leakbench.data.generator import CalibratedGenerator
from leakbench.data.live_capture import LiveCaptureHarness
from leakbench.data.measured import MeasuredHarness
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset, platform_feature_slices
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.provenance import capture_provenance

SEED = 20270214
tax = reference_taxonomy()


def adv(X, y, prior):
    return estimate_leakage(X, y, vantage="platform", prior=prior,
                            bootstrap=200, seed=SEED).advantage


def with_and_without(ds, prior, label):
    X, y = featurize_dataset(ds, "platform")
    keep = np.ones(X.shape[1], dtype=bool)
    keep[platform_feature_slices()["bigram"]] = False
    full, dropped = adv(X, y, prior), adv(X[:, keep], y, prior)
    print(f"{label:<28} full={full:.3f}  no-bigram={dropped:.3f}  "
          f"delta={dropped - full:+.3f}", flush=True)
    return {"full": full, "no_bigram": dropped, "delta": dropped - full,
            "n_features_full": int(X.shape[1]), "n_features_dropped": int(keep.sum())}


rows = {}
rows["generator"] = with_and_without(
    CalibratedGenerator(taxonomy=tax, seed=SEED).sample_dataset(600), tax.prior, "generator (i.i.d. tools)")
rows["measured"] = with_and_without(
    MeasuredHarness(taxonomy=tax, seed=SEED, overlap=0.15).sample_dataset(130), tax.prior,
    "measured (templated plans)")

# Live: a stochastic planner actually produces order structure.
live = LiveCaptureHarness(seed=SEED)
lds = live.capture_dataset(70)
lprior = np.full(len(set(t.label for t in lds.traces)),
                 1.0 / len(set(t.label for t in lds.traces)))
rows["live"] = with_and_without(lds, lprior, "live (stochastic planner)")

json.dump({"status": "real_computation_at_paper_scale",
           "result": {"rq": "bigram_block_ablation", "vantage": "platform", "rows": rows,
                      "note": "delta < 0 means dropping the block LOSES leakage, i.e. the "
                              "block is carrying signal a lower-bound estimator needs"},
           "provenance": capture_provenance(
               seed=SEED, taxonomy_hash=tax.content_hash(), taxonomy_name=tax.name,
               config={"traces_per_class": {"generator": 600, "measured": 130, "live": 70}},
               note="platform-vantage ablation of the 8x8 bigram transition block").to_dict()},
          open("outputs/scale_study/bigram_ablation.json", "w"), indent=2)
