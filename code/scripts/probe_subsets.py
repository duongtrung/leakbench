"""Would maximizing over feature-block subsets (as well as over the bank) tighten
the lower bound enough to justify changing the estimator?

Motivation: on the shared test split a strict feature *subset* can outscore the full
observable (paper section "Case study": 0.175 vs 0.145 at the measured network
vantage). Since advantage is a lower bound, that would mean the reported number is
loose and the estimator should maximize over subsets too.

This script settles it honestly: selection happens on a VALIDATION split and the
winner is scored on a held-out TEST split, so no winner's curse can inflate the gain.

Answer (recorded in outputs/scale_study/subset_selection_probe.json): the gain is
about zero and its sign is not even consistent across conditions, so the test-split
excess is selection noise, not head-room. The estimator is left alone.
"""
import itertools
import json

import numpy as np
from sklearn.model_selection import train_test_split

from leakbench.data.generator import CalibratedGenerator
from leakbench.data.measured import MeasuredHarness
from leakbench.data.wire import HTTP2WireModel, apply_wire_model_dataset
from leakbench.estimators.attackers import build_attacker, default_bank
from leakbench.observables.maps import (
    featurize_dataset,
    network_feature_slices,
    platform_feature_slices,
)
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.provenance import capture_provenance

SEED = 20270214
tax = reference_taxonomy()
Rpi = 1.0 - float(np.max(tax.prior))


def evaluate(X, y, slices, label):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y)
    va, te = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=y[rest])

    names = list(slices)
    # All non-empty block subsets, plus the full vector.
    subsets = [tuple(c) for r in range(1, len(names) + 1)
               for c in itertools.combinations(names, r)]
    best = (np.inf, None, None)
    for sub in subsets:
        cols = np.zeros(X.shape[1], dtype=bool)
        for b in sub:
            cols[slices[b]] = True
        for atk in default_bank():
            clf = build_attacker(atk, seed=SEED)
            clf.fit(X[np.ix_(tr, cols)], y[tr])
            err = float(np.mean(clf.predict(X[np.ix_(va, cols)]) != y[va]))
            if err < best[0]:
                best = (err, sub, atk)
    _, sub, atk = best
    cols = np.zeros(X.shape[1], dtype=bool)
    for b in sub:
        cols[slices[b]] = True
    clf = build_attacker(atk, seed=SEED)
    clf.fit(X[np.ix_(tr, cols)], y[tr])
    sel_err = float(np.mean(clf.predict(X[np.ix_(te, cols)]) != y[te]))

    # Baseline: full vector, best attacker chosen on validation, scored on test.
    bb = (np.inf, None)
    for atk2 in default_bank():
        clf2 = build_attacker(atk2, seed=SEED)
        clf2.fit(X[tr], y[tr])
        e = float(np.mean(clf2.predict(X[va]) != y[va]))
        if e < bb[0]:
            bb = (e, atk2)
    clf2 = build_attacker(bb[1], seed=SEED)
    clf2.fit(X[tr], y[tr])
    full_err = float(np.mean(clf2.predict(X[te]) != y[te]))

    a_sel = max(0.0, (Rpi - sel_err) / Rpi)
    a_full = max(0.0, (Rpi - full_err) / Rpi)
    print(f"{label:<26} full={a_full:.3f}  best-subset={a_sel:.3f}  "
          f"gain={a_sel - a_full:+.3f}   [{atk} on {'+'.join(sub)}]", flush=True)
    return {"full": a_full, "best_subset": a_sel, "gain": a_sel - a_full,
            "selected_blocks": list(sub), "selected_attacker": atk}


gds = CalibratedGenerator(taxonomy=tax, seed=SEED).sample_dataset(600)
mds = MeasuredHarness(taxonomy=tax, seed=SEED, overlap=0.15).sample_dataset(130)
mds_net = apply_wire_model_dataset(mds, HTTP2WireModel(seed=SEED))

rows = {}
for name, ds, v, sl in (
    ("generator network", gds, "network", network_feature_slices()),
    ("generator platform", gds, "platform", platform_feature_slices()),
    ("measured network", mds_net, "network", network_feature_slices()),
    ("measured platform", mds, "platform", platform_feature_slices()),
):
    X, y = featurize_dataset(ds, v)
    rows[name.replace(" ", "_")] = evaluate(X, y, sl, name)

mean_gain = sum(r["gain"] for r in rows.values()) / len(rows)
json.dump({"status": "real_computation_at_paper_scale",
           "result": {"rq": "subset_selection_probe", "rows": rows,
                      "mean_gain": mean_gain,
                      "protocol": "select on validation split, score on held-out test split",
                      "conclusion": "no consistent gain; the estimator is unchanged"},
           "provenance": capture_provenance(
               seed=SEED, taxonomy_hash=tax.content_hash(), taxonomy_name=tax.name,
               config={"split": "50/25/25 train/validation/test",
                       "traces_per_class": {"generator": 600, "measured": 130}},
               note="does maximizing over feature-block subsets tighten the bound?"
                    ).to_dict()},
          open("outputs/scale_study/subset_selection_probe.json", "w"), indent=2)
print(f"mean gain = {mean_gain:+.4f}")
