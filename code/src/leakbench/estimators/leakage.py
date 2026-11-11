"""Standardized leakage estimation --- Contribution C3, the algorithm
"Standardized leakage estimation at vantage v".

Given labeled feature vectors ``(X, y)`` from an observation map at vantage ``v``,
a prior ``pi``, an estimator bank, and bootstrap ``B``:

1. stratified 60/40 train/test split;
2. fit each estimator on train, predict on test;
3. ``R̂* = min test error over the bank``; ``Adv = (R_pi - R̂*)/R_pi`` (clamped to [0,1]);
4. ``Î = I(t; ĥt)`` from the best (min-error) confusion matrix;
5. bootstrap the *test set* B times -> percentile CI on ``Adv``;
6. re-run per subgroup -> disparate-vulnerability profile.

Honesty invariants (paper §Estimator, math_to_code_audit.md):
- report the *strongest* attacker (min error = tightest lower bound);
- ``Adv`` and ``Î`` are leakage LOWER bounds ("at least this much leaks");
- never invert a single attacker's accuracy into a privacy upper bound;
- the CI uses the point-estimate's own estimator across resamples (not re-argmin
  per bootstrap sample, which would bias the interval upward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from leakbench.estimators.attackers import build_attacker, default_bank
from leakbench.metrics.information import confusion_mutual_information
from leakbench.utils.logging import get_logger

REFERENCE_SEED = 20270214

logger = get_logger(__name__)


@dataclass
class LeakageResult:
    """The leakage profile at a vantage (paper §Design Stage 5: a profile, not a
    number)."""

    vantage: str
    advantage: float
    advantage_ci: tuple[float, float]
    bayes_risk: float
    mi_lower_bound_bits: float
    accuracy: float
    guessing_risk: float
    best_estimator: str
    per_estimator: dict[str, float]           # advantage per estimator
    n_train: int
    n_test: int
    num_classes: int
    subgroups: dict[str, "LeakageResult"] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "vantage": self.vantage,
            "advantage": self.advantage,
            "advantage_ci": list(self.advantage_ci),
            "bayes_risk": self.bayes_risk,
            "mi_lower_bound_bits": self.mi_lower_bound_bits,
            "accuracy": self.accuracy,
            "guessing_risk": self.guessing_risk,
            "best_estimator": self.best_estimator,
            "per_estimator_advantage": self.per_estimator,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "num_classes": self.num_classes,
            "subgroups": {k: v.to_dict() for k, v in self.subgroups.items()},
            "note": self.note,
            "interpretation": "advantage and MI are LOWER bounds on leakage "
            "('at least this much leaks'); never an upper bound on privacy.",
        }


def _advantage_from_error(error: float, guessing_risk: float) -> float:
    """``Adv = (R_pi - R*)/R_pi``, clamped to [0,1] (Def. Bayes risk / advantage)."""
    if guessing_risk <= 0:
        return 0.0
    adv = (guessing_risk - error) / guessing_risk
    return float(min(1.0, max(0.0, adv)))


def estimate_leakage(
    X: np.ndarray,
    y: np.ndarray,
    *,
    vantage: str = "unknown",
    prior: np.ndarray | None = None,
    bank: Sequence[str] | None = None,
    bootstrap: int = 200,
    test_size: float = 0.4,
    seed: int = REFERENCE_SEED,
    subgroup_labels: np.ndarray | None = None,
    subgroup_names: dict[int, str] | None = None,
) -> LeakageResult:
    """Run the standardized leakage-estimation algorithm at one vantage."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    bank = tuple(bank) if bank is not None else default_bank()
    classes = np.unique(y)
    K = len(classes)

    if prior is None:
        # Empirical prior from label frequencies.
        counts = np.array([(y == c).sum() for c in classes], dtype=float)
        prior = counts / counts.sum()
    else:
        prior = np.asarray(prior, dtype=float)
        if len(prior) != K:
            # Legitimate when a corpus does not exercise every declared class (the
            # prior is a property of the taxonomy, not of the sample), but it changes
            # R_pi and therefore the advantage scale, so it must never pass silently.
            logger.warning(
                "Prior has %d entries but only %d classes are present in y; keeping the "
                "declared prior (R_pi = %.3f). Advantage is reported against the "
                "declared secret distribution, not the observed one.",
                len(prior), K, 1.0 - float(np.max(prior)),
            )
    guessing_risk = float(1.0 - np.max(prior))

    # Step 1: stratified split. Carry indices so subgroups align with the test set.
    idx = np.arange(len(y))
    class_counts = {int(c): int((y == c).sum()) for c in classes}
    too_small = {c: n for c, n in class_counts.items() if n < 2}
    if too_small:
        raise ValueError(
            f"Stratified split needs >= 2 traces per class; classes {sorted(too_small)} "
            f"have {sorted(set(too_small.values()))}. Provide more traces for those "
            f"classes or drop them before estimating."
        )
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Steps 2-4: fit each estimator, keep the strongest (min test error).
    best_error = np.inf
    best_pred: np.ndarray | None = None
    best_name = ""
    per_estimator: dict[str, float] = {}
    for name in bank:
        clf = build_attacker(name, seed=seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        err = float(np.mean(pred != y_te))
        per_estimator[name] = _advantage_from_error(err, guessing_risk)
        if err < best_error:
            best_error, best_pred, best_name = err, pred, name

    assert best_pred is not None
    advantage = _advantage_from_error(best_error, guessing_risk)
    accuracy = float(1.0 - best_error)
    conf = confusion_matrix(y_te, best_pred, labels=classes)
    mi = confusion_mutual_information(conf)

    # Step 5: bootstrap percentile CI on advantage (resample the test set).
    ci = _bootstrap_ci(y_te, best_pred, guessing_risk, bootstrap, seed)

    result = LeakageResult(
        vantage=vantage,
        advantage=advantage,
        advantage_ci=ci,
        bayes_risk=best_error,
        mi_lower_bound_bits=mi,
        accuracy=accuracy,
        guessing_risk=guessing_risk,
        best_estimator=best_name,
        per_estimator=per_estimator,
        n_train=len(tr_idx),
        n_test=len(te_idx),
        num_classes=K,
    )

    # Step 6: subgroup re-estimation on the test predictions.
    if subgroup_labels is not None:
        sg_test = np.asarray(subgroup_labels)[te_idx]
        for sg in np.unique(sg_test):
            mask = sg_test == sg
            if mask.sum() < 5:  # too small to estimate; skip with a note
                continue
            sg_err = float(np.mean(best_pred[mask] != y_te[mask]))
            sg_adv = _advantage_from_error(sg_err, guessing_risk)
            sg_conf = confusion_matrix(y_te[mask], best_pred[mask], labels=classes)
            sg_ci = _bootstrap_ci(y_te[mask], best_pred[mask], guessing_risk,
                                  min(bootstrap, 100), seed)
            name = (subgroup_names or {}).get(int(sg), f"subgroup_{int(sg)}")
            result.subgroups[name] = LeakageResult(
                vantage=vantage,
                advantage=sg_adv,
                advantage_ci=sg_ci,
                bayes_risk=sg_err,
                mi_lower_bound_bits=confusion_mutual_information(sg_conf),
                accuracy=1.0 - sg_err,
                guessing_risk=guessing_risk,
                best_estimator=best_name,
                per_estimator={best_name: sg_adv},
                n_train=len(tr_idx),
                n_test=int(mask.sum()),
                num_classes=K,
                note="subgroup (disparate-vulnerability) profile",
            )
    return result


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    guessing_risk: float,
    B: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on advantage over test-set resamples (Alg. step 5)."""
    if B <= 0 or len(y_true) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed + 1)
    n = len(y_true)
    advs = np.empty(B, dtype=float)
    for b in range(B):
        take = rng.integers(0, n, size=n)
        err = float(np.mean(y_pred[take] != y_true[take]))
        advs[b] = _advantage_from_error(err, guessing_risk)
    lo = float(np.percentile(advs, 100 * alpha / 2))
    hi = float(np.percentile(advs, 100 * (1 - alpha / 2)))
    return (lo, hi)


def estimate_leakage_projection(
    X_network: np.ndarray,
    y: np.ndarray,
    projection: str,
    **kwargs,
) -> LeakageResult:
    """Estimate leakage on a named projection of a base-vantage feature matrix
    (§Projections; used to reproduce deployed attacks on common traces)."""
    from leakbench.observables.projections import PROJECTIONS, project_features

    proj = PROJECTIONS[projection]
    Xp = project_features(X_network, projection)
    kwargs.setdefault("vantage", f"{proj.base_vantage}:{projection}")
    return estimate_leakage(Xp, y, **kwargs)
