"""The attacker bank (paper §Impl "Estimator bank", §Eval "Attacker robustness").

Core bank for the estimation algorithm: {random forest, k-NN}. Robustness family
adds {histogram gradient boosting, logistic regression}. Hyperparameters are fixed
across all conditions (App. "Equation Audit") so differences reflect the observable,
not estimator tuning:

- Random forest: 200 trees, default depth, seed 20270214.
- k-NN: k = 7.

k-NN plays the F-BLEAU Bayes-risk-estimator role (consistent for R* as data grows)
and is treated as a *conservative floor*, never inverted into a privacy guarantee.
"""

from __future__ import annotations

from typing import Callable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REFERENCE_SEED = 20270214


def _random_forest(seed: int = REFERENCE_SEED) -> ClassifierMixin:
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


def _knn(seed: int = REFERENCE_SEED) -> ClassifierMixin:
    # F-BLEAU-style nearest-neighbor Bayes-risk estimator; standardize features so
    # distances are meaningful across heterogeneous feature blocks.
    return Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7))])


def _hgb(seed: int = REFERENCE_SEED) -> ClassifierMixin:
    return HistGradientBoostingClassifier(random_state=seed)


def _logreg(seed: int = REFERENCE_SEED) -> ClassifierMixin:
    # Multinomial handling is scikit-learn's default for multiclass problems; the
    # explicit ``multi_class`` argument was deprecated in 1.5 and removed in 1.7, so
    # passing it would raise on current scikit-learn.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def _mlp(seed: int = REFERENCE_SEED) -> ClassifierMixin:
    """A learned attacker for the raw-sequence representation (planned experiment iii).

    Used with :mod:`leakbench.observables.sequences`, whose input is an ordered padded
    per-call matrix rather than a histogram summary. Kept dependency-free on purpose:
    an artifact reviewer should not have to install a deep-learning stack to reproduce
    a leakage number. ``early_stopping`` holds out 10% of train as a validation split,
    so the reported test advantage is not tuned on the test set.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                                  early_stopping=True, n_iter_no_change=15,
                                  random_state=seed)),
        ]
    )


ATTACKER_FACTORIES: dict[str, Callable[[int], ClassifierMixin]] = {
    "random_forest": _random_forest,
    "knn": _knn,
    "hist_gbdt": _hgb,
    "logistic": _logreg,
    "mlp": _mlp,
}


def build_attacker(name: str, seed: int = REFERENCE_SEED) -> ClassifierMixin:
    if name not in ATTACKER_FACTORIES:
        raise ValueError(f"Unknown attacker {name!r}; available: {sorted(ATTACKER_FACTORIES)}.")
    return ATTACKER_FACTORIES[name](seed)


def default_bank() -> tuple[str, ...]:
    """The core estimator bank used by the leakage algorithm (RF + kNN)."""
    return ("random_forest", "knn")


def robustness_bank() -> tuple[str, ...]:
    """The four attacker families for the robustness check (RQ6)."""
    return ("random_forest", "hist_gbdt", "logistic", "knn")


def sequence_bank() -> tuple[str, ...]:
    """Attackers used over the raw-sequence representation.

    The random forest is kept alongside the MLP because on a flattened ordered matrix
    a tree ensemble is a strong and much cheaper baseline; reporting the max over both
    keeps the bound tight without pretending one architecture is definitive.
    """
    return ("mlp", "random_forest")
