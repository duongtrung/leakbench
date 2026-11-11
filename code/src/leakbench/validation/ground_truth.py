"""Ground-truth validation channel (paper §Impl "Ground-truth harness", §Eval RQ5).

A synthetic symmetric channel whose mutual information and Bayes risk are known in
**closed form** as a function of a crossover parameter ``p``. A correct estimator
must track the analytic curve; this is a precondition for trusting any cross-system
number the benchmark produces (paper §Estimator "Validation requirement").

Definition (E8): for crossover ``p`` over ``K`` uniform classes, the observation
equals the true class with probability ``a = 1 - p + p/K`` and a uniform-random
class otherwise. Closed forms:

    R*  = 1 - (1 - p + p/K) = p (K-1)/K
    MI  = log2(K) - H(row),  H(row) = -a log2 a - (K-1)(p/K) log2(p/K)

At ``p = 0``: perfect channel (R* = 0, MI = log2 K).
At ``p = 1``: useless channel (R* = R_pi = (K-1)/K, MI = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _correct_prob(p: float, K: int) -> float:
    """``a = 1 - p + p/K`` --- probability the observation equals the true class."""
    return 1.0 - p + p / K


def analytic_bayes_risk(p: float, K: int) -> float:
    """Closed-form Bayes risk ``R* = 1 - a`` for the symmetric channel."""
    return float(1.0 - _correct_prob(p, K))


def analytic_mi_bits(p: float, K: int) -> float:
    """Closed-form mutual information ``MI = log2(K) - H(row)`` (bits)."""
    if K <= 1:
        return 0.0
    a = _correct_prob(p, K)
    off = p / K  # probability of each of the (K-1) wrong classes
    terms = []
    if a > 0:
        terms.append(-a * np.log2(a))
    if off > 0:
        terms.append(-(K - 1) * off * np.log2(off))
    h_row = float(np.sum(terms))
    return float(np.log2(K) - h_row)


@dataclass
class SymmetricChannel:
    """Sampler for the analytic validation channel.

    Emits feature vectors (one-hot of the observed class, with a little jitter so a
    learned classifier has a continuous input) labeled by the true class.
    """

    K: int
    p: float
    seed: int = 20270214

    def sample(self, n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample the channel *exactly* as the closed forms above define it.

        The escape branch redraws uniformly over all ``K`` classes, so it can land
        back on ``t``. The pass-through probability must therefore be ``1 - p``, not
        ``a``: the two paths compose to ``P(obs = t) = (1-p) + p/K = a``. Thresholding
        on ``a`` instead would realize a *stronger* channel than the analytic curve
        (an extra ``p(K-1)/K^2`` of correct observations), which shows up as the
        estimator apparently beating its own ground truth.
        """
        rng = np.random.default_rng(self.seed)
        X = []
        y = []
        for t in range(self.K):
            for _ in range(n_per_class):
                if rng.random() >= self.p:
                    obs = t
                else:
                    obs = int(rng.integers(self.K))
                vec = np.zeros(self.K, dtype=float)
                vec[obs] = 1.0
                vec += rng.normal(0.0, 1e-3, size=self.K)  # tiny jitter
                X.append(vec)
                y.append(t)
        return np.asarray(X), np.asarray(y)

    def analytic(self) -> dict[str, float]:
        return {
            "p": self.p,
            "K": self.K,
            "bayes_risk": analytic_bayes_risk(self.p, self.K),
            "mi_bits": analytic_mi_bits(self.p, self.K),
        }
