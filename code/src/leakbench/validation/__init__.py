"""Estimator validation against an analytic ground-truth channel."""

from leakbench.validation.ground_truth import (
    SymmetricChannel,
    analytic_bayes_risk,
    analytic_mi_bits,
)

__all__ = ["SymmetricChannel", "analytic_bayes_risk", "analytic_mi_bits"]
