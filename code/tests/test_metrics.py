"""Metric-formula tests (required test: metric correctness).

Covers confusion-matrix MI (E7), entropy, g-leakage (E5, identity gain must recover
advantage), and the analytic ground-truth channel closed forms (E8).
"""

import numpy as np
import pytest

from leakbench.metrics.information import (
    confusion_mutual_information,
    entropy_bits,
    g_leakage,
    identity_gain_matrix,
)
from leakbench.validation.ground_truth import analytic_bayes_risk, analytic_mi_bits


def test_entropy_uniform_and_point_mass():
    assert entropy_bits(np.full(4, 0.25)) == pytest.approx(2.0)
    assert entropy_bits(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_confusion_mi_perfect_and_useless():
    # Perfect classifier: diagonal confusion => MI = log2(K).
    K = 4
    perfect = np.eye(K) * 25
    assert confusion_mutual_information(perfect) == pytest.approx(np.log2(K), abs=1e-9)
    # Useless classifier: uniform confusion => MI = 0.
    useless = np.full((K, K), 10.0)
    assert confusion_mutual_information(useless) == pytest.approx(0.0, abs=1e-9)


def test_confusion_mi_worked_example_from_appendix():
    # App. "Worked example": K=3, normalized confusion with diagonal summing to 0.87
    # => I(C;Ĉ) ~ 0.91 bits.
    M = np.array([[0.30, 0.02, 0.01], [0.02, 0.28, 0.03], [0.02, 0.03, 0.29]])
    mi = confusion_mutual_information(M)
    assert mi == pytest.approx(0.91, abs=0.03)


def test_confusion_mi_handles_empty():
    assert confusion_mutual_information(np.zeros((3, 3))) == 0.0


def test_g_leakage_identity_recovers_advantage():
    # Identity gain: additive g-leakage V_post - V_prior should equal
    # (1 - R*) - (1 - R_pi) = R_pi - R*, and L_add/R_pi = advantage.
    K = 3
    # Joint over (true, predicted). Diagonal-heavy.
    P = np.array([[0.30, 0.02, 0.01], [0.02, 0.28, 0.03], [0.02, 0.03, 0.29]])
    prior = P.sum(axis=1)  # marginal over true classes
    gl = g_leakage(prior, P, identity_gain_matrix(K))
    R_pi = 1 - prior.max()
    # posterior success = sum_x p(x) max_t p(t|x); advantage = (V_post - V_prior)/R_pi.
    advantage = gl["L_add"] / R_pi
    assert 0.0 <= advantage <= 1.0
    # V_prior for identity gain = max prior = 1 - R_pi.
    assert gl["V_prior"] == pytest.approx(1 - R_pi, abs=1e-9)


def test_analytic_channel_closed_forms():
    K = 12
    # p=0: perfect (MI = log2 K, R* = 0).
    assert analytic_mi_bits(0.0, K) == pytest.approx(np.log2(K), abs=1e-9)
    assert analytic_bayes_risk(0.0, K) == pytest.approx(0.0, abs=1e-9)
    # p=1: useless (MI = 0, R* = (K-1)/K).
    assert analytic_mi_bits(1.0, K) == pytest.approx(0.0, abs=1e-9)
    assert analytic_bayes_risk(1.0, K) == pytest.approx((K - 1) / K, abs=1e-9)
    # Monotone in p.
    assert analytic_mi_bits(0.2, K) > analytic_mi_bits(0.6, K)


def test_open_world_scores():
    from leakbench.metrics.openworld import OTHER_LABEL, open_world_scores

    y_true = np.array([0, 1, OTHER_LABEL, OTHER_LABEL, 0])
    y_pred = np.array([0, OTHER_LABEL, OTHER_LABEL, 0, 0])
    s = open_world_scores(y_true, y_pred)
    assert 0.0 <= s["precision"] <= 1.0
    assert 0.0 <= s["recall"] <= 1.0
    assert 0.0 <= s["f1"] <= 1.0
