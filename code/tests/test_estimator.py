"""Leakage-estimator tests (required tests: calibration/estimation correctness,
gate-style decision on controlled examples, reproducibility under fixed seed)."""

import numpy as np
import pytest

from leakbench.estimators.leakage import estimate_leakage
from leakbench.validation.ground_truth import SymmetricChannel


def test_advantage_zero_on_pure_noise():
    # Features independent of labels => advantage clamped to ~0.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8))
    y = rng.integers(0, 4, size=600)
    res = estimate_leakage(X, y, vantage="noise", bootstrap=20, seed=0)
    assert res.advantage <= 0.15  # near zero (finite-sample slack)
    assert 0.0 <= res.advantage <= 1.0


def test_advantage_high_on_separable_channel():
    # p=0 symmetric channel is perfectly informative => advantage ~ 1.
    X, y = SymmetricChannel(K=6, p=0.0, seed=1).sample(200)
    res = estimate_leakage(X, y, vantage="gt", prior=np.full(6, 1 / 6),
                           bootstrap=20, seed=1)
    assert res.advantage > 0.9
    assert res.mi_lower_bound_bits > 2.0


def test_advantage_clamped_and_ci_ordered():
    X, y = SymmetricChannel(K=4, p=0.5, seed=2).sample(300)
    res = estimate_leakage(X, y, vantage="gt", prior=np.full(4, 1 / 4),
                           bootstrap=50, seed=2)
    assert 0.0 <= res.advantage <= 1.0
    lo, hi = res.advantage_ci
    assert lo <= res.advantage + 1e-6
    assert hi >= res.advantage - 1e-6


def test_estimator_reproducible_under_seed():
    X, y = SymmetricChannel(K=5, p=0.3, seed=9).sample(200)
    prior = np.full(5, 1 / 5)
    r1 = estimate_leakage(X, y, prior=prior, bootstrap=30, seed=123)
    r2 = estimate_leakage(X, y, prior=prior, bootstrap=30, seed=123)
    assert r1.advantage == pytest.approx(r2.advantage)
    assert r1.advantage_ci == pytest.approx(r2.advantage_ci)
    assert r1.mi_lower_bound_bits == pytest.approx(r2.mi_lower_bound_bits)


def test_subgroup_profile_populated():
    X, y = SymmetricChannel(K=4, p=0.4, seed=4).sample(400)
    sg = np.array([i % 2 for i in range(len(y))])  # arbitrary 2-way subgroup
    res = estimate_leakage(X, y, prior=np.full(4, 1 / 4), bootstrap=20, seed=4,
                           subgroup_labels=sg, subgroup_names={0: "a", 1: "b"})
    assert set(res.subgroups) <= {"a", "b"}
    for sub in res.subgroups.values():
        assert 0.0 <= sub.advantage <= 1.0


def test_mi_is_reported_as_lower_bound_metadata():
    X, y = SymmetricChannel(K=3, p=0.2, seed=5).sample(150)
    res = estimate_leakage(X, y, prior=np.full(3, 1 / 3), bootstrap=10, seed=5)
    d = res.to_dict()
    assert "lower bound" in d["interpretation"].lower()


def test_sampled_channel_matches_its_own_closed_form():
    """The sampler must realize *exactly* the channel the closed forms describe.

    Regression guard: the escape branch redraws over all K classes, so the
    pass-through probability has to be ``1 - p``. Thresholding on ``a = 1-p+p/K``
    instead realizes a stronger channel (extra ``p(K-1)/K^2`` correct observations),
    and the estimator then appears to beat its own ground truth --- which would
    invalidate the validation figure rather than support it.
    """
    for K, p in ((12, 0.2), (12, 0.6), (6, 0.4), (4, 0.8)):
        ch = SymmetricChannel(K=K, p=p, seed=7)
        X, y = ch.sample(4000)
        obs = np.argmax(X, axis=1)
        empirical_correct = float(np.mean(obs == y))
        analytic_correct = 1.0 - ch.analytic()["bayes_risk"]
        # 48k samples => se < 0.003; 0.01 is comfortably tight enough to catch the
        # p(K-1)/K^2 offset (0.015 at K=12, p=0.2) while staying non-flaky.
        assert empirical_correct == pytest.approx(analytic_correct, abs=0.01)

        # Off-diagonal mass must be uniform at p/K, not concentrated.
        for j in range(K):
            wrong = float(np.mean((obs == j) & (y != j))) / (K - 1) * K
            assert wrong == pytest.approx(p / K, abs=0.02)
