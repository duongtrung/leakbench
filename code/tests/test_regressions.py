"""Regression guards for defects found by auditing this codebase.

Each test pins a specific bug that was fixed, so it cannot come back silently. The
bugs themselves are described in `audits/implementation_gaps.md` §9.
"""

import numpy as np
import pytest

from leakbench.data.generator import CalibratedGenerator
from leakbench.data.measured import MeasuredHarness
from leakbench.data.schemas import ToolCall, Trace
from leakbench.data.wire import HTTP2WireModel, apply_wire_model_dataset
from leakbench.estimators.attackers import build_attacker, robustness_bank
from leakbench.estimators.leakage import estimate_leakage
from leakbench.experiments.config import ExperimentConfig
from leakbench.experiments.runners import run_subgroup
from leakbench.metrics.openworld import OTHER_LABEL
from leakbench.observables.maps import (
    NETWORK_DIM,
    PLATFORM_DIM,
    PROVIDER_DIM,
    network_map,
    platform_map,
    provider_map,
)
from leakbench.observables.projections import PROJECTIONS
from leakbench.taxonomy import reference_taxonomy


def _trace(label=0, n=5):
    calls = [ToolCall(category="search", request_size=100.0 + i,
                      response_size=1000.0 + i, gap=0.0 if i == 0 else 0.2)
             for i in range(n)]
    return Trace(calls=calls, label=label, family="research")


# --- wire model: noise must be independent of the trace's label -------------
def test_wire_model_noise_is_independent_across_traces():
    """Seeding the wire RNG from (label, length) made the coalescing pattern a
    function of the secret, inflating the network advantage."""
    model = HTTP2WireModel(seed=1)
    shaped = [model.apply(_trace(label=3, n=8)) for _ in range(6)]
    lengths = {t.n for t in shaped}
    assert len(lengths) > 1, "identical inputs must still get independent wire noise"


def test_wire_model_dataset_pass_is_reproducible():
    ds = CalibratedGenerator(taxonomy=reference_taxonomy(), seed=5).sample_dataset(3)
    model = HTTP2WireModel(seed=7)
    a = apply_wire_model_dataset(ds, model)
    b = apply_wire_model_dataset(ds, model)
    for x, y in zip(a.traces, b.traces):
        assert np.allclose(x.total_sizes(), y.total_sizes())


def test_wire_model_does_not_leak_label_into_length():
    """Shaped trace length must not be a deterministic function of the label."""
    model = HTTP2WireModel(seed=2)
    by_label = {}
    for label in range(4):
        by_label[label] = {model.apply(_trace(label=label, n=10)).n for _ in range(5)}
    assert all(len(v) > 1 for v in by_label.values())


# --- attacker bank: every family must build on the installed scikit-learn ----
@pytest.mark.parametrize("name", robustness_bank())
def test_every_attacker_family_builds_and_fits(name):
    """`LogisticRegression(multi_class=...)` was removed in scikit-learn 1.7."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.repeat(np.arange(6), 10)
    clf = build_attacker(name, seed=0)
    clf.fit(X, y)
    assert len(clf.predict(X)) == 60


# --- feature widths: projections must track the maps ------------------------
def test_feature_dims_match_the_maps():
    t = _trace()
    assert len(network_map(t)) == NETWORK_DIM
    assert len(platform_map(t)) == PLATFORM_DIM
    assert len(provider_map(t)) == PROVIDER_DIM


def test_every_projection_mask_fits_its_base_vantage():
    for name, proj in PROJECTIONS.items():
        idx = proj.indices()
        dim = NETWORK_DIM if proj.base_vantage == "network" else PLATFORM_DIM
        assert idx.size > 0, f"{name} selects no features"
        assert idx.max() < dim, f"{name} indexes past its base vantage"


# --- estimator: loud failure instead of a confusing sklearn error -----------
def test_estimator_rejects_singleton_classes():
    X = np.random.default_rng(0).normal(size=(5, 3))
    y = np.array([0, 0, 1, 2, 2])           # class 1 has a single member
    with pytest.raises(ValueError, match=">= 2 traces per class"):
        estimate_leakage(X, y, bootstrap=5)


def test_estimator_warns_when_prior_and_sample_disagree(caplog):
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.repeat(np.arange(4), 10)
    prior = np.full(12, 1 / 12)              # declared taxonomy is larger than the sample
    with caplog.at_level("WARNING"):
        res = estimate_leakage(X, y, prior=prior, bootstrap=5)
    assert any("Prior has 12 entries" in r.getMessage() for r in caplog.records)
    assert res.num_classes == 4
    # R_pi comes from the declared taxonomy, not from the four observed classes.
    assert res.guessing_risk == pytest.approx(11 / 12)


# --- subgroup runner: must survive labels outside the taxonomy --------------
def test_subgroup_handles_out_of_taxonomy_labels(monkeypatch):
    """An open-world 'other' (-1) used to crash the family lookup."""
    from leakbench.experiments import runners

    tax = reference_taxonomy()
    ds = CalibratedGenerator(taxonomy=tax, seed=3).sample_dataset(10)
    for t in ds.traces[:20]:
        t.label = OTHER_LABEL
        t.family = None
    monkeypatch.setattr(runners, "build_dataset", lambda cfg: (ds, tax))
    res = run_subgroup(ExperimentConfig(traces_per_class=10, bootstrap=5))
    assert "other" in res["by_family"]


# --- measured harness: gaps must contain real service time ------------------
def test_measured_gaps_are_inter_arrival_not_loop_overhead():
    ds = MeasuredHarness(seed=4, overlap=0.5).sample_dataset(3)
    gaps = np.concatenate([t.gaps()[1:] for t in ds.traces if t.n > 1])
    # Inter-arrival gaps include each tool's real service time; the old end-to-start
    # definition produced ~2e-6 s of pure loop overhead.
    assert float(np.median(gaps)) > 5e-6
