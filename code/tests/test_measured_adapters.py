"""Measured-harness and adapter tests (real local tools; adapter interface)."""

import numpy as np
import pytest

from leakbench.data.adapters import DummyAdapter, ReActAdapter
from leakbench.data.measured import MeasuredHarness
from leakbench.observables.maps import featurize_dataset
from leakbench.taxonomy import reference_taxonomy


def test_measured_harness_produces_real_traces():
    harness = MeasuredHarness(taxonomy=reference_taxonomy(), seed=1, overlap=0.5)
    ds = harness.sample_dataset(4)
    assert len(ds) == 12 * 4
    assert ds.source == "measured"
    for t in ds.traces:
        assert t.n >= 2
        assert np.all(t.request_sizes() > 0)
        assert np.all(t.response_sizes() > 0)


def test_measured_vantage_ordering_holds():
    # Providers refine both; must not leak less than network/platform on measured data.
    from leakbench.estimators.leakage import estimate_leakage

    harness = MeasuredHarness(seed=2, overlap=0.4)
    ds = harness.sample_dataset(40)
    tax = reference_taxonomy()
    adv = {}
    for v in ("network", "platform", "providers"):
        X, y = featurize_dataset(ds, v)
        adv[v] = estimate_leakage(X, y, vantage=v, prior=tax.prior,
                                  bootstrap=20, seed=2).advantage
    assert adv["providers"] + 0.05 >= adv["platform"]
    assert adv["providers"] + 0.05 >= adv["network"]


def test_dummy_adapter_end_to_end():
    tax = reference_taxonomy()
    adapter = DummyAdapter(taxonomy=tax, seed=1)
    ds = adapter.sample_dataset(tax, 3)
    assert len(ds) == 12 * 3
    X, y = featurize_dataset(ds, "platform")
    assert X.shape[0] == len(ds)


def test_real_adapters_are_documented_stubs():
    # Real framework adapters must raise NotImplementedError (never silently fake).
    with pytest.raises(NotImplementedError):
        list(ReActAdapter().run_task(0))
