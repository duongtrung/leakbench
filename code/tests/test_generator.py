"""Calibrated generator + wire-model tests (required test: synthetic data generation)."""

import numpy as np

from leakbench.data.generator import CalibratedGenerator
from leakbench.data.wire import HTTP2WireModel, apply_tls_framing, tls_frame_size
from leakbench.taxonomy import reference_taxonomy


def test_generator_shapes_and_positivity():
    gen = CalibratedGenerator(taxonomy=reference_taxonomy(), seed=1)
    ds = gen.sample_dataset(5)
    assert len(ds) == 12 * 5
    for t in ds.traces:
        assert t.n >= 2  # n <- max(2, Poisson)
        assert np.all(t.request_sizes() > 0)
        assert np.all(t.response_sizes() > 0)
        assert t.gaps()[0] == 0.0  # first call has no preceding gap
        assert np.all(t.gaps() >= 0)


def test_generator_deterministic_under_seed():
    a = CalibratedGenerator(seed=42).sample_dataset(3)
    b = CalibratedGenerator(seed=42).sample_dataset(3)
    # Same seed => identical first trace.
    ta, tb = a.traces[0], b.traces[0]
    assert ta.n == tb.n
    assert np.allclose(ta.total_sizes(), tb.total_sizes())


def test_generator_profiles_signature_upweighted():
    gen = CalibratedGenerator(seed=1)
    # Each class profile should put more mass on its family's signature tools than a
    # flat 1/8 baseline on average.
    from leakbench.taxonomy import TOOL_INDEX
    for ci, cls in enumerate(gen.taxonomy.classes):
        p = gen.profiles[ci].tool_dist
        sig_mass = sum(p[TOOL_INDEX[t]] for t in cls.signature_tools)
        assert sig_mass > 2 / 8  # signature tools carry above-uniform mass


def test_tls_framing_monotone():
    # Larger payloads never frame smaller.
    assert tls_frame_size(100) < tls_frame_size(100000)
    # A >16KB payload spans multiple records => extra overhead.
    assert tls_frame_size(20000) > 20000


def test_http2_wire_model_runs():
    gen = CalibratedGenerator(seed=1)
    ds = gen.sample_dataset(2)
    model = HTTP2WireModel(seed=1)
    wired = model.apply(ds.traces[0])
    assert wired.n >= 1
    assert np.all(wired.total_sizes() > 0)


def test_apply_tls_framing_preserves_categories():
    gen = CalibratedGenerator(seed=1)
    t = gen.sample_dataset(1).traces[0]
    framed = apply_tls_framing(t)
    assert [c.category for c in framed.calls] == [c.category for c in t.calls]
