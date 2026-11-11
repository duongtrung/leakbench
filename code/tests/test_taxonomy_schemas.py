"""Taxonomy + schema validation tests (required tests #2, #3)."""

import numpy as np
import pytest

from leakbench.data.schemas import ToolCall, Trace, TraceDataset
from leakbench.taxonomy import (
    NUM_TOOLS,
    reference_taxonomy,
    scaled_taxonomy,
    zipf_prior,
)


def test_reference_taxonomy_entropy():
    tax = reference_taxonomy()
    assert tax.num_classes == 12
    assert len(tax.families) == 4
    # H(C) = log2(12) for the uniform prior.
    assert tax.entropy_bits() == pytest.approx(np.log2(12), abs=1e-9)
    assert tax.guessing_risk() == pytest.approx(1 - 1 / 12, abs=1e-9)


def test_taxonomy_hash_is_deterministic_and_version_sensitive():
    assert reference_taxonomy().content_hash() == reference_taxonomy().content_hash()
    assert reference_taxonomy().content_hash() != scaled_taxonomy(16).content_hash()


def test_num_tool_categories():
    assert NUM_TOOLS == 8


def test_zipf_prior_sums_to_one():
    p = zipf_prior(12)
    assert p.sum() == pytest.approx(1.0)
    # Skewed: p[0] > p[-1].
    assert p[0] > p[-1]


def test_toolcall_validation():
    ToolCall("search", 100.0, 200.0, 0.0)  # ok
    with pytest.raises(ValueError):
        ToolCall("not_a_tool", 100.0, 200.0, 0.0)
    with pytest.raises(ValueError):
        ToolCall("search", 0.0, 200.0, 0.0)  # size must be > 0
    with pytest.raises(ValueError):
        ToolCall("search", 100.0, 200.0, -1.0)  # gap must be >= 0


def test_trace_accessors_and_roundtrip():
    calls = [ToolCall("search", 100, 200, 0.0), ToolCall("fetch", 50, 400, 0.3)]
    tr = Trace(calls=calls, label=0, family="research")
    assert tr.n == 2
    assert tr.total_sizes().tolist() == [300, 450]
    assert tr.category_counts().shape == (NUM_TOOLS,)
    rec = tr.to_record()
    tr2 = Trace.from_record(rec)
    assert tr2.n == tr.n and tr2.label == tr.label


def test_dataset_length_band_median_split():
    calls_short = [ToolCall("search", 100, 200, 0.0)]
    calls_long = [ToolCall("search", 100, 200, 0.0)] * 5
    traces = [Trace(calls_short, 0), Trace(calls_long, 1),
              Trace(calls_short, 0), Trace(calls_long, 1)]
    ds = TraceDataset(traces, "t", "h")
    band = ds.subgroup_length_band()
    assert set(band.tolist()) <= {0, 1}
