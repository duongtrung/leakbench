"""Observation-map, refinement-order, and defense tests (required: aggregator-style
structural tests; defense transforms)."""

import numpy as np

from leakbench.data.generator import CalibratedGenerator
from leakbench.defenses import get_defense, list_defenses
from leakbench.observables.maps import (
    VantageOrder,
    network_map,
    platform_map,
    provider_map,
)
from leakbench.observables.projections import project_features


def _sample_trace():
    return CalibratedGenerator(seed=3).sample_dataset(1).traces[0]


def test_map_dimensions_fixed():
    t = _sample_trace()
    assert network_map(t).shape == (31,)
    assert platform_map(t).shape == (80,)
    # provider = platform(80) + network(31) + per-category aggregates(16)
    assert provider_map(t).shape == (127,)


def test_provider_map_contains_shallower_maps():
    # Refinement by construction: O_prov's first blocks ARE O_plat then O_net.
    t = _sample_trace()
    prov = provider_map(t)
    assert np.allclose(prov[:80], platform_map(t))
    assert np.allclose(prov[80:111], network_map(t))


def test_platform_map_has_no_size_features_but_network_does():
    # Two traces identical in categories/timing but different in sizes must have the
    # SAME platform features and DIFFERENT network features (net & plat incomparable).
    from leakbench.data.schemas import ToolCall, Trace

    small = Trace([ToolCall("search", 100, 100, 0.0), ToolCall("fetch", 100, 100, 0.5)], 0)
    big = Trace([ToolCall("search", 5000, 5000, 0.0), ToolCall("fetch", 5000, 5000, 0.5)], 0)
    assert np.allclose(platform_map(small), platform_map(big))
    assert not np.allclose(network_map(small), network_map(big))


def test_vantage_order_relations():
    order = VantageOrder.canonical()
    assert order.dominates("providers", "network")
    assert order.dominates("providers", "platform")
    # net & plat are incomparable.
    assert not order.dominates("network", "platform")
    assert not order.dominates("platform", "network")


def test_projections_reduce_dimension():
    t = _sample_trace()
    X = network_map(t).reshape(1, -1)
    for name in ("keylogger", "whisper_leak", "agentprint", "size_only", "timing_only"):
        Xp = project_features(X, name)
        assert Xp.shape[1] <= X.shape[1]
        assert Xp.shape[1] > 0
    # AgentPrint keeps (essentially) all of O_net.
    assert project_features(X, "agentprint").shape[1] == 31


def test_all_defenses_registered_and_apply():
    ds = CalibratedGenerator(seed=1).sample_dataset(3)
    expected = {"none", "pad-512", "shuffle", "constant-rate", "cover", "pad-max", "full"}
    assert expected <= set(list_defenses())
    for name in expected:
        res = get_defense(name).apply(ds, seed=1)
        assert len(res.dataset) == len(ds)
        assert res.bandwidth_overhead >= -1e-9


def test_padmax_makes_sizes_constant():
    ds = CalibratedGenerator(seed=1).sample_dataset(3)
    res = get_defense("pad-max").apply(ds, seed=1)
    sizes = np.concatenate([t.request_sizes() for t in res.dataset.traces])
    assert np.unique(sizes).size == 1  # all requests padded to the same size


def test_pad512_costs_bandwidth_padmax_costs_more():
    ds = CalibratedGenerator(seed=1).sample_dataset(5)
    bw512 = get_defense("pad-512").apply(ds, seed=1).bandwidth_overhead
    bwmax = get_defense("pad-max").apply(ds, seed=1).bandwidth_overhead
    assert bwmax > bw512 >= 0
