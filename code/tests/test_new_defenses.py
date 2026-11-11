"""Generative-shaping (Surakav analogue) and DP-shaping (NetShaper analogue) tests.

The DP tests check the properties the privacy claim rests on: a data-independent
release grid, a mechanism that actually adds noise, and the analytic advantage
ceiling holding on measured data.
"""

import math

import numpy as np
import pytest

from leakbench.data.generator import CalibratedGenerator
from leakbench.defenses import get_defense, list_defenses
from leakbench.defenses.dp_shaping import (
    DPShapingDefense,
    discrete_laplace,
    dp_advantage_ceiling,
)
from leakbench.defenses.generative import GenerativeShapingDefense
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset
from leakbench.taxonomy import reference_taxonomy


@pytest.fixture(scope="module")
def dataset():
    return CalibratedGenerator(taxonomy=reference_taxonomy(), seed=11).sample_dataset(20)


def test_both_defenses_are_registered():
    assert "gen-shaping" in list_defenses()
    assert "dp-shaping" in list_defenses()
    assert isinstance(get_defense("gen-shaping"), GenerativeShapingDefense)
    assert isinstance(get_defense("dp-shaping"), DPShapingDefense)


# ------------------------------------------------------------- generative shaping
def test_generative_shaping_sends_at_least_the_payload(dataset):
    d = GenerativeShapingDefense()
    res = d.apply(dataset, seed=3)
    for original, shaped in zip(dataset.traces, res.dataset.traces):
        assert shaped.total_sizes().sum() >= original.total_sizes().sum() - 1e-6
    assert res.bandwidth_overhead > 0


def test_generative_shaping_decoy_is_independent_of_the_label(dataset):
    """The decoy class must not track the true label (that would leak by design)."""
    d = GenerativeShapingDefense()
    res = d.apply(dataset, seed=3)
    true = np.array([t.label for t in dataset.traces])
    decoy = np.array([t.metadata["decoy_class"] for t in res.dataset.traces])
    # Chance agreement for 12 classes is 1/12; anything close to 1.0 means the decoy
    # was chosen from the secret.
    assert float(np.mean(true == decoy)) < 0.30


def test_generative_shaping_reduces_advantage(dataset):
    tax = reference_taxonomy()
    X0, y0 = featurize_dataset(dataset, "network")
    base = estimate_leakage(X0, y0, prior=tax.prior, bootstrap=20, seed=3).advantage
    res = GenerativeShapingDefense().apply(dataset, seed=3)
    X1, y1 = featurize_dataset(res.dataset, "network")
    shaped = estimate_leakage(X1, y1, prior=tax.prior, bootstrap=20, seed=3).advantage
    assert shaped < base


def test_generative_shaping_refuses_to_run_unfitted(dataset):
    d = GenerativeShapingDefense()
    with pytest.raises(RuntimeError, match="fitted"):
        d.transform(dataset.traces[0], np.random.default_rng(0))


# --------------------------------------------------------------------- DP shaping
def test_discrete_laplace_is_centred_and_scales():
    rng = np.random.default_rng(0)
    small = discrete_laplace(rng, 2.0, size=20000)
    large = discrete_laplace(rng, 20.0, size=20000)
    assert abs(float(np.mean(small))) < 0.5
    assert float(np.std(large)) > float(np.std(small))


def test_dp_ceiling_matches_closed_form():
    # Uniform prior over K: ceiling = (e^eps - 1)/(K - 1).
    for eps in (0.1, 0.5, 1.0):
        assert dp_advantage_ceiling(eps, 12) == pytest.approx(
            min(1.0, (math.exp(eps) - 1) / 11), rel=1e-9
        )
    assert dp_advantage_ceiling(0.0, 12) == 0.0
    assert dp_advantage_ceiling(50.0, 12) == 1.0
    # A larger adjacency distance can only weaken (raise) the bound.
    assert dp_advantage_ceiling(0.1, 12, distance=5) > dp_advantage_ceiling(0.1, 12)


def test_dp_release_grid_is_data_independent(dataset):
    """Every trace must produce the same number of shaping windows.

    A data-dependent stopping rule would leak outside the mechanism; this test is the
    regression guard for that bug.
    """
    d = DPShapingDefense(epsilon=4.0, windows=16, window_s=1.0)
    res = d.apply(dataset, seed=7)
    assert {t.metadata["windows"] for t in res.dataset.traces} == {16}


def test_dp_shaping_respects_the_advantage_ceiling(dataset):
    tax = reference_taxonomy()
    eps = 0.5
    d = DPShapingDefense(epsilon=eps, windows=32, window_s=None, adjacency="all_pairs")
    res = d.apply(dataset, seed=7)
    X, y = featurize_dataset(res.dataset, "network")
    leak = estimate_leakage(X, y, prior=tax.prior, bootstrap=20, seed=7)
    assert leak.advantage <= dp_advantage_ceiling(eps, tax.prior) + 1e-9


def test_dp_shaping_advantage_grows_with_epsilon(dataset):
    """More budget (less noise) must not protect better --- a monotonicity sanity check."""
    tax = reference_taxonomy()
    advs = []
    for eps in (1.0, 256.0):
        d = DPShapingDefense(epsilon=eps, windows=32, window_s=None,
                             adjacency="per_window")
        res = d.apply(dataset, seed=7)
        X, y = featurize_dataset(res.dataset, "network")
        advs.append(estimate_leakage(X, y, prior=tax.prior, bootstrap=20, seed=7).advantage)
    assert advs[1] > advs[0]


def test_dp_shaping_reports_overflow_and_params(dataset):
    d = DPShapingDefense(epsilon=4.0, windows=8, window_s=1e-3)  # deliberately tiny horizon
    d.apply(dataset, seed=7)
    params = d.params()
    assert params["windows_W"] == 8
    assert params["adjacency"] == "all_pairs"
    assert params["n_transformed"] == len(dataset)
    # A horizon this short cannot drain the payload, and that must be *reported*.
    assert params["overflow_traces"] > 0


def test_dp_rejects_unknown_adjacency():
    with pytest.raises(ValueError, match="adjacency"):
        DPShapingDefense(adjacency="whatever")


# ------------------------------------------- multi-user mixing (planned exp. iv)
def test_mixing_keeps_the_targets_label_and_grows_the_stream(dataset):
    from leakbench.defenses.mixing import MixingDefense

    res = MixingDefense(n_users=4).apply(dataset, seed=3)
    assert len(res.dataset) == len(dataset)
    for original, mixed in zip(dataset.traces, res.dataset.traces):
        # The secret is still THIS user's task, not the mix's.
        assert mixed.label == original.label
        assert mixed.n >= original.n
    assert res.bandwidth_overhead > 0  # the OBSERVED stream grows


def test_mixing_with_one_user_is_a_no_op(dataset):
    from leakbench.defenses.mixing import MixingDefense

    res = MixingDefense(n_users=1).apply(dataset, seed=3)
    for original, mixed in zip(dataset.traces, res.dataset.traces):
        assert mixed.n == original.n


def test_mixing_draws_co_users_from_other_classes(dataset):
    """Mixing with the same task would flatter the defense."""
    import numpy as np

    from leakbench.defenses.mixing import MixingDefense

    d = MixingDefense(n_users=3, cross_class_only=True)
    d.fit(dataset, np.random.default_rng(0))
    target = dataset.traces[0]
    mixed = d.transform(target, np.random.default_rng(1))
    # More calls arrived than the target had, and they came from somewhere else.
    assert mixed.n > target.n
    assert mixed.metadata["n_users"] == 3


def test_mixing_reduces_advantage(dataset):
    from leakbench.defenses.mixing import MixingDefense

    tax = reference_taxonomy()
    X0, y0 = featurize_dataset(dataset, "network")
    base = estimate_leakage(X0, y0, prior=tax.prior, bootstrap=20, seed=3).advantage
    res = MixingDefense(n_users=4).apply(dataset, seed=3)
    X1, y1 = featurize_dataset(res.dataset, "network")
    mixed = estimate_leakage(X1, y1, prior=tax.prior, bootstrap=20, seed=3).advantage
    assert mixed < base


def test_mixing_refuses_to_run_unfitted(dataset):
    import numpy as np

    from leakbench.defenses.mixing import MixingDefense

    with pytest.raises(RuntimeError, match="fitted"):
        MixingDefense(n_users=4).transform(dataset.traces[0], np.random.default_rng(0))
