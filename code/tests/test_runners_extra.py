"""Tests for the scale/fidelity runners added for the paper-scale study.

All run at deliberately tiny sizes: they check the runner's *contract* (shape of the
result, invariants that must hold at any size), not the paper's magnitudes, which
require the full-scale run on a server.
"""

import numpy as np
import pytest

from leakbench.experiments.config import ExperimentConfig
from leakbench.experiments.runners import (
    run_attacker_robustness,
    run_coarse_secret,
    run_dp_sweep,
    run_estimator_agreement,
    run_feature_importance,
    run_generator_variance,
    run_learning_curve,
    run_measured_multiseed,
    run_openworld,
    run_prior_sensitivity,
    run_projections,
    run_scalability,
    run_separability_sweep,
    run_subgroup,
    run_wire_sweep,
)
from leakbench.metrics.openworld import OTHER_LABEL, open_world_scores


def _cfg(**kw):
    base = dict(name="t", num_classes=12, traces_per_class=8, bootstrap=10,
                bank=("random_forest", "knn"), test_size=0.4)
    base.update(kw)
    return ExperimentConfig(**base)


def test_projections_cover_every_deployed_attack():
    res = run_projections(_cfg(traces_per_class=10))
    for name in ("keylogger", "whisper_leak", "agentprint", "privacy_sidechannel",
                 "size_only", "timing_only", "network_full", "platform_full"):
        assert name in res["rows"]
        assert 0.0 <= res["rows"][name]["advantage"] <= 1.0
    # A projection is a subset of the full network observable, so it cannot beat it by
    # more than estimator noise.
    assert res["rows"]["keylogger"]["advantage"] <= res["rows"]["network_full"]["advantage"] + 0.2


def test_measured_multiseed_reports_spread_and_ordering():
    res = run_measured_multiseed(_cfg(source="measured", traces_per_class=6),
                                 seeds=[1, 2, 3])
    assert len(res["per_seed_advantage"]) == 3
    for v in ("network", "platform", "providers"):
        assert res["summary"][v]["std"] >= 0.0
    step = res["ordering_steps"]["platform<providers"]
    assert step["n_seeds"] == 3
    assert 0 <= step["holds_in_seeds"] <= 3


def test_wire_sweep_covers_the_grid():
    res = run_wire_sweep(_cfg(source="measured", traces_per_class=6),
                         coalesce_probs=(0.2, 0.7), sigmas=(0.05, 0.3))
    assert res["n_configs"] == 4
    lo, hi = res["network_advantage_range"]
    assert lo <= hi
    assert all(0.0 <= r["network_advantage"] <= 1.0 for r in res["rows"])


def test_attacker_robustness_runs_every_family():
    res = run_attacker_robustness(_cfg(traces_per_class=10))
    for name in ("random_forest", "hist_gbdt", "logistic", "knn"):
        assert name in res["by_attacker"]
    assert not np.isnan(res["strong_learner_spread"])


def test_openworld_recall_falls_with_base_rate():
    res = run_openworld(_cfg(traces_per_class=10), base_rates=(0.5, 0.05),
                        total_traces=1200)
    recalls = [r["recall"] for r in res["rows"]]
    assert recalls[0] > recalls[-1]
    assert res["rows"][0]["other_kind"] == "out_of_taxonomy_random_tool_mix"
    assert "f1" in res["within_taxonomy"]


def test_open_world_scores_handle_all_other():
    y = np.array([OTHER_LABEL, OTHER_LABEL])
    scores = open_world_scores(y, y)
    assert scores["precision"] == 0.0 and scores["recall"] == 0.0


def test_dp_sweep_never_significantly_exceeds_the_ceiling():
    """No *significant* violation: the CI must never clear the analytic ceiling.

    The point estimate may exceed it on a small test set (the advantage is granular in
    1/n_test), which is why the runner separates the two.
    """
    res = run_dp_sweep(_cfg(traces_per_class=10), epsilons=(0.5, 4.0),
                       adjacencies=("all_pairs",), windows=16)
    assert res["ceiling_violations"] == 0
    for row in res["rows"]:
        assert row["significant_violation"] is False
        assert row["advantage_ci"][0] <= row["ceiling_distance_1"] + 1e-12


def test_learning_curve_is_monotone_ish_and_reports_convergence():
    res = run_learning_curve(_cfg(traces_per_class=20), fractions=(0.2, 1.0))
    assert len(res["rows"]) == 2
    assert res["rows"][1]["n_train"] > res["rows"][0]["n_train"]
    assert "random_forest" in res["last_step_delta"]


@pytest.mark.parametrize("stage_source", ["generator", "measured"])
def test_build_dataset_sources(stage_source):
    from leakbench.experiments.runners import build_dataset

    ds, tax = build_dataset(_cfg(source=stage_source, traces_per_class=2))
    assert len(ds) == 24
    assert tax.num_classes == 12


# --------------------------------------------------------- paper-completeness
def test_coarse_secret_reports_both_routes():
    res = run_coarse_secret(_cfg(traces_per_class=12))
    assert res["family_entropy_bits"] == pytest.approx(2.0, abs=1e-9)
    for v, row in res["rows"].items():
        assert 0.0 <= row["family_advantage_reestimated"] <= 1.0
        assert row["family_g_leakage_multiplicative"] >= 1.0 - 1e-9


def test_feature_importance_blocks_sum_to_one():
    res = run_feature_importance(_cfg(traces_per_class=12))
    assert sum(res["importance_by_block"].values()) == pytest.approx(1.0, abs=1e-9)
    assert set(res["importance_by_block"]) == {
        "volume", "size_hist", "timing_hist", "size_stats", "timing_stats"
    }


def test_separability_sweep_separates_the_two_knobs():
    """alpha controls separability; the signature up-weight controls family structure."""
    res = run_separability_sweep(_cfg(traces_per_class=10), alphas=(5.0, 0.1),
                                 upweights=(0.3, 2.5))
    assert res["separability_knob"] == "dirichlet_alpha"
    assert len(res["rows"]) == 2
    assert all("dirichlet_alpha" in r for r in res["rows"])
    # The flag must reflect the data, not assert an expectation.
    advs = [r["advantage"] for r in res["rows"]]
    assert res["advantage_monotone"] == (advs[1] >= advs[0] - 1e-9)
    fam = res["family_structure_knob"]
    assert fam["param"] == "signature_upweight"
    assert len(fam["rows"]) == 2


def test_scalability_reports_timings_and_a_slope():
    res = run_scalability(_cfg(traces_per_class=40), trace_counts=(10, 20, 40))
    assert [r["traces_per_class"] for r in res["rows"]] == [10, 20, 40]
    assert all(r["estimate_seconds"] > 0 for r in res["rows"])
    assert np.isfinite(res["loglog_slope"])


def test_generator_variance_spread_is_reported():
    res = run_generator_variance(_cfg(traces_per_class=8), n_draws=3)
    assert len(res["per_draw"]) == 3
    for v, s in res["summary"].items():
        assert s["min"] <= s["mean"] <= s["max"]
        assert s["std"] >= 0


def test_prior_sensitivity_covers_both_priors():
    res = run_prior_sensitivity(_cfg(traces_per_class=10))
    assert set(res["rows"]) == {"uniform", "zipf"}
    # A skewed prior lowers H(C) and raises the guessing baseline.
    assert res["rows"]["zipf"]["entropy_bits"] < res["rows"]["uniform"]["entropy_bits"]
    assert res["rows"]["zipf"]["guessing_risk"] < res["rows"]["uniform"]["guessing_risk"]


def test_estimator_agreement_reports_rankings():
    res = run_estimator_agreement(_cfg(traces_per_class=10,
                                       defenses=("none", "pad-max")))
    assert set(res["by_vantage"]) == {"network", "platform", "providers"}
    assert set(res["by_defense"]) == {"none", "pad-max"}
    assert "vantage_ranking_agrees" in res["agreement"]


def test_subgroup_reports_within_family_discrimination():
    res = run_subgroup(_cfg(traces_per_class=20))
    within = res["within_family_discrimination"]
    assert set(within) == {"research", "coding", "personal", "ops"}
    for fam, row in within.items():
        if "skipped" not in row:
            assert row["num_classes"] == 3      # the paper's 3-way within-family task
            assert 0.0 <= row["advantage"] <= 1.0


def test_bank_comparison_never_regresses():
    """A maximum over a superset cannot be smaller — that is the whole point."""
    from leakbench.experiments.runners import run_bank_comparison

    res = run_bank_comparison(_cfg(traces_per_class=12))
    assert res["regressions"] == {}
    for v, e in res["rows"].items():
        assert e["full"]["advantage"] >= e["core"]["advantage"] - 1e-9
        assert set(e["core"]["per_estimator"]) == {"random_forest", "knn"}
        assert len(e["full"]["per_estimator"]) == 4


def test_sequence_attacker_reports_its_gain_honestly():
    """The runner must report whichever way the comparison goes, not assume a gain."""
    from leakbench.experiments.runners import run_sequence_attacker

    res = run_sequence_attacker(_cfg(traces_per_class=12), max_len=16)
    # Truncation is reported, not assumed away: at max_len=16 some traces do lose
    # tail calls, and the runner has to say so rather than silently compare unlike
    # representations.
    assert 0.0 <= res["truncation_rate"] <= 1.0
    for v, e in res["rows"].items():
        assert e["advantage_gain"] == pytest.approx(
            e["sequence_advantage"] - e["summary_advantage"], abs=1e-12
        )
        assert e["n_features_sequence"] > e["n_features_summary"]
    assert set(res["tightens_bound_at"]) == {v for v, g in res["advantage_gain"].items() if g > 0}


def test_sequence_representation_respects_the_vantage():
    """A network-vantage sequence must not encode tool categories."""
    from leakbench.data.schemas import ToolCall, Trace
    from leakbench.observables.sequences import sequence_map, sequence_width

    a = Trace(calls=[ToolCall("search", 100.0, 200.0, 0.0)], label=0)
    b = Trace(calls=[ToolCall("calc", 100.0, 200.0, 0.0)], label=1)
    # Same sizes/timing, different category: identical at the network vantage...
    assert np.allclose(sequence_map(a, "network", 4), sequence_map(b, "network", 4))
    # ...and different at platform and providers, which do see categories.
    assert not np.allclose(sequence_map(a, "platform", 4), sequence_map(b, "platform", 4))
    assert not np.allclose(sequence_map(a, "providers", 4), sequence_map(b, "providers", 4))
    for v in ("network", "platform", "providers"):
        assert sequence_map(a, v, 4).size == 4 * sequence_width(v)


def test_sequence_padding_marks_real_calls():
    from leakbench.data.schemas import ToolCall, Trace
    from leakbench.observables.sequences import sequence_map, sequence_width

    t = Trace(calls=[ToolCall("search", 10.0, 20.0, 0.0)] * 3, label=0)
    w = sequence_width("network")
    m = sequence_map(t, "network", 8).reshape(8, w)
    assert list(m[:, -1]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
