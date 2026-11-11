"""RQ runners. Each returns a result dict and (optionally) writes it to disk.

RQ1 vantage ordering, RQ2 feature ablation, RQ3 disparate vulnerability (subgroups),
RQ4 defense cost, RQ5 estimator validation, RQ6 measured harness (single- and
multi-seed, wire-model sweep, attacker robustness), the open-world protocol, the DP
shaping epsilon sweep, and an estimator learning curve. RQ7 (taxonomy-size sweep) and
RQ8 (live capture) have dedicated scripts.

CRITICAL: every result produced here is a REAL computation on synthetic/local data.
None is a measurement of a deployed product. The synthetic demo labels all outputs
"synthetic smoke-test output, not a paper result."
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from leakbench.data.generator import CalibratedGenerator, OtherTrafficGenerator
from leakbench.data.measured import MeasuredHarness
from leakbench.data.schemas import Trace, TraceDataset
from leakbench.data.wire import HTTP2WireModel, apply_wire_model_dataset
from leakbench.defenses import get_defense
from leakbench.defenses.dp_shaping import DPShapingDefense, dp_advantage_ceiling
from leakbench.defenses.mixing import MixingDefense
from leakbench.estimators.attackers import (
    build_attacker,
    robustness_bank,
    sequence_bank,
)
from leakbench.estimators.leakage import estimate_leakage
from leakbench.metrics.information import entropy_bits, family_gain_matrix, g_leakage
from leakbench.metrics.openworld import OTHER_LABEL, open_world_scores
from leakbench.observables.maps import (
    VantageOrder,
    featurize_dataset,
    network_feature_slices,
    network_map,
    platform_feature_slices,
)
from leakbench.observables.projections import PROJECTIONS, project_features
from leakbench.observables.sequences import (
    DEFAULT_MAX_LEN,
    featurize_sequences,
    truncation_rate,
)
from leakbench.taxonomy import Taxonomy, reference_taxonomy, scaled_taxonomy, zipf_prior
from leakbench.utils.logging import get_logger
from leakbench.validation.ground_truth import SymmetricChannel

logger = get_logger(__name__)


def _build_generator_dataset(cfg) -> tuple[TraceDataset, Taxonomy]:
    prior = None
    if getattr(cfg, "prior", "uniform") == "zipf":
        prior = zipf_prior(cfg.num_classes)
    tax = (reference_taxonomy(prior) if cfg.num_classes == 12
           else scaled_taxonomy(cfg.num_classes, prior))
    gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed,
                              signature_upweight=cfg.signature_upweight)
    ds = gen.sample_dataset(cfg.traces_per_class)
    return ds, tax


def _build_real_dataset(cfg) -> tuple[TraceDataset, Taxonomy]:
    """Ingest user-provided traces from ``data.root`` (never downloaded)."""
    from leakbench.data.loader import load_real_dataset

    root = getattr(cfg, "data_root", None)
    if not root:
        raise ValueError(
            "source: real requires `data.root` in the config, pointing at a directory "
            "laid out per docs/data_format.md. LeakBench never downloads data."
        )
    bundle = load_real_dataset(root)
    if bundle.has_explicit_test:
        logger.warning(
            "%s ships a canonical train/test split (%d/%d traces). LeakBench's stages "
            "do their own stratified split, so the two are merged here; use "
            "leakbench.data.loader.load_real_dataset() directly if you need to honour "
            "the provider's split for a comparable number.",
            root, len(bundle.train), len(bundle.test or []),
        )
    ds = bundle.combined()
    logger.info("Loaded %d real traces from %s (taxonomy %s)",
                len(ds), root, bundle.taxonomy.name)
    return ds, bundle.taxonomy


def build_dataset(cfg) -> tuple[TraceDataset, Taxonomy]:
    """Build the trace dataset a stage should run on, per ``cfg.source``.

    ``generator`` (calibrated generator), ``measured`` (real local tools), or ``real``
    (user-provided traces under ``data.root``).
    """
    source = getattr(cfg, "source", "generator")
    if source == "generator":
        return _build_generator_dataset(cfg)
    if source == "measured":
        tax = (reference_taxonomy() if cfg.num_classes == 12
               else scaled_taxonomy(cfg.num_classes))
        harness = MeasuredHarness(taxonomy=tax, seed=cfg.seed, overlap=cfg.overlap)
        return harness.sample_dataset(cfg.traces_per_class), tax
    if source == "real":
        return _build_real_dataset(cfg)
    raise ValueError(
        f"Unknown source {source!r}; expected 'generator', 'measured', or 'real'."
    )


# ------------------------------------------------------------------ RQ1 vantage
def run_vantage(cfg) -> dict[str, Any]:
    """RQ1: leakage per vantage on the generator, with the falsifiability check."""
    ds, tax = build_dataset(cfg)
    prior = tax.prior
    results: dict[str, Any] = {}
    adv_by_vantage: dict[str, float] = {}
    for v in cfg.vantages:
        X, y = featurize_dataset(ds, v)
        res = estimate_leakage(
            X, y, vantage=v, prior=prior, bank=cfg.bank,
            bootstrap=cfg.bootstrap, test_size=cfg.test_size, seed=cfg.seed,
        )
        results[v] = res.to_dict()
        adv_by_vantage[v] = res.advantage

    # Falsifiability guard (Prop. monotonicity): providers must refine both.
    order = VantageOrder.canonical()
    warnings = []
    if "providers" in adv_by_vantage:
        for u in ("network", "platform"):
            if u in adv_by_vantage and order.dominates("providers", u):
                if adv_by_vantage["providers"] + 1e-9 < adv_by_vantage[u]:
                    warnings.append(
                        f"REFINEMENT VIOLATION: Adv_providers ({adv_by_vantage['providers']:.3f})"
                        f" < Adv_{u} ({adv_by_vantage[u]:.3f}) -- indicates an estimator/feature bug."
                    )
    for w in warnings:
        logger.warning(w)
    return {
        "rq": "RQ1_vantage",
        "entropy_bits": tax.entropy_bits(),
        "vantages": results,
        "advantage_by_vantage": adv_by_vantage,
        "refinement_warnings": warnings,
        "empirical_net_le_plat": (
            adv_by_vantage.get("network", 0) <= adv_by_vantage.get("platform", 1)
        ),
    }


# ------------------------------------------------------------------ RQ2 ablation
def run_ablation(cfg) -> dict[str, Any]:
    """RQ2: which network features carry the leakage (length/timing/size)."""
    ds, tax = build_dataset(cfg)
    X = np.vstack([network_map(t) for t in ds.traces])
    y = ds.labels()
    slices = network_feature_slices()

    def _adv(cols: np.ndarray) -> dict[str, Any]:
        res = estimate_leakage(cols, y, vantage="network", prior=tax.prior,
                               bank=cfg.bank, bootstrap=min(cfg.bootstrap, 50),
                               test_size=cfg.test_size, seed=cfg.seed)
        return {"advantage": res.advantage, "mi_bits": res.mi_lower_bound_bits}

    ablations = {
        "length_only": _adv(X[:, slices["volume"]][:, :1]),
        "timing_only": _adv(np.hstack([X[:, slices["timing_hist"]], X[:, slices["timing_stats"]]])),
        "size_only": _adv(np.hstack([X[:, slices["size_hist"]], X[:, slices["size_stats"]]])),
        "size_plus_timing_all": _adv(X),
    }
    return {"rq": "RQ2_ablation", "ablations": ablations,
            "note": "expected ordering: length < timing < size < all (size dominant)"}


# ------------------------------------------------------------------ RQ3 subgroup
def run_subgroup(cfg) -> dict[str, Any]:
    """RQ3: disparate vulnerability across families and trace-length bands."""
    ds, tax = build_dataset(cfg)
    X, y = featurize_dataset(ds, "platform")

    # Family subgroup labels. A trace may carry its own family (real data) or fall
    # outside the taxonomy (an open-world "other"); both are handled explicitly rather
    # than crashing on an index lookup.
    fam_names = tax.families
    fam_to_idx = {f: i for i, f in enumerate(fam_names)}
    fam_name_map = {i: f for f, i in fam_to_idx.items()}
    OTHER_FAMILY_IDX = -1
    fam_name_map[OTHER_FAMILY_IDX] = "other"

    def _family_index(trace: Trace) -> int:
        if 0 <= trace.label < tax.num_classes:
            return fam_to_idx[tax.family_of(trace.label)]
        if trace.family in fam_to_idx:
            return fam_to_idx[trace.family]
        return OTHER_FAMILY_IDX

    fam_labels = np.array([_family_index(t) for t in ds.traces])

    res_fam = estimate_leakage(
        X, y, vantage="platform", prior=tax.prior, bank=cfg.bank,
        bootstrap=min(cfg.bootstrap, 100), test_size=cfg.test_size, seed=cfg.seed,
        subgroup_labels=fam_labels, subgroup_names=fam_name_map,
    )

    # Length-band subgroup.
    length_band = ds.subgroup_length_band()
    res_len = estimate_leakage(
        X, y, vantage="platform", prior=tax.prior, bank=cfg.bank,
        bootstrap=min(cfg.bootstrap, 100), test_size=cfg.test_size, seed=cfg.seed,
        subgroup_labels=length_band, subgroup_names={0: "short", 1: "long"},
    )
    # Within-family discrimination: how well is the *specific* task told apart GIVEN
    # the family? This is what the paper's subgroup table reports for family rows, and
    # it is a different question from "how identifiable is a family member in the full
    # 12-way task" (which the subgroup profile above answers). Both are reported.
    within_family: dict[str, Any] = {}
    for fam in fam_names:
        members = [i for i, t in enumerate(ds.traces)
                   if 0 <= t.label < tax.num_classes and tax.family_of(t.label) == fam]
        if len(members) < 20:
            within_family[fam] = {"skipped": f"only {len(members)} traces"}
            continue
        Xf, yf = X[members], y[members]
        k = len(np.unique(yf))
        sub_prior = np.full(k, 1.0 / k)
        leak = estimate_leakage(Xf, yf, vantage=f"platform:{fam}", prior=sub_prior,
                                bank=cfg.bank, bootstrap=min(cfg.bootstrap, 100),
                                test_size=cfg.test_size, seed=cfg.seed)
        within_family[fam] = {
            "advantage": leak.advantage,
            "advantage_ci": list(leak.advantage_ci),
            "mi_bits": leak.mi_lower_bound_bits,
            "num_classes": k,
        }

    return {
        "rq": "RQ3_subgroup",
        "overall_advantage": res_fam.advantage,
        "by_family": {k: v.advantage for k, v in res_fam.subgroups.items()},
        "by_family_mi_bits": {k: v.mi_lower_bound_bits for k, v in res_fam.subgroups.items()},
        "by_length_band": {k: v.advantage for k, v in res_len.subgroups.items()},
        "by_length_band_mi_bits": {
            k: v.mi_lower_bound_bits for k, v in res_len.subgroups.items()
        },
        "within_family_discrimination": within_family,
        "note": "aggregate leakage hides subgroups that leak more (disparate "
                "vulnerability). 'by_family' = how the full 12-way task fares within "
                "each family's traces; 'within_family_discrimination' = the k-way task "
                "restricted to one family (the paper's family rows).",
    }


# ------------------------------------------------------------------ RQ4 defenses
def run_defenses(cfg) -> dict[str, Any]:
    """RQ4: privacy-utility frontier of the WF defense toolbox at the network vantage."""
    ds, tax = build_dataset(cfg)
    rows = []
    for dname in cfg.defenses:
        defense = get_defense(dname)
        dres = defense.apply(ds, seed=cfg.seed)
        X, y = featurize_dataset(dres.dataset, "network")
        leak = estimate_leakage(X, y, vantage="network", prior=tax.prior,
                                bank=cfg.bank, bootstrap=min(cfg.bootstrap, 100),
                                test_size=cfg.test_size, seed=cfg.seed)
        rows.append({
            "defense": dname,
            "advantage": leak.advantage,
            "mi_bits": leak.mi_lower_bound_bits,
            "bandwidth_overhead": dres.bandwidth_overhead,
            "latency_overhead": dres.latency_overhead,
        })
    return {"rq": "RQ4_defenses", "vantage": "network", "rows": rows,
            "note": "only size padding neutralizes the channel, at order-of-magnitude cost"}


# ------------------------------------------------------------------ RQ5 validate
def run_validation(cfg, crossovers: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95),
                   K: int = 12, n_per_class: int = 3000) -> dict[str, Any]:
    """RQ5: does the estimator track the analytic ground-truth channel?

    ``n_per_class`` is deliberately large. The confusion-matrix MI is a *plug-in*
    estimate on a ``K x K`` table, so it carries the usual ``(K-1)^2 / (2 N ln 2)``
    upward bias: ~0.08 bits at 300/class, ~0.02 at 3000/class. Validating at 300
    would report the estimator's small-sample behaviour as estimator error. We measured
    the decay directly (``scripts/val_scaling.py``): the max error follows ``N^-0.60``,
    between the ``1/N`` of the bias term and the ``1/sqrt(N)`` of the sampling noise
    that a maximum over crossover points also absorbs.
    """
    rows = []
    prior = np.full(K, 1.0 / K)
    for p in crossovers:
        ch = SymmetricChannel(K=K, p=p, seed=cfg.seed)
        X, y = ch.sample(n_per_class)
        leak = estimate_leakage(X, y, vantage="ground_truth", prior=prior,
                                bank=cfg.bank, bootstrap=50, test_size=cfg.test_size,
                                seed=cfg.seed)
        ana = ch.analytic()
        rows.append({
            "p": p,
            "analytic_mi_bits": ana["mi_bits"],
            "estimated_mi_bits": leak.mi_lower_bound_bits,
            "analytic_bayes_risk": ana["bayes_risk"],
            "estimated_bayes_risk": leak.bayes_risk,
            "mi_abs_error": abs(ana["mi_bits"] - leak.mi_lower_bound_bits),
        })
    max_err = max(r["mi_abs_error"] for r in rows)
    return {"rq": "RQ5_validation", "K": K, "rows": rows, "max_mi_abs_error": max_err,
            "note": "estimated MI is a LOWER bound; it should track (from below) the analytic curve"}


# ------------------------------------------------------------------ RQ6 measured
def run_measured(cfg, wire: HTTP2WireModel | None = None) -> dict[str, Any]:
    """RQ6: do the findings hold on measured traces from real local tools?"""
    tax = reference_taxonomy() if cfg.num_classes == 12 else scaled_taxonomy(cfg.num_classes)
    harness = MeasuredHarness(taxonomy=tax, seed=cfg.seed, overlap=cfg.overlap)
    ds = harness.sample_dataset(cfg.traces_per_class)
    wire = wire or HTTP2WireModel(seed=cfg.seed)

    results: dict[str, Any] = {}
    adv_by_vantage: dict[str, float] = {}
    for v in cfg.vantages:
        # Network vantage sees the wire-modeled sizes; platform/providers see pre-wire.
        source = apply_wire_model_dataset(ds, wire) if v == "network" else ds
        X, y = featurize_dataset(source, v)
        leak = estimate_leakage(X, y, vantage=v, prior=tax.prior, bank=cfg.bank,
                                bootstrap=min(cfg.bootstrap, 100),
                                test_size=cfg.test_size, seed=cfg.seed)
        results[v] = leak.to_dict()
        adv_by_vantage[v] = leak.advantage
    return {"rq": "RQ6_measured", "advantage_by_vantage": adv_by_vantage,
            "vantages": results,
            "note": "real tool I/O + real timing + HTTP/2 wire model; ordering should hold"}


# ------------------------------------------------- RQ1b coarse (family) secret
def run_coarse_secret(cfg) -> dict[str, Any]:
    """Table "coarse": the 4-way **family** secret beside the 12-way class secret.

    A leakage number is meaningless without a stated secret, so the paper re-runs the
    study against the coarse secret. Two routes are reported, because they answer
    slightly different questions and the paper claims both:

    - **re-estimated**: fit the bank directly on family labels (`H(C)=2` bits);
    - **g-leakage view** (Def. "g-leakage view"): score family-level guessing from the
      *class* confusion matrix with a family gain function, i.e. without changing the
      estimator at all.

    Agreement between the two is a soft check on the g-leakage machinery; a gap means
    the family-level task is genuinely easier to learn directly than to read off a
    12-way classifier's mistakes.
    """
    ds, tax = build_dataset(cfg)
    fam_names = tax.families
    fam_to_idx = {f: i for i, f in enumerate(fam_names)}
    y_fam = np.array([fam_to_idx[tax.family_of(t.label)] for t in ds.traces])
    fam_prior = np.array(
        [sum(tax.prior[c.index] for c in tax.classes if c.family == f) for f in fam_names]
    )
    class_family = np.array([fam_to_idx[c.family] for c in tax.classes])

    rows: dict[str, Any] = {}
    for v in cfg.vantages:
        X, y = featurize_dataset(ds, v)
        fine = estimate_leakage(X, y, vantage=v, prior=tax.prior, bank=cfg.bank,
                                bootstrap=min(cfg.bootstrap, 100),
                                test_size=cfg.test_size, seed=cfg.seed)
        coarse = estimate_leakage(X, y_fam, vantage=f"{v}:family", prior=fam_prior,
                                  bank=cfg.bank, bootstrap=min(cfg.bootstrap, 100),
                                  test_size=cfg.test_size, seed=cfg.seed)
        # g-leakage route: family gain over the fine-grained confusion matrix.
        conf = _confusion_for(X, y, tax, cfg)
        gain = family_gain_matrix(class_family)
        g = g_leakage(tax.prior, conf, gain)
        rows[v] = {
            "class_advantage": fine.advantage,
            "class_mi_bits": fine.mi_lower_bound_bits,
            "family_advantage_reestimated": coarse.advantage,
            "family_advantage_ci": list(coarse.advantage_ci),
            "family_mi_bits": coarse.mi_lower_bound_bits,
            "family_g_leakage_multiplicative": g["L_mult"],
            "family_success_prob_g_view": g["V_post"],
        }
    return {
        "rq": "RQ1b_coarse_secret",
        "class_entropy_bits": tax.entropy_bits(),
        "family_entropy_bits": float(entropy_bits(fam_prior)),
        "families": list(fam_names),
        "rows": rows,
        "note": "the same estimator scores both secrets; advantage is only meaningful "
                "against a stated taxonomy and prior",
    }


def _confusion_for(X: np.ndarray, y: np.ndarray, tax: Taxonomy, cfg) -> np.ndarray:
    """Test-set confusion matrix of the strongest estimator (for the g-leakage view)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed,
                              stratify=y)
    best_err, best_pred = np.inf, None
    for name in cfg.bank:
        clf = build_attacker(name, seed=cfg.seed)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        err = float(np.mean(pred != y[te]))
        if err < best_err:
            best_err, best_pred = err, pred
    return confusion_matrix(y[te], best_pred, labels=np.arange(tax.num_classes))


# ------------------------------------------------ RQ2b feature importance
def run_feature_importance(cfg, vantage: str = "network") -> dict[str, Any]:
    """RQ2 corroboration: random-forest Gini importances grouped by feature block.

    The paper reports the ablation and the importances as "two different windows onto
    the same conclusion" --- if size features dominate the importances *and* the
    size-only ablation dominates, the conclusion is not an artifact of either method.
    """
    ds, tax = build_dataset(cfg)
    if getattr(cfg, "source", "generator") == "measured" and vantage == "network":
        ds = apply_wire_model_dataset(ds, HTTP2WireModel(seed=cfg.seed))
    X, y = featurize_dataset(ds, vantage)
    slices = (network_feature_slices() if vantage == "network"
              else platform_feature_slices())

    idx = np.arange(len(y))
    tr, _ = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed,
                             stratify=y)
    clf = build_attacker("random_forest", seed=cfg.seed)
    clf.fit(X[tr], y[tr])
    importances = np.asarray(clf.feature_importances_, dtype=float)

    by_block = {name: float(importances[sl].sum()) for name, sl in slices.items()}
    total = sum(by_block.values()) or 1.0
    by_block = {k: v / total for k, v in by_block.items()}
    size_share = sum(v for k, v in by_block.items() if k.startswith("size") or k == "volume")
    timing_share = sum(v for k, v in by_block.items() if k.startswith("timing"))
    return {
        "rq": "RQ2b_feature_importance",
        "vantage": vantage,
        "importance_by_block": by_block,
        "size_related_share": float(size_share),
        "timing_related_share": float(timing_share),
        "note": "should corroborate the ablation from the other direction "
                "(size dominant, raw call count marginal)",
    }


# ------------------------------------------- Fig separability (sweep)
def run_separability_sweep(
    cfg,
    alphas: Sequence[float] = (5.0, 1.5, 0.6, 0.3, 0.1),
    upweights: Sequence[float] = (0.3, 0.6, 1.0, 1.5, 2.5),
    vantage: str = "platform",
) -> dict[str, Any]:
    """Sweep ground-truth separability and check the instrument tracks it.

    Two knobs are swept, because they do different things and conflating them is what
    made the original sweep read backwards:

    - **``dirichlet_alpha`` — the separability knob.** It sets how distinct the
      per-class tool distributions are. Large alpha drives every class toward uniform
      (confusable); small alpha gives sparse, sharply different profiles (separable).
      Swept from confusable to separable, so a correct instrument must rise
      monotonically. This is the curve to put in the paper.
    - **``signature_upweight`` — a family-structure knob, not a separability knob.**
      It adds mass to the *family's* signature tools, and three classes share each
      family, so raising it makes same-family classes **more alike** and advantage
      *falls*. Reported alongside because the contrast is informative, and because
      calling it "separability" would misdescribe what it controls.
    """
    prior_arg = zipf_prior(cfg.num_classes) if getattr(cfg, "prior", "uniform") == "zipf" else None
    tax = (reference_taxonomy(prior_arg) if cfg.num_classes == 12
           else scaled_taxonomy(cfg.num_classes, prior_arg))

    def _sweep(param: str, values: Sequence[float]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for v in values:
            kwargs = {param: float(v)}
            gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed, **kwargs)
            ds = gen.sample_dataset(cfg.traces_per_class)
            X, y = featurize_dataset(ds, vantage)
            leak = estimate_leakage(X, y, vantage=vantage, prior=tax.prior, bank=cfg.bank,
                                    bootstrap=min(cfg.bootstrap, 50),
                                    test_size=cfg.test_size, seed=cfg.seed)
            out.append({param: float(v), "advantage": leak.advantage,
                        "mi_bits": leak.mi_lower_bound_bits})
            logger.info("separability %s=%.2f adv=%.3f MI=%.3f", param, v,
                        leak.advantage, leak.mi_lower_bound_bits)
        return out

    def _monotone(rows: list[dict[str, Any]], key: str) -> bool:
        vals = [float(r[key]) for r in rows]
        return bool(all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])))

    alpha_rows = _sweep("dirichlet_alpha", alphas)
    upweight_rows = _sweep("signature_upweight", upweights)
    return {
        "rq": "separability_sweep",
        "vantage": vantage,
        "separability_knob": "dirichlet_alpha",
        "rows": alpha_rows,
        "advantage_monotone": _monotone(alpha_rows, "advantage"),
        "mi_monotone": _monotone(alpha_rows, "mi_bits"),
        "family_structure_knob": {
            "param": "signature_upweight",
            "rows": upweight_rows,
            "advantage_monotone_increasing": _monotone(upweight_rows, "advantage"),
            "note": "expected to DECREASE: the signature is shared by a family, so "
                    "up-weighting it makes same-family classes more alike",
        },
        "note": "alpha is swept from confusable (large) to separable (small), so a "
                "correct instrument rises monotonically on the alpha curve",
    }


# ------------------------------------------------- Fig scalability (cost)
def run_scalability(
    cfg,
    trace_counts: Sequence[int] | None = None,
    vantage: str = "platform",
) -> dict[str, Any]:
    """Estimation cost vs. dataset size (paper Fig. "scalability", §Impl "Cost").

    The paper's claim is that estimation is *roughly linear* in trace count, so
    re-running the benchmark is routine rather than heroic. We time the featurize +
    estimate path (excluding trace generation) and fit a log-log slope: ~1 means
    linear, ~2 quadratic.
    """
    import time as _time

    counts = list(trace_counts) if trace_counts else [50, 100, 200, 400, cfg.traces_per_class]
    counts = sorted({c for c in counts if c >= 10})
    prior_arg = zipf_prior(cfg.num_classes) if getattr(cfg, "prior", "uniform") == "zipf" else None
    tax = (reference_taxonomy(prior_arg) if cfg.num_classes == 12
           else scaled_taxonomy(cfg.num_classes, prior_arg))

    rows = []
    for n in counts:
        gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed,
                                  signature_upweight=cfg.signature_upweight)
        ds = gen.sample_dataset(n)
        t0 = _time.perf_counter()
        X, y = featurize_dataset(ds, vantage)
        leak = estimate_leakage(X, y, vantage=vantage, prior=tax.prior, bank=cfg.bank,
                                bootstrap=min(cfg.bootstrap, 50),
                                test_size=cfg.test_size, seed=cfg.seed)
        elapsed = _time.perf_counter() - t0
        rows.append({"traces_per_class": n, "n_traces": len(ds),
                     "estimate_seconds": float(elapsed), "advantage": leak.advantage})
        logger.info("scalability n=%d (%d traces) %.2fs", n, len(ds), elapsed)

    slope = float("nan")
    if len(rows) >= 2:
        xs = np.log(np.array([r["n_traces"] for r in rows], dtype=float))
        ys = np.log(np.array([max(1e-6, r["estimate_seconds"]) for r in rows]))
        slope = float(np.polyfit(xs, ys, 1)[0])
    return {
        "rq": "scalability",
        "vantage": vantage,
        "rows": rows,
        "loglog_slope": slope,
        "note": "slope ~1 means estimation cost is roughly linear in trace count; "
                "timings are host-specific and are reported as such",
    }


# ------------------------------------- sensitivity: generator-draw variance
def run_generator_variance(cfg, n_draws: int = 8) -> dict[str, Any]:
    """Re-draw the generator's class profiles and re-estimate (paper §Eval sensitivity).

    Bootstrap CIs capture test-resampling variance but *not* the variance of the
    generator's one-time profile draw. Re-drawing the Dirichlet profiles answers the
    question a CI cannot: is the headline number a lucky draw?
    """
    prior_arg = zipf_prior(cfg.num_classes) if getattr(cfg, "prior", "uniform") == "zipf" else None
    tax = (reference_taxonomy(prior_arg) if cfg.num_classes == 12
           else scaled_taxonomy(cfg.num_classes, prior_arg))
    per_draw: list[dict[str, float]] = []
    for i in range(n_draws):
        gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed + 1000 * (i + 1),
                                  signature_upweight=cfg.signature_upweight)
        ds = gen.sample_dataset(cfg.traces_per_class)
        entry: dict[str, float] = {}
        for v in cfg.vantages:
            X, y = featurize_dataset(ds, v)
            entry[v] = estimate_leakage(
                X, y, vantage=v, prior=tax.prior, bank=cfg.bank,
                bootstrap=min(cfg.bootstrap, 30), test_size=cfg.test_size, seed=cfg.seed,
            ).advantage
        per_draw.append(entry)
        logger.info("generator draw %d: %s", i + 1, {k: round(v, 3) for k, v in entry.items()})
    summary = {
        v: {"mean": float(np.mean([d[v] for d in per_draw])),
            "std": float(np.std([d[v] for d in per_draw])),
            "min": float(np.min([d[v] for d in per_draw])),
            "max": float(np.max([d[v] for d in per_draw]))}
        for v in per_draw[0]
    }
    return {"rq": "generator_draw_variance", "n_draws": n_draws,
            "per_draw": per_draw, "summary": summary,
            "note": "spread across independent Dirichlet profile draws, which the "
                    "bootstrap CI does not cover"}


# ------------------------------------------- sensitivity: non-uniform prior
def run_prior_sensitivity(cfg) -> dict[str, Any]:
    """Uniform vs. skewed (Zipf) task prior (paper §Eval "Non-uniform prior").

    Uniform maximizes ``H(C)`` but is **not** worst case for advantage: a skewed prior
    makes blind guessing easier, raising the baseline the attacker must beat, so
    advantage falls even though no less information leaks. Reporting ``R_pi`` and the
    MI alongside the advantage is what keeps that readable.
    """
    rows: dict[str, Any] = {}
    for prior_name in ("uniform", "zipf"):
        prior = zipf_prior(cfg.num_classes) if prior_name == "zipf" else None
        tax = (reference_taxonomy(prior) if cfg.num_classes == 12
               else scaled_taxonomy(cfg.num_classes, prior))
        gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed,
                                  signature_upweight=cfg.signature_upweight)
        ds = gen.sample_dataset(cfg.traces_per_class)
        entry: dict[str, Any] = {"entropy_bits": tax.entropy_bits(),
                                 "guessing_risk": tax.guessing_risk(), "vantages": {}}
        for v in cfg.vantages:
            X, y = featurize_dataset(ds, v)
            leak = estimate_leakage(X, y, vantage=v, prior=tax.prior, bank=cfg.bank,
                                    bootstrap=min(cfg.bootstrap, 50),
                                    test_size=cfg.test_size, seed=cfg.seed)
            entry["vantages"][v] = {"advantage": leak.advantage,
                                    "mi_bits": leak.mi_lower_bound_bits,
                                    "accuracy": leak.accuracy}
        rows[prior_name] = entry
    return {"rq": "prior_sensitivity", "rows": rows,
            "note": "a lower advantage under a skewed prior means easier guessing, "
                    "not less leakage -- compare the MI and accuracy columns"}


# --------------------------------------------------- estimator agreement
def run_estimator_agreement(cfg) -> dict[str, Any]:
    """Do the estimators agree on the *ordering* of vantages and defenses?

    Paper App. "Estimator agreement": the headline trends must not hinge on one
    learner. We report each estimator's per-vantage and per-defense advantage and
    check that every estimator induces the same ranking as the reported (strongest)
    one --- disagreement is a diagnostic, not something to average away.
    """
    ds, tax = build_dataset(cfg)
    bank = tuple(cfg.bank)

    by_vantage: dict[str, dict[str, float]] = {}
    for v in cfg.vantages:
        X, y = featurize_dataset(ds, v)
        by_vantage[v] = {
            name: estimate_leakage(X, y, vantage=v, prior=tax.prior, bank=(name,),
                                   bootstrap=0, test_size=cfg.test_size,
                                   seed=cfg.seed).advantage
            for name in bank
        }

    by_defense: dict[str, dict[str, float]] = {}
    for dname in cfg.defenses:
        defense = get_defense(dname)
        dres = defense.apply(ds, seed=cfg.seed)
        X, y = featurize_dataset(dres.dataset, "network")
        by_defense[dname] = {
            name: estimate_leakage(X, y, vantage="network", prior=tax.prior, bank=(name,),
                                   bootstrap=0, test_size=cfg.test_size,
                                   seed=cfg.seed).advantage
            for name in bank
        }

    def _ranking(values: dict[str, dict[str, float]], estimator: str) -> list[str]:
        return [k for k, _ in sorted(values.items(), key=lambda kv: kv[1][estimator])]

    ref = bank[0]
    agreement = {
        "vantage_ranking_agrees": {
            name: _ranking(by_vantage, name) == _ranking(by_vantage, ref) for name in bank
        },
        "defense_ranking_agrees": {
            name: _ranking(by_defense, name) == _ranking(by_defense, ref) for name in bank
        },
    }
    return {
        "rq": "estimator_agreement",
        "by_vantage": by_vantage,
        "by_defense": by_defense,
        "reference_estimator": ref,
        "vantage_ranking": _ranking(by_vantage, ref),
        "defense_ranking": _ranking(by_defense, ref),
        "agreement": agreement,
        "note": "orderings must agree across estimators even where magnitudes differ",
    }


# ------------------------------- multi-user gateway mixing (planned exp. iv)
def run_mixing_sweep(
    cfg,
    sizes: Sequence[int] = (1, 2, 4, 8, 16),
    vantage: str = "network",
) -> dict[str, Any]:
    """How much does a shared gateway's anonymity set actually buy? (planned exp. iv)

    Sweeps the number of concurrent users behind one gateway and measures the residual
    advantage against a *target* user's task. Reported next to the manufactured-traffic
    defenses, because the comparison is the point: mixing reuses bytes other users were
    already sending, so its per-user marginal bandwidth is ~0, whereas padding and cover
    charge the protected user for every byte.

    The ``bandwidth_overhead`` column therefore means something different here --- it is
    the growth of the *observed* stream, not new traffic --- and the result dict says so
    rather than letting the number be read as a cost.
    """
    ds, tax = build_dataset(cfg)
    rows: list[dict[str, Any]] = []
    for n in sizes:
        defense = MixingDefense(n_users=int(n))
        dres = defense.apply(ds, seed=cfg.seed)
        source = (apply_wire_model_dataset(dres.dataset, HTTP2WireModel(seed=cfg.seed))
                  if vantage == "network" and getattr(cfg, "source", "") == "measured"
                  else dres.dataset)
        X, y = featurize_dataset(source, vantage)
        leak = estimate_leakage(X, y, vantage=vantage, prior=tax.prior, bank=cfg.bank,
                                bootstrap=min(cfg.bootstrap, 50),
                                test_size=cfg.test_size, seed=cfg.seed)
        rows.append({
            "n_users": int(n),
            "advantage": leak.advantage,
            "advantage_ci": list(leak.advantage_ci),
            "mi_bits": leak.mi_lower_bound_bits,
            "observed_stream_growth": dres.bandwidth_overhead,
            "marginal_bandwidth_per_user": 0.0,
            "mean_observed_calls": float(np.mean([t.n for t in dres.dataset.traces])),
        })
        logger.info("mixing n_users=%d: advantage %.3f (observed calls %.1f)",
                    n, leak.advantage, rows[-1]["mean_observed_calls"])
    base = float(rows[0]["advantage"]) if rows else float("nan")
    return {
        "rq": "multi_user_mixing",
        "vantage": vantage,
        "rows": rows,
        "advantage_at_one_user": base,
        "advantage_reduction": {int(r["n_users"]): base - float(r["advantage"])
                                for r in rows},
        "note": "observed_stream_growth is how much MORE the observer sees, not new "
                "bytes: the co-mixed traffic would have been sent anyway, so unlike "
                "padding/cover the per-user marginal bandwidth is ~0",
    }


# ------------------------------------ raw-sequence attacker (planned exp. iii)
def run_sequence_attacker(
    cfg,
    max_len: int = DEFAULT_MAX_LEN,
) -> dict[str, Any]:
    """Summary features vs. the **raw ordered trace**, at the same vantage.

    The paper lists "a learned sequence-model attacker over raw traces as a tighter
    estimator" among its planned experiments (App. "Additional Results"). This runs it.

    The comparison is deliberately like-for-like: both representations are restricted
    to exactly what the vantage observes (see
    :mod:`leakbench.observables.sequences`), so any gain comes from *not aggregating*
    — the histograms of ``O_v`` discard the order in which calls occurred, and a real
    observer sees that order. Since advantage and MI are lower bounds, a gain here
    means the paper's reported values are loose by that much; a loss would mean the
    sequence learner is simply weaker, which is reported rather than hidden.
    """
    ds, tax = build_dataset(cfg)
    wired = (apply_wire_model_dataset(ds, HTTP2WireModel(seed=cfg.seed))
             if getattr(cfg, "source", "generator") == "measured" else ds)
    trunc = truncation_rate(ds, max_len)
    if trunc > 0.05:
        logger.warning(
            "%.1f%% of traces exceed max_len=%d and lose tail calls; raise max_len or "
            "read the comparison as no longer like-for-like.", 100 * trunc, max_len)

    rows: dict[str, Any] = {}
    for v in cfg.vantages:
        source = wired if v == "network" else ds
        X_sum, y = featurize_dataset(source, v)
        summary = estimate_leakage(X_sum, y, vantage=v, prior=tax.prior, bank=cfg.bank,
                                   bootstrap=min(cfg.bootstrap, 50),
                                   test_size=cfg.test_size, seed=cfg.seed)
        X_seq, y_seq = featurize_sequences(source, v, max_len)
        seq = estimate_leakage(X_seq, y_seq, vantage=f"{v}:sequence", prior=tax.prior,
                               bank=sequence_bank(), bootstrap=min(cfg.bootstrap, 50),
                               test_size=cfg.test_size, seed=cfg.seed)
        rows[v] = {
            "summary_advantage": summary.advantage,
            "summary_mi_bits": summary.mi_lower_bound_bits,
            "sequence_advantage": seq.advantage,
            "sequence_advantage_ci": list(seq.advantage_ci),
            "sequence_mi_bits": seq.mi_lower_bound_bits,
            "sequence_best_estimator": seq.best_estimator,
            "advantage_gain": seq.advantage - summary.advantage,
            "mi_gain_bits": seq.mi_lower_bound_bits - summary.mi_lower_bound_bits,
            "n_features_summary": int(X_sum.shape[1]),
            "n_features_sequence": int(X_seq.shape[1]),
        }
        logger.info("sequence %s: summary %.3f -> raw sequence %.3f (%s)", v,
                    summary.advantage, seq.advantage, seq.best_estimator)

    gains = {v: e["advantage_gain"] for v, e in rows.items()}
    return {
        "rq": "sequence_attacker",
        "source": getattr(cfg, "source", "generator"),
        "max_len": max_len,
        "truncation_rate": trunc,
        "rows": rows,
        "advantage_gain": gains,
        "tightens_bound_at": [v for v, g in gains.items() if g > 0],
        "note": "same vantage, same information, un-aggregated representation. A "
                "positive gain means the histogram features under-report leakage, "
                "since advantage is only ever a lower bound.",
    }


# ------------------------------------------------- estimator bank comparison
def run_bank_comparison(cfg) -> dict[str, Any]:
    """Core bank {RF, kNN} vs. the full bank {RF, HGB, LR, kNN}, per vantage.

    The paper is internally inconsistent about which bank produces the reported
    number: the estimation algorithm lists ``E = {RF, kNN}`` and §Impl describes only
    those two, but Fig. "architecture" draws the bank as *RF / GBM / LR* plus k-NN, and
    App. "Estimator details" states "we report the maximum over the bank".

    The distinction is not cosmetic and it is not tuning: advantage and MI are
    **lower** bounds, so adding an estimator can only raise the reported value, never
    lower it. Reporting the maximum over a larger bank is therefore a *tighter* bound
    on the same quantity — which is exactly the paper's stated policy. This runner
    reports both so the choice is explicit rather than accidental.
    """
    core = ("random_forest", "knn")
    full = robustness_bank()
    ds, tax = build_dataset(cfg)
    wired = (apply_wire_model_dataset(ds, HTTP2WireModel(seed=cfg.seed))
             if getattr(cfg, "source", "generator") == "measured" else ds)

    rows: dict[str, Any] = {}
    for v in cfg.vantages:
        source = wired if v == "network" else ds
        X, y = featurize_dataset(source, v)
        entry: dict[str, Any] = {}
        for label, bank in (("core", core), ("full", full)):
            leak = estimate_leakage(X, y, vantage=v, prior=tax.prior, bank=bank,
                                    bootstrap=min(cfg.bootstrap, 50),
                                    test_size=cfg.test_size, seed=cfg.seed)
            entry[label] = {"advantage": leak.advantage,
                            "mi_bits": leak.mi_lower_bound_bits,
                            "best_estimator": leak.best_estimator,
                            "per_estimator": leak.per_estimator}
        entry["advantage_gain"] = entry["full"]["advantage"] - entry["core"]["advantage"]
        entry["mi_gain_bits"] = entry["full"]["mi_bits"] - entry["core"]["mi_bits"]
        rows[v] = entry
        logger.info("bank %s: core %.3f -> full %.3f (best %s)", v,
                    entry["core"]["advantage"], entry["full"]["advantage"],
                    entry["full"]["best_estimator"])

    regressions = {v: e["advantage_gain"] for v, e in rows.items() if e["advantage_gain"] < -1e-9}
    for v, g in regressions.items():
        logger.warning(
            "BANK REGRESSION at %s: the larger bank reported %.4f LESS advantage. A "
            "maximum over a superset cannot decrease -- this indicates a bug.", v, g)
    return {
        "rq": "estimator_bank_comparison",
        "source": getattr(cfg, "source", "generator"),
        "core_bank": list(core),
        "full_bank": list(full),
        "rows": rows,
        "regressions": regressions,
        "note": "advantage and MI are lower bounds, so the maximum over the larger bank "
                "is a tighter bound on the same quantity, not a different measurement",
    }


# ------------------------------------------- case study: deployed attacks
def run_projections(cfg) -> dict[str, Any]:
    """Reproduce deployed attacks as projections of the unified observable.

    Paper §Eval "Case study: reproducing a deployed attack": each 2024-2026 attack
    reads a sub-vector of some ``O_v``; measuring those sub-vectors on *common* traces
    is what makes the published numbers comparable. Reported alongside the full
    network observable and the platform vantage, which is the comparison the deployed
    papers cannot make against one another.
    """
    ds, tax = build_dataset(cfg)
    if getattr(cfg, "source", "generator") == "measured":
        net_source = apply_wire_model_dataset(ds, HTTP2WireModel(seed=cfg.seed))
    else:
        net_source = ds
    X_net, y = featurize_dataset(net_source, "network")
    X_plat, _ = featurize_dataset(ds, "platform")

    def _adv(X: np.ndarray, tag: str) -> dict[str, Any]:
        leak = estimate_leakage(X, y, vantage=tag, prior=tax.prior, bank=cfg.bank,
                                bootstrap=min(cfg.bootstrap, 50),
                                test_size=cfg.test_size, seed=cfg.seed)
        return {"advantage": leak.advantage, "mi_bits": leak.mi_lower_bound_bits}

    rows: dict[str, Any] = {
        "network_full": {"vantage": "network", "reference": "zhang2025agentprint (~O_net)",
                         **_adv(X_net, "network")},
        "platform_full": {"vantage": "platform", "reference": "—",
                          **_adv(X_plat, "platform")},
    }
    for name, proj in PROJECTIONS.items():
        base = X_net if proj.base_vantage == "network" else X_plat
        rows[name] = {
            "vantage": proj.base_vantage,
            "reference": proj.reference,
            "blocks": list(proj.blocks),
            **_adv(project_features(base, name), f"{proj.base_vantage}:{name}"),
        }
    return {
        "rq": "projections_case_study",
        "source": getattr(cfg, "source", "generator"),
        "rows": rows,
        "note": "deployed attacks measured as feature-subset projections on common traces; "
                "these are the paper's comparators, not re-runs of the original code",
    }


# ------------------------------------------------- RQ6 measured, multi-seed
def run_measured_multiseed(cfg, seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """RQ6 at paper scale: repeat the measured harness over several seeds.

    Reports mean +/- std per vantage, how many seeds satisfy each step of the
    ``net < plat < prov`` ordering, and a paired t-test on each step (the paper's
    "the ordering is significant, not a coincidence" paragraph). The t-test is over
    seeds, i.e. it asks whether the per-seed gap is reliably positive --- with a
    handful of seeds this is a small-sample test and is reported as such.
    """
    seeds = list(seeds) if seeds is not None else [cfg.seed + i for i in range(5)]
    per_seed: list[dict[str, float]] = []
    per_seed_mi: list[dict[str, float]] = []
    ci_at_reference: dict[str, list[float]] = {}
    for i, seed in enumerate(seeds):
        seed_cfg = replace(cfg, seed=seed)
        res = run_measured(seed_cfg)
        per_seed.append(dict(res["advantage_by_vantage"]))
        per_seed_mi.append({v: res["vantages"][v]["mi_lower_bound_bits"]
                            for v in res["vantages"]})
        if i == 0:
            ci_at_reference = {v: list(res["vantages"][v]["advantage_ci"])
                               for v in res["vantages"]}
        logger.info("measured seed %d: %s", seed,
                    {k: round(v, 3) for k, v in per_seed[-1].items()})

    vantages = list(per_seed[0])
    summary = {
        v: {
            "mean": float(np.mean([p[v] for p in per_seed])),
            "std": float(np.std([p[v] for p in per_seed])),
            "min": float(np.min([p[v] for p in per_seed])),
            "max": float(np.max([p[v] for p in per_seed])),
            "mi_bits_at_reference_seed": float(per_seed_mi[0][v]),
            "advantage_ci_at_reference_seed": ci_at_reference.get(v, [float("nan")] * 2),
        }
        for v in vantages
    }

    steps: dict[str, Any] = {}
    for lo, hi in (("network", "platform"), ("platform", "providers")):
        if lo not in vantages or hi not in vantages:
            continue
        gaps = np.array([p[hi] - p[lo] for p in per_seed], dtype=float)
        steps[f"{lo}<{hi}"] = {
            "holds_in_seeds": int(np.sum(gaps > 0)),
            "n_seeds": len(seeds),
            "mean_gap": float(gaps.mean()),
            "min_gap": float(gaps.min()),
            **_paired_t_test(gaps),
        }
    return {
        "rq": "RQ6_measured_multiseed",
        "seeds": seeds,
        "traces_per_class": cfg.traces_per_class,
        "per_seed_advantage": per_seed,
        "summary": summary,
        "ordering_steps": steps,
        "note": "advantage is mean+/-std over seeds; MI and CI are at the first seed",
    }


def _paired_t_test(gaps: np.ndarray) -> dict[str, float]:
    """One-sample t-test that the per-seed gap is > 0 (paired across seeds)."""
    n = len(gaps)
    if n < 2 or float(np.std(gaps, ddof=1)) == 0.0:
        return {"t_statistic": float("nan"), "p_value_one_sided": float("nan")}
    t_stat = float(np.mean(gaps) / (np.std(gaps, ddof=1) / np.sqrt(n)))
    try:
        from scipy import stats

        p = float(stats.t.sf(t_stat, df=n - 1))
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        p = float("nan")
    return {"t_statistic": t_stat, "p_value_one_sided": p}


# ------------------------------------------------------- RQ6 wire-model sweep
def run_wire_sweep(
    cfg,
    coalesce_probs: Sequence[float] = (0.2, 0.45, 0.7),
    sigmas: Sequence[float] = (0.05, 0.15, 0.30),
) -> dict[str, Any]:
    """RQ6 robustness: does the network result hinge on the HTTP/2 wire model?

    Sweeps the coalescing probability ``p`` and the cross-traffic noise ``sigma``
    (nine configurations) on one measured dataset, and compares each network
    advantage against the platform advantage measured on the *same* traces.
    """
    tax = reference_taxonomy() if cfg.num_classes == 12 else scaled_taxonomy(cfg.num_classes)
    harness = MeasuredHarness(taxonomy=tax, seed=cfg.seed, overlap=cfg.overlap)
    ds = harness.sample_dataset(cfg.traces_per_class)

    X_plat, y = featurize_dataset(ds, "platform")
    plat = estimate_leakage(X_plat, y, vantage="platform", prior=tax.prior, bank=cfg.bank,
                            bootstrap=min(cfg.bootstrap, 50), test_size=cfg.test_size,
                            seed=cfg.seed)

    rows: list[dict[str, Any]] = []
    for p in coalesce_probs:
        for sigma in sigmas:
            wire = HTTP2WireModel(coalesce_prob=p, cross_traffic_sigma=sigma, seed=cfg.seed)
            X, yy = featurize_dataset(apply_wire_model_dataset(ds, wire), "network")
            leak = estimate_leakage(X, yy, vantage="network", prior=tax.prior,
                                    bank=cfg.bank, bootstrap=min(cfg.bootstrap, 50),
                                    test_size=cfg.test_size, seed=cfg.seed)
            rows.append({
                "coalesce_prob": p,
                "cross_traffic_sigma": sigma,
                "network_advantage": leak.advantage,
                "network_mi_bits": leak.mi_lower_bound_bits,
                "below_platform": bool(leak.advantage < plat.advantage),
            })
    advs = [r["network_advantage"] for r in rows]
    return {
        "rq": "RQ6_wire_sweep",
        "platform_advantage": plat.advantage,
        "rows": rows,
        "network_advantage_range": [float(min(advs)), float(max(advs))],
        "configs_below_platform": int(sum(r["below_platform"] for r in rows)),
        "n_configs": len(rows),
        "note": "conclusion is invariant to the wire model iff every config stays below platform",
    }


# ----------------------------------------------- RQ6 overlap sensitivity
def run_overlap_sweep(
    cfg,
    overlaps: Sequence[float] = (0.0, 0.15, 0.3, 0.5, 0.7),
) -> dict[str, Any]:
    """Sensitivity of the measured harness to its **unstated** confusability knob.

    The paper says tasks are drawn from "overlapping class-conditional tool
    distributions" but never gives the degree of overlap, and that single number moves
    the RQ6 result more than anything else in the harness. Rather than pick a value and
    present it as if it were specified, we sweep it and publish the mapping, so a
    reader can see exactly which operating point any reported triple corresponds to.

    Also reports the network vantage **with and without** the HTTP/2 wire model, since
    the two differ sharply and the paper reports only the post-wire number.
    """
    tax = reference_taxonomy() if cfg.num_classes == 12 else scaled_taxonomy(cfg.num_classes)
    rows = []
    for ov in overlaps:
        harness = MeasuredHarness(taxonomy=tax, seed=cfg.seed, overlap=float(ov))
        ds = harness.sample_dataset(cfg.traces_per_class)
        entry: dict[str, Any] = {"overlap": float(ov)}
        for v in cfg.vantages:
            source = (apply_wire_model_dataset(ds, HTTP2WireModel(seed=cfg.seed))
                      if v == "network" else ds)
            X, y = featurize_dataset(source, v)
            entry[v] = estimate_leakage(
                X, y, vantage=v, prior=tax.prior, bank=cfg.bank,
                bootstrap=min(cfg.bootstrap, 50), test_size=cfg.test_size, seed=cfg.seed,
            ).advantage
        X, y = featurize_dataset(ds, "network")
        entry["network_pre_wire"] = estimate_leakage(
            X, y, vantage="network", prior=tax.prior, bank=cfg.bank,
            bootstrap=min(cfg.bootstrap, 50), test_size=cfg.test_size, seed=cfg.seed,
        ).advantage
        sizes = np.concatenate([t.total_sizes() for t in ds.traces])
        entry["payload_median_bytes"] = float(np.median(sizes))
        entry["payload_p90_bytes"] = float(np.percentile(sizes, 90))
        entry["fraction_below_one_mtu"] = float(np.mean(sizes < 1500))
        rows.append(entry)
        logger.info("overlap=%.2f: %s", ov,
                    {k: round(v, 3) for k, v in entry.items() if k in cfg.vantages})
    return {
        "rq": "RQ6_overlap_sensitivity",
        "rows": rows,
        "note": "overlap is not specified in the paper and dominates the RQ6 operating "
                "point; network_pre_wire isolates how much of the network number the "
                "HTTP/2 wire model removes",
    }


# ------------------------------------------------------- RQ6 attacker robustness
def run_attacker_robustness(cfg, vantage: str = "platform") -> dict[str, Any]:
    """RQ6 robustness: re-estimate leakage with each attacker family separately.

    The policy is to report the *strongest* attacker; this runner exposes the whole
    bank so a reader can see the strong learners agree and that k-NN sits below them
    as a conservative Bayes-risk floor (never inverted into a privacy guarantee).
    """
    ds, tax = build_dataset(cfg)
    if getattr(cfg, "source", "generator") == "measured" and vantage == "network":
        ds = apply_wire_model_dataset(ds, HTTP2WireModel(seed=cfg.seed))
    X, y = featurize_dataset(ds, vantage)

    rows = {}
    for name in robustness_bank():
        leak = estimate_leakage(X, y, vantage=vantage, prior=tax.prior, bank=(name,),
                                bootstrap=min(cfg.bootstrap, 100),
                                test_size=cfg.test_size, seed=cfg.seed)
        rows[name] = {"advantage": leak.advantage, "mi_bits": leak.mi_lower_bound_bits,
                      "bayes_risk": leak.bayes_risk}
    strong = [rows[n]["advantage"] for n in ("random_forest", "hist_gbdt", "logistic")
              if n in rows]
    return {
        "rq": "RQ6_attacker_robustness",
        "vantage": vantage,
        "source": getattr(cfg, "source", "generator"),
        "by_attacker": rows,
        "strong_learner_spread": float(max(strong) - min(strong)) if strong else float("nan"),
        "knn_advantage": rows.get("knn", {}).get("advantage"),
        "note": "strong learners should agree; kNN is the conservative floor",
    }


# ------------------------------------------------------------------- open world
def run_openworld(
    cfg,
    base_rates: Sequence[float] = (0.5, 0.2, 0.05),
    total_traces: int = 24000,
    vantage: str = "platform",
) -> dict[str, Any]:
    """Open-world protocol with an **out-of-taxonomy** ``other`` bucket.

    The paper's Table "openworld": monitored taxonomy classes are mixed with random
    tool traffic that carries no signature, at three monitored base rates. The total
    dataset size is held fixed so the three rows differ only in composition; as
    monitored prevalence falls, precision should hold while recall collapses.

    Also reports the *within-taxonomy* variant (half the classes folded into
    ``other``), which the paper notes is the friendlier setting.
    """
    prior_arg = zipf_prior(cfg.num_classes) if getattr(cfg, "prior", "uniform") == "zipf" else None
    tax = (reference_taxonomy(prior_arg) if cfg.num_classes == 12
           else scaled_taxonomy(cfg.num_classes, prior_arg))

    rows: list[dict[str, Any]] = []
    for rate in base_rates:
        n_monitored = max(cfg.num_classes, int(round(total_traces * rate)))
        per_class = max(2, n_monitored // cfg.num_classes)
        n_other = max(1, total_traces - per_class * cfg.num_classes)

        gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed,
                                  signature_upweight=cfg.signature_upweight)
        monitored = gen.sample_dataset(per_class)
        other = OtherTrafficGenerator(seed=cfg.seed + 101).sample_traces(n_other)
        ds = TraceDataset(
            traces=list(monitored.traces) + other,
            taxonomy_name=tax.name, taxonomy_hash=tax.content_hash(),
            source="generator+other",
            metadata={"base_rate": rate, "n_monitored": per_class * cfg.num_classes,
                      "n_other": n_other},
        )
        scores = _open_world_scores_for(ds, vantage, cfg)
        row = {
            "monitored_base_rate": rate,
            "n_monitored": per_class * cfg.num_classes,
            "n_monitored_per_class": per_class,
            "n_other": n_other,
            "other_kind": "out_of_taxonomy_random_tool_mix",
            **scores,
        }
        # A row where the attacker never predicts a monitored class is degenerate:
        # precision is 0/0. Say so rather than reporting a bare 0.00 that reads like a
        # measured value.
        if scores["tp"] + scores["fp"] == 0:
            row["degenerate"] = (
                "the attacker predicted 'other' for every test trace; with "
                f"{per_class} monitored traces per class there is too little "
                "monitored training data to score precision"
            )
            logger.warning("open-world base rate %.0f%%: %s", rate * 100,
                           row["degenerate"])
        rows.append(row)

    within = _run_openworld_within_taxonomy(cfg, tax, vantage)
    return {
        "rq": "openworld",
        "vantage": vantage,
        "total_traces": total_traces,
        "rows": rows,
        "within_taxonomy": within,
        "note": "precision should hold while recall collapses as monitored tasks become rare",
    }


def _run_openworld_within_taxonomy(cfg, tax: Taxonomy, vantage: str) -> dict[str, Any]:
    """The friendlier variant: half the taxonomy folded into ``other``."""
    gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed,
                              signature_upweight=cfg.signature_upweight)
    base = gen.sample_dataset(cfg.traces_per_class)
    n_monitored = max(1, tax.num_classes // 2)
    traces = []
    for t in base.traces:
        if t.label >= n_monitored:
            t = Trace(calls=t.calls, label=OTHER_LABEL, family=t.family,
                      metadata={**t.metadata, "folded_from": int(t.label)})
        traces.append(t)
    ds = TraceDataset(traces=traces, taxonomy_name=tax.name,
                      taxonomy_hash=tax.content_hash(), source="generator",
                      metadata={"open_world": "within_taxonomy"})
    scores = _open_world_scores_for(ds, vantage, cfg)
    return {"n_monitored_classes": n_monitored, "other_kind": "within_taxonomy_fold",
            **scores}


def _open_world_scores_for(ds: TraceDataset, vantage: str, cfg) -> dict[str, Any]:
    """Fit the bank on a stratified split and score the monitored classes."""
    X, y = featurize_dataset(ds, vantage)
    idx = np.arange(len(y))
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    thin = {c: n for c, n in counts.items() if n < 2}
    if thin:
        raise ValueError(
            f"Open-world split needs >= 2 traces per label; {sorted(thin)} have "
            f"{sorted(set(thin.values()))}. Raise total_traces (the 'other' bucket is "
            f"what shrinks first at low base rates)."
        )
    tr, te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)
    best_err, best_pred, best_name = np.inf, None, ""
    for name in cfg.bank:
        clf = build_attacker(name, seed=cfg.seed)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        err = float(np.mean(pred != y[te]))
        if err < best_err:
            best_err, best_pred, best_name = err, pred, name
    assert best_pred is not None
    scores: dict[str, Any] = dict(open_world_scores(y[te], best_pred))
    scores["accuracy"] = float(1.0 - best_err)
    scores["best_estimator"] = best_name
    return scores


# ------------------------------------------------------ DP shaping epsilon sweep
def run_dp_sweep(
    cfg,
    epsilons: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    adjacencies: Sequence[str] = ("all_pairs", "per_window"),
    windows: int | None = None,
    window_s: float | None = None,
) -> dict[str, Any]:
    """Map a DP shaping budget onto the advantage axis (the paper's open problem).

    For each ``epsilon`` we apply :class:`~leakbench.defenses.dp_shaping.DPShapingDefense`,
    measure the residual advantage at the network vantage, and compare it against the
    analytic ceiling ``Adv <= V_prior (e^eps - 1)/R_pi`` (valid at distance 1, i.e.
    under the all-pairs adjacency). A measured value above the ceiling would falsify
    either the mechanism or the estimator, so the check is reported per row.

    The shaping horizon (``windows * window_s``) is a public deployment parameter; by
    default it is set to cover the longest session in the dataset, and the number of
    traces whose payload did not drain inside it is reported (those are not covered by
    the stated epsilon).
    """
    ds, tax = build_dataset(cfg)
    durations = [float(t.gaps().sum()) for t in ds.traces]
    W = windows or 64
    tau = window_s if window_s is not None else max(0.05, float(np.max(durations)) / W)

    X0, y0 = featurize_dataset(ds, "network")
    undefended = estimate_leakage(X0, y0, vantage="network", prior=tax.prior, bank=cfg.bank,
                                  bootstrap=min(cfg.bootstrap, 50), test_size=cfg.test_size,
                                  seed=cfg.seed)

    rows: list[dict[str, Any]] = []
    for adjacency in adjacencies:
        for eps in epsilons:
            defense = DPShapingDefense(epsilon=eps, windows=W, window_s=tau,
                                       adjacency=adjacency)
            dres = defense.apply(ds, seed=cfg.seed)
            X, y = featurize_dataset(dres.dataset, "network")
            leak = estimate_leakage(X, y, vantage="network", prior=tax.prior, bank=cfg.bank,
                                    bootstrap=min(cfg.bootstrap, 50),
                                    test_size=cfg.test_size, seed=cfg.seed)
            ceiling = dp_advantage_ceiling(eps, tax.prior, distance=1.0)
            applies = adjacency == "all_pairs"
            ci_lo = float(leak.advantage_ci[0])
            rows.append({
                "epsilon": eps,
                "adjacency": adjacency,
                "advantage": leak.advantage,
                "advantage_ci": list(leak.advantage_ci),
                "mi_bits": leak.mi_lower_bound_bits,
                "ceiling_distance_1": ceiling,
                "ceiling_applies": applies,
                # The ceiling bounds the POPULATION Bayes risk. A finite test set can
                # exceed it by chance (the advantage is granular in 1/n_test), so a
                # point-estimate exceedance is reported but only a CI that clears the
                # ceiling counts as a real violation.
                "within_ceiling_point": bool(leak.advantage <= ceiling + 1e-12)
                if applies else None,
                "significant_violation": bool(ci_lo > ceiling) if applies else None,
                "bandwidth_overhead": dres.bandwidth_overhead,
                "latency_overhead": dres.latency_overhead,
                "overflow_traces": defense.overflow_traces,
                "n_traces": defense.n_transformed,
            })
    violations = [r for r in rows if r["ceiling_applies"] and r["significant_violation"]]
    exceedances = [r for r in rows if r["ceiling_applies"] and not r["within_ceiling_point"]]
    for r in violations:
        logger.warning(
            "DP CEILING VIOLATION at epsilon=%s: advantage CI [%.3f, %.3f] lies above "
            "ceiling %.3f -- mechanism or estimator bug.", r["epsilon"],
            r["advantage_ci"][0], r["advantage_ci"][1], r["ceiling_distance_1"],
        )
    for r in exceedances:
        if r not in violations:
            logger.info(
                "DP ceiling exceeded by the point estimate at epsilon=%s (%.3f > %.3f) "
                "but not significantly (CI low %.3f); finite-sample noise on %d test "
                "traces.", r["epsilon"], r["advantage"], r["ceiling_distance_1"],
                r["advantage_ci"][0], undefended.n_test,
            )
    total_overflow = sum(r["overflow_traces"] for r in rows)
    return {
        "rq": "dp_shaping_sweep",
        "vantage": "network",
        "undefended_advantage": undefended.advantage,
        "shaping_horizon_s": W * tau,
        "windows": W,
        "window_s": tau,
        "n_test": undefended.n_test,
        "rows": rows,
        "ceiling_violations": len(violations),
        "ceiling_point_exceedances": len(exceedances),
        "overflow_traces_total": int(total_overflow),
        "note": "ceiling Adv <= V_prior (e^eps - 1)/R_pi holds at distance 1 (all-pairs "
                "adjacency) for the POPULATION channel; point-estimate exceedances within "
                "the bootstrap CI are finite-sample noise. Under per_window adjacency the "
                "task-level ceiling must be evaluated at the real trace distance and is "
                "vacuous. Traces counted in overflow_traces_total are not covered by the "
                "stated epsilon.",
    }


# ------------------------------------------------------------ RQ7 taxonomy sweep
def run_taxonomy_sweep(
    cfg,
    sizes: Sequence[int] = (8, 12, 16, 24),
    vantage: str = "platform",
) -> dict[str, Any]:
    """RQ7: does the finding survive a larger secret space?

    Sweeps ``|T|`` at the configured trace count. The paper's claim is that the
    *normalized* advantage dips gently while the *absolute* leaked information rises,
    so the qualitative conclusion strengthens with taxonomy size.
    """
    rows: list[dict[str, Any]] = []
    for k in sizes:
        tax = scaled_taxonomy(k)
        gen = CalibratedGenerator(taxonomy=tax, seed=cfg.seed,
                                  signature_upweight=cfg.signature_upweight)
        ds = gen.sample_dataset(cfg.traces_per_class)
        X, y = featurize_dataset(ds, vantage)
        leak = estimate_leakage(X, y, vantage=vantage, prior=tax.prior, bank=cfg.bank,
                                bootstrap=min(cfg.bootstrap, 50),
                                test_size=cfg.test_size, seed=cfg.seed)
        rows.append({
            "num_classes": k,
            "entropy_bits": tax.entropy_bits(),
            "advantage": leak.advantage,
            "advantage_ci": list(leak.advantage_ci),
            "mi_bits": leak.mi_lower_bound_bits,
            "n_traces": len(ds),
        })
        logger.info("|T|=%d adv=%.3f MI=%.3f bits", k, leak.advantage,
                    leak.mi_lower_bound_bits)
    mi = [float(r["mi_bits"]) for r in rows]
    return {
        "rq": "RQ7_taxonomy_sweep",
        "vantage": vantage,
        "traces_per_class": cfg.traces_per_class,
        "rows": rows,
        "mi_is_monotone_increasing": bool(all(b >= a - 1e-9 for a, b in zip(mi, mi[1:]))),
        "note": "advantage may dip with |T| while absolute MI rises",
    }


# ------------------------------------------------------------------ RQ8 live
def run_live(
    cfg,
    traces_per_class: int = 70,
    seeds: Sequence[int] | None = None,
    use_tls: bool = True,
    server_host: str | None = None,
    server_port: int | None = None,
    server_ca_file: str | None = None,
    planner_factory: Any = None,
) -> dict[str, Any]:
    """RQ8: capture live traffic off a real (TLS) loopback socket and estimate leakage.

    By default everything runs on 127.0.0.1. Pass ``server_host``/``server_port`` (and
    ``server_ca_file`` for TLS) to drive a tool server on another host: the capture
    proxy then sits on this host's uplink and the captured bytes are the ones that
    crossed the real link, which is the WAN variant of the experiment
    (``docs/ec2_experiment_guide.md`` §5a).

    ``planner_factory(seed) -> Planner`` swaps in a different controller; pass one
    built around :class:`~leakbench.data.live_capture.LLMPlanner` to run the neural
    variant (the dataset metadata then records ``planner_kind = neural_llm``).

    Reports per-vantage advantage per seed, the size-only/timing-only projections on
    *captured* bytes, the capture mode actually achieved (real TLS vs. modeled framing
    over captured TCP), and the link type.
    """
    from leakbench.data.live_capture import LIVE_CLASSES, LiveCaptureHarness

    seeds = list(seeds) if seeds is not None else [cfg.seed, cfg.seed + 1]
    prior = np.full(len(LIVE_CLASSES), 1.0 / len(LIVE_CLASSES))
    per_seed = []
    for seed in seeds:
        harness = LiveCaptureHarness(
            seed=seed, use_tls=use_tls,
            server_host=server_host, server_port=server_port,
            server_ca_file=server_ca_file,
            planner=planner_factory(seed) if planner_factory is not None else None,
        )
        ds = harness.capture_dataset(traces_per_class, classes=LIVE_CLASSES)
        # Record the inter-call gap distribution. This is the quantity the network
        # observer actually times, and under a neural planner it is dominated by model
        # generation latency -- so it is what explains a collapsing timing channel.
        # Reporting the timing-only advantage without it leaves the mechanism unstated.
        _gaps = np.concatenate([t.gaps()[1:] for t in ds.traces if t.n > 1]) \
            if any(t.n > 1 for t in ds.traces) else np.zeros(1)
        entry: dict[str, Any] = {"seed": seed, "capture": dict(ds.metadata),
                                 "n_traces": len(ds), "vantages": {}, "projections": {},
                                 "inter_call_gap_s": {
                                     "median": float(np.median(_gaps)),
                                     "p10": float(np.percentile(_gaps, 10)),
                                     "p90": float(np.percentile(_gaps, 90))}}
        for v in ("network", "platform", "providers"):
            X, y = featurize_dataset(ds, v)
            leak = estimate_leakage(X, y, vantage=v, prior=prior, bank=cfg.bank,
                                    bootstrap=min(cfg.bootstrap, 100),
                                    test_size=cfg.test_size, seed=seed)
            entry["vantages"][v] = {"advantage": leak.advantage,
                                    "advantage_ci": list(leak.advantage_ci),
                                    "mi_bits": leak.mi_lower_bound_bits}
        X_net = np.vstack([network_map(t) for t in ds.traces])
        y = ds.labels()
        for proj in ("size_only", "timing_only"):
            leak = estimate_leakage(project_features(X_net, proj), y,
                                    vantage=f"network:{proj}", prior=prior, bank=cfg.bank,
                                    bootstrap=min(cfg.bootstrap, 100),
                                    test_size=cfg.test_size, seed=seed)
            entry["projections"][proj] = leak.advantage
        entry["mean_trace_length"] = float(np.mean([t.n for t in ds.traces]))
        per_seed.append(entry)
        logger.info("live seed %d: %s", seed,
                    {k: round(v["advantage"], 3) for k, v in entry["vantages"].items()})

    ordering_holds = sum(
        1 for e in per_seed
        if e["vantages"]["network"]["advantage"] < e["vantages"]["platform"]["advantage"]
    )
    return {
        "rq": "RQ8_live",
        "classes": list(LIVE_CLASSES),
        "entropy_bits": float(np.log2(len(LIVE_CLASSES))),
        "traces_per_class": traces_per_class,
        "seeds": seeds,
        "per_seed": per_seed,
        "net_below_platform_in_seeds": ordering_holds,
        "capture_mode": per_seed[0]["capture"].get("capture_mode") if per_seed else None,
        "link": per_seed[0]["capture"].get("link") if per_seed else None,
        "planner_kind": per_seed[0]["capture"].get("planner_kind") if per_seed else None,
        "gaps": per_seed[0]["capture"].get("gaps") if per_seed else None,
        "note": "network sizes are CAPTURED off the proxy socket, not modeled; "
                "remaining gaps are recorded in the capture metadata",
    }


# ---------------------------------------------------------------- learning curve
def run_learning_curve(
    cfg,
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
    vantage: str = "providers",
) -> dict[str, Any]:
    """F-BLEAU-style convergence check: is the estimate saturated at this sample size?

    Re-estimates leakage on nested subsets of the training data (the test set is held
    fixed) and reports the advantage per training size for the whole bank. A curve
    that has flattened means more traces would not raise the estimate much; a still-
    rising curve means the reported value is an under-estimate --- which matters
    because the advantage is only ever a *lower* bound on leakage.
    """
    ds, tax = build_dataset(cfg)
    X, y = featurize_dataset(ds, vantage)
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.test_size,
                                      random_state=cfg.seed, stratify=y)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(tr_idx)

    rows = []
    for frac in fractions:
        n = max(len(np.unique(y)) * 2, int(round(frac * len(order))))
        sub = order[:n]
        row: dict[str, Any] = {"train_fraction": frac, "n_train": int(n)}
        for name in cfg.bank:
            clf = build_attacker(name, seed=cfg.seed)
            clf.fit(X[sub], y[sub])
            err = float(np.mean(clf.predict(X[te_idx]) != y[te_idx]))
            guessing = float(1.0 - np.max(tax.prior))
            row[name] = float(min(1.0, max(0.0, (guessing - err) / guessing)))
        rows.append(row)

    deltas = {}
    for name in cfg.bank:
        if len(rows) >= 2:
            deltas[name] = float(rows[-1][name] - rows[-2][name])
    return {
        "rq": "learning_curve",
        "vantage": vantage,
        "n_test": int(len(te_idx)),
        "rows": rows,
        "last_step_delta": deltas,
        "note": "a small last-step delta means the estimate has converged at this sample size",
    }
