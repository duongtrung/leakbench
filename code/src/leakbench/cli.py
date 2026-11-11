"""LeakBench command-line interface.

Examples
--------
    python -m leakbench.cli --help
    python -m leakbench.cli info
    python -m leakbench.cli demo --config configs/synthetic_demo.yaml
    python -m leakbench.cli run --config configs/default.yaml --stage vantage
    python -m leakbench.cli validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from leakbench._version import __version__
from leakbench.experiments.config import ExperimentConfig, load_config
from leakbench.experiments.runners import (
    run_ablation,
    run_attacker_robustness,
    run_bank_comparison,
    run_coarse_secret,
    run_defenses,
    run_dp_sweep,
    run_estimator_agreement,
    run_feature_importance,
    run_generator_variance,
    run_learning_curve,
    run_measured,
    run_measured_multiseed,
    run_mixing_sweep,
    run_openworld,
    run_overlap_sweep,
    run_prior_sensitivity,
    run_projections,
    run_scalability,
    run_separability_sweep,
    run_sequence_attacker,
    run_subgroup,
    run_validation,
    run_vantage,
    run_wire_sweep,
)
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.io import dump_json
from leakbench.utils.logging import get_logger
from leakbench.utils.provenance import capture_provenance
from leakbench.utils.seeding import seed_everything

logger = get_logger("leakbench.cli")

STAGES = {
    "vantage": run_vantage,
    "ablation": run_ablation,
    "subgroup": run_subgroup,
    "defenses": run_defenses,
    "validation": lambda cfg: run_validation(cfg),
    "measured": run_measured,
    "measured_multiseed": lambda cfg: run_measured_multiseed(cfg),
    "wire_sweep": lambda cfg: run_wire_sweep(cfg),
    "robustness": lambda cfg: run_attacker_robustness(cfg),
    "projections": run_projections,
    "openworld": lambda cfg: run_openworld(cfg),
    "dp_sweep": lambda cfg: run_dp_sweep(cfg),
    "learning_curve": lambda cfg: run_learning_curve(cfg),
    # Paper results that were previously prose-only:
    "coarse": run_coarse_secret,
    "feature_importance": lambda cfg: run_feature_importance(cfg),
    "separability": lambda cfg: run_separability_sweep(cfg),
    "scalability": lambda cfg: run_scalability(cfg),
    "generator_variance": lambda cfg: run_generator_variance(cfg),
    "prior_sensitivity": run_prior_sensitivity,
    "estimator_agreement": run_estimator_agreement,
    "overlap_sweep": lambda cfg: run_overlap_sweep(cfg),
    "bank_comparison": run_bank_comparison,
    "sequence_attacker": lambda cfg: run_sequence_attacker(cfg),
    "mixing": lambda cfg: run_mixing_sweep(cfg),
}

# Stages included when ``--stage all`` is used: the RQ1-RQ6 core. The heavier sweeps
# are opt-in by name so a routine run stays the "under three minutes" pass the paper
# advertises.
DEFAULT_STAGES = ("vantage", "ablation", "subgroup", "defenses", "validation", "measured")


def _cmd_info(_: argparse.Namespace) -> int:
    tax = reference_taxonomy()
    print(f"LeakBench v{__version__}")
    print(f"Reference taxonomy: {tax.name} (hash {tax.content_hash()})")
    print(f"  classes: {tax.num_classes}, families: {len(tax.families)}")
    print(f"  H(C) = {tax.entropy_bits():.3f} bits (uniform prior)")
    print(f"  guessing risk R_pi = {tax.guessing_risk():.3f}")
    print("Vantages: network < platform < providers (providers refine both)")
    print("Stages:", ", ".join(STAGES))
    print("\nNOTE: no paper-level experimental results are claimed by this repository")
    print("unless real benchmark outputs are generated and their provenance recorded.")
    return 0


def _run_stage(cfg: ExperimentConfig, stage: str) -> dict:
    if stage not in STAGES:
        raise SystemExit(f"Unknown stage {stage!r}; choose from {list(STAGES)}.")
    logger.info("Running stage %s (source=%s, source classes=%d, %d/class)",
                stage, cfg.source, cfg.num_classes, cfg.traces_per_class)
    return STAGES[stage](cfg)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    seed_everything(cfg.seed)
    tax = reference_taxonomy()
    prov = capture_provenance(cfg.seed, tax.content_hash(), tax.name, cfg.to_dict())

    if args.stage in (None, "all"):
        stages = list(DEFAULT_STAGES)
    elif args.stage == "everything":
        stages = list(STAGES)
    else:
        stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    out_dir = Path(cfg.output_dir)
    for stage in stages:
        result = _run_stage(cfg, stage)
        payload = {"provenance": prov.to_dict(), "result": result,
                   "status": "real_computation_on_synthetic_or_local_data"}
        path = dump_json(payload, out_dir / f"{stage}.json")
        logger.info("Wrote %s", path)
        _print_summary(stage, result)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    seed_everything(cfg.seed)
    result = run_validation(cfg)
    print(f"Estimator validation: max |MI - analytic| = {result['max_mi_abs_error']:.3f} bits")
    for r in result["rows"]:
        print(f"  p={r['p']:.2f}  analytic MI={r['analytic_mi_bits']:.3f}  "
              f"estimated MI={r['estimated_mi_bits']:.3f}")
    return 0


def _print_summary(stage: str, result: dict) -> None:
    if stage == "vantage":
        for v, a in result["advantage_by_vantage"].items():
            print(f"  {v:10s} advantage = {a:.3f}")
        for w in result.get("refinement_warnings", []):
            print(f"  WARNING: {w}")
    elif stage == "defenses":
        for r in result["rows"]:
            print(f"  {r['defense']:14s} adv={r['advantage']:.3f}  "
                  f"BW=+{r['bandwidth_overhead']*100:.0f}%  lat=+{r['latency_overhead']*100:.0f}%")
    elif stage == "measured":
        for v, a in result["advantage_by_vantage"].items():
            print(f"  {v:10s} advantage = {a:.3f}")
    elif stage == "measured_multiseed":
        for v, s in result["summary"].items():
            print(f"  {v:10s} advantage = {s['mean']:.3f} +/- {s['std']:.3f} "
                  f"(MI {s['mi_bits_at_reference_seed']:.2f} bits)")
        for step, s in result["ordering_steps"].items():
            print(f"  {step}: holds in {s['holds_in_seeds']}/{s['n_seeds']} seeds, "
                  f"mean gap {s['mean_gap']:.3f}, p={s['p_value_one_sided']:.2g}")
    elif stage == "wire_sweep":
        lo, hi = result["network_advantage_range"]
        print(f"  network advantage in [{lo:.3f}, {hi:.3f}]; platform "
              f"{result['platform_advantage']:.3f}; below platform in "
              f"{result['configs_below_platform']}/{result['n_configs']} configs")
    elif stage == "robustness":
        for name, r in result["by_attacker"].items():
            print(f"  {name:14s} advantage = {r['advantage']:.3f}")
    elif stage == "projections":
        for name, r in result["rows"].items():
            print(f"  {name:22s} adv={r['advantage']:.3f}  ({r['vantage']})")
    elif stage == "openworld":
        for r in result["rows"]:
            print(f"  base rate {r['monitored_base_rate']:>5.0%}: P={r['precision']:.2f} "
                  f"R={r['recall']:.2f} F1={r['f1']:.2f}")
        w = result["within_taxonomy"]
        print(f"  within-taxonomy other: F1={w['f1']:.2f}")
    elif stage == "dp_sweep":
        print(f"  undefended advantage = {result['undefended_advantage']:.3f}")
        for r in result["rows"]:
            ceil = (f"<= {r['ceiling_distance_1']:.3f}" if r["ceiling_applies"]
                    else "ceiling n/a (per-window adjacency)")
            print(f"  eps={r['epsilon']:>6.1f} {r['adjacency']:11s} adv={r['advantage']:.3f} "
                  f"{ceil}  BW=+{r['bandwidth_overhead']*100:.0f}%")
        if result["ceiling_violations"]:
            print(f"  WARNING: {result['ceiling_violations']} DP ceiling violation(s)")
    elif stage == "learning_curve":
        for r in result["rows"]:
            vals = "  ".join(f"{k}={v:.3f}" for k, v in r.items()
                             if k not in ("train_fraction", "n_train"))
            print(f"  n_train={r['n_train']:>6d}  {vals}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="leakbench",
        description="LeakBench: metadata-privacy leakage benchmark for LLM agents.",
    )
    p.add_argument("--version", action="version", version=f"leakbench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print taxonomy + configuration summary").set_defaults(
        func=_cmd_info
    )

    pr = sub.add_parser("run", help="run one or more experiment stages")
    pr.add_argument("--config", type=str, default=None, help="path to a YAML config")
    pr.add_argument("--stage", type=str, default="all",
                    help=(f"comma-separated stages from {list(STAGES)}; "
                          f"'all' = {list(DEFAULT_STAGES)}; 'everything' = every stage"))
    pr.set_defaults(func=_cmd_run)

    pd = sub.add_parser("demo", help="run the synthetic smoke-test demo")
    pd.add_argument("--config", type=str, default="configs/synthetic_demo.yaml")
    pd.set_defaults(func=_cmd_demo)

    pv = sub.add_parser("validate", help="validate the estimator vs. the analytic channel")
    pv.add_argument("--config", type=str, default=None)
    pv.set_defaults(func=_cmd_validate)
    return p


def _cmd_demo(args: argparse.Namespace) -> int:
    # Delegate to the demo script's function for a single source of truth.
    from leakbench.experiments.demo import run_synthetic_demo

    cfg = load_config(args.config) if Path(args.config).exists() else ExperimentConfig(
        name="synthetic_demo", traces_per_class=40, bootstrap=50,
        output_dir="outputs/synthetic_demo",
    )
    out = run_synthetic_demo(cfg)
    print(f"Synthetic demo complete. Outputs (labeled synthetic) in: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
