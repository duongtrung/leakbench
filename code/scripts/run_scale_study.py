#!/usr/bin/env python3
"""Run the **full paper-scale study** end to end and archive it with provenance.

This is the script to run on a server (see ``docs/ec2_experiment_guide.md``). It
executes every stage at the paper's sizes, writes one JSON per stage plus a combined
``summary.json`` and a human-readable ``report.md``, and records how long each stage
took and what it actually ran on.

    # full run (paper sizes)
    python scripts/run_scale_study.py --output outputs/scale_study

    # fast end-to-end validation before paying for the big instance
    python scripts/run_scale_study.py --quick --output outputs/scale_quick

    # subset
    python scripts/run_scale_study.py --only vantage,measured_multiseed,live

Stages (paper mapping; the full table is in docs/ec2_experiment_guide.md §2):

    vantage, ablation, subgroup, defenses, validation   RQ1-RQ5 (generator)
    coarse                                               Tbl. coarse (4-way family secret)
    feature_importance                                   RQ2 corroboration
    separability, scalability                            Fig. separability / scalability
    generator_variance, prior_sensitivity                sensitivity checks (§Eval)
    estimator_agreement                                  App. "Estimator agreement"
    projections                                          case study (deployed attacks)
    measured_multiseed                                   RQ6 (5 seeds, paired t-test)
    wire_sweep, robustness                               RQ6 robustness
    openworld                                            RQ6 open-world (out-of-taxonomy)
    taxonomy_sweep                                       RQ7
    live                                                 RQ8 (captured TLS loopback)
    dp_sweep, learning_curve                             beyond the paper's artifact

Together these cover every measured result in the paper. Three of them report a *flag*
rather than assuming the paper's claim (separability monotonicity, estimator ordering
agreement, and the 5/5 measured ordering); read those before writing numbers up.

Nothing here downloads data. Every number is computed at run time from the calibrated
generator, real local tools, or captured loopback traffic, and every output file
records which.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

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
    run_live,
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
    run_taxonomy_sweep,
    run_validation,
    run_vantage,
)
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.io import dump_json
from leakbench.utils.logging import get_logger
from leakbench.utils.provenance import capture_provenance
from leakbench.utils.seeding import seed_everything

logger = get_logger("leakbench.scale_study")

PAPER_DEFENSES = ("none", "pad-512", "shuffle", "constant-rate", "cover", "pad-max",
                  "full", "gen-shaping", "dp-shaping")


def _stage_plan(gen_cfg: ExperimentConfig, meas_cfg: ExperimentConfig,
                args: argparse.Namespace) -> dict[str, Callable[[], dict[str, Any]]]:
    """Every stage as a zero-arg callable, so the driver can time and skip them."""
    return {
        # --- RQ1-RQ5, calibrated generator
        "vantage": lambda: run_vantage(gen_cfg),
        "coarse": lambda: run_coarse_secret(gen_cfg),
        "ablation": lambda: run_ablation(gen_cfg),
        "feature_importance": lambda: run_feature_importance(gen_cfg),
        "subgroup": lambda: run_subgroup(gen_cfg),
        "defenses": lambda: run_defenses(replace(gen_cfg, defenses=PAPER_DEFENSES)),
        "validation": lambda: run_validation(gen_cfg, n_per_class=args.validation_per_class),
        "separability": lambda: run_separability_sweep(gen_cfg),
        "scalability": lambda: run_scalability(gen_cfg),
        # --- sensitivity checks the paper reports in prose
        "generator_variance": lambda: run_generator_variance(gen_cfg, n_draws=args.draws),
        "prior_sensitivity": lambda: run_prior_sensitivity(gen_cfg),
        "estimator_agreement": lambda: run_estimator_agreement(gen_cfg),
        "bank_comparison": lambda: run_bank_comparison(gen_cfg),
        "sequence_attacker": lambda: run_sequence_attacker(gen_cfg),
        "mixing": lambda: run_mixing_sweep(gen_cfg),
        # --- case study: deployed attacks as projections, on measured traces
        "projections": lambda: run_projections(meas_cfg),
        # --- RQ6, measured harness
        "measured_multiseed": lambda: run_measured_multiseed(
            meas_cfg, seeds=[meas_cfg.seed + i for i in range(args.seeds)]
        ),
        "overlap_sweep": lambda: run_overlap_sweep(meas_cfg),
        "wire_sweep": lambda: run_wire_sweep_stage(meas_cfg),
        "robustness": lambda: run_attacker_robustness(meas_cfg, vantage="platform"),
        "openworld": lambda: run_openworld(gen_cfg, total_traces=args.openworld_total),
        # --- RQ7
        "taxonomy_sweep": lambda: run_taxonomy_sweep(gen_cfg, sizes=tuple(args.tax_sizes)),
        # --- RQ8
        "live": lambda: run_live(gen_cfg, traces_per_class=args.live_per_class,
                                 seeds=[gen_cfg.seed, gen_cfg.seed + 1],
                                 use_tls=not args.no_tls),
        # --- beyond the paper's artifact
        "dp_sweep": lambda: run_dp_sweep(gen_cfg),
        "learning_curve": lambda: run_learning_curve(gen_cfg),
    }


def run_wire_sweep_stage(cfg: ExperimentConfig) -> dict[str, Any]:
    from leakbench.experiments.runners import run_wire_sweep

    return run_wire_sweep(cfg)


def _summarize(results: dict[str, Any]) -> dict[str, Any]:
    """A compact, comparable digest of the run (the numbers a reader wants first)."""
    out: dict[str, Any] = {}
    if "vantage" in results:
        out["RQ1_advantage_by_vantage"] = results["vantage"]["advantage_by_vantage"]
        out["RQ1_mi_bits"] = {
            v: results["vantage"]["vantages"][v]["mi_lower_bound_bits"]
            for v in results["vantage"]["vantages"]
        }
        out["RQ1_refinement_warnings"] = results["vantage"]["refinement_warnings"]
    if "ablation" in results:
        out["RQ2_ablation"] = {k: v["advantage"]
                               for k, v in results["ablation"]["ablations"].items()}
    if "subgroup" in results:
        out["RQ3_by_family"] = results["subgroup"]["by_family"]
        out["RQ3_by_length_band"] = results["subgroup"]["by_length_band"]
    if "defenses" in results:
        out["RQ4_defenses"] = [
            {"defense": r["defense"], "advantage": r["advantage"],
             "bandwidth_overhead": r["bandwidth_overhead"],
             "latency_overhead": r["latency_overhead"]}
            for r in results["defenses"]["rows"]
        ]
    if "validation" in results:
        out["RQ5_max_mi_abs_error_bits"] = results["validation"]["max_mi_abs_error"]
    if "measured_multiseed" in results:
        out["RQ6_measured"] = results["measured_multiseed"]["summary"]
        out["RQ6_ordering_steps"] = results["measured_multiseed"]["ordering_steps"]
    if "wire_sweep" in results:
        out["RQ6_wire_sweep"] = {
            "network_advantage_range": results["wire_sweep"]["network_advantage_range"],
            "platform_advantage": results["wire_sweep"]["platform_advantage"],
            "configs_below_platform": results["wire_sweep"]["configs_below_platform"],
            "n_configs": results["wire_sweep"]["n_configs"],
        }
    if "overlap_sweep" in results:
        out["RQ6_overlap_sensitivity"] = results["overlap_sweep"]["rows"]
    if "robustness" in results:
        out["RQ6_attackers"] = {k: v["advantage"]
                                for k, v in results["robustness"]["by_attacker"].items()}
    if "openworld" in results:
        out["RQ6_openworld"] = [
            {k: r[k] for k in ("monitored_base_rate", "precision", "recall", "f1")}
            for r in results["openworld"]["rows"]
        ]
    if "taxonomy_sweep" in results:
        out["RQ7_taxonomy_sweep"] = results["taxonomy_sweep"]["rows"]
    if "live" in results:
        out["RQ8_live"] = {
            "capture_mode": results["live"]["capture_mode"],
            "per_seed": [
                {"seed": e["seed"],
                 **{v: e["vantages"][v]["advantage"] for v in e["vantages"]},
                 **e["projections"]}
                for e in results["live"]["per_seed"]
            ],
            "net_below_platform_in_seeds": results["live"]["net_below_platform_in_seeds"],
        }
    if "projections" in results:
        out["case_study_projections"] = {
            k: v["advantage"] for k, v in results["projections"]["rows"].items()
        }
    if "dp_sweep" in results:
        out["dp_sweep"] = {
            "undefended_advantage": results["dp_sweep"]["undefended_advantage"],
            "ceiling_violations": results["dp_sweep"]["ceiling_violations"],
            "rows": [{k: r[k] for k in ("epsilon", "adjacency", "advantage",
                                        "ceiling_distance_1", "bandwidth_overhead")}
                     for r in results["dp_sweep"]["rows"]],
        }
    if "learning_curve" in results:
        out["learning_curve_last_step_delta"] = results["learning_curve"]["last_step_delta"]
    if "coarse" in results:
        out["RQ1b_coarse_secret"] = results["coarse"]["rows"]
    if "feature_importance" in results:
        out["RQ2b_feature_importance"] = results["feature_importance"]["importance_by_block"]
    if "separability" in results:
        out["separability_sweep"] = {
            "knob": results["separability"].get("separability_knob", "dirichlet_alpha"),
            "rows": results["separability"]["rows"],
            "advantage_monotone": results["separability"].get("advantage_monotone"),
            "family_structure_knob": results["separability"].get("family_structure_knob"),
        }
    if "scalability" in results:
        out["scalability"] = {"rows": results["scalability"]["rows"],
                              "loglog_slope": results["scalability"]["loglog_slope"]}
    if "generator_variance" in results:
        out["generator_draw_variance"] = results["generator_variance"]["summary"]
    if "prior_sensitivity" in results:
        out["prior_sensitivity"] = {
            k: {vv: d["advantage"] for vv, d in v["vantages"].items()}
            for k, v in results["prior_sensitivity"]["rows"].items()
        }
    if "estimator_agreement" in results:
        out["estimator_agreement"] = results["estimator_agreement"]["agreement"]
    if "mixing" in results:
        out["multi_user_mixing"] = [
            {k: r[k] for k in ("n_users", "advantage", "observed_stream_growth")}
            for r in results["mixing"]["rows"]
        ]
    if "sequence_attacker" in results:
        out["sequence_attacker"] = {
            v: {"summary": e["summary_advantage"], "sequence": e["sequence_advantage"],
                "gain": e["advantage_gain"], "best": e["sequence_best_estimator"]}
            for v, e in results["sequence_attacker"]["rows"].items()
        }
    if "bank_comparison" in results:
        out["estimator_bank_comparison"] = {
            v: {"core": e["core"]["advantage"], "full": e["full"]["advantage"],
                "gain": e["advantage_gain"], "best": e["full"]["best_estimator"]}
            for v, e in results["bank_comparison"]["rows"].items()
        }
    return out


def _report_md(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """A short markdown digest so a reader does not have to open the JSON."""
    lines = [
        "# LeakBench scale study",
        "",
        f"- host: `{meta['platform']}`",
        f"- started: {meta['started_utc']}  (wall clock {meta['elapsed_s']:.0f} s)",
        f"- generator: {meta['generator_traces_per_class']} traces/class, "
        f"measured: {meta['measured_traces_per_class']} traces/class x "
        f"{meta['seeds']} seeds, live: {meta['live_traces_per_class']}/class",
        f"- seed: {meta['seed']}, taxonomy: {meta['taxonomy_name']} "
        f"(hash `{meta['taxonomy_hash']}`)",
        "",
        "Every number below is computed from the calibrated generator, real local "
        "tools, or captured loopback traffic at run time. None is a measurement of a "
        "deployed product.",
        "",
    ]
    if "RQ1_advantage_by_vantage" in summary:
        lines += ["## RQ1 - leakage by vantage", "",
                  "| vantage | advantage | MI (bits) |", "|---|---|---|"]
        for v, a in summary["RQ1_advantage_by_vantage"].items():
            lines.append(f"| {v} | {a:.3f} | {summary['RQ1_mi_bits'][v]:.2f} |")
        lines.append("")
    if "RQ2_ablation" in summary:
        lines += ["## RQ2 - which features leak", ""]
        for k, v in summary["RQ2_ablation"].items():
            lines.append(f"- {k}: {v:.3f}")
        lines.append("")
    if "RQ4_defenses" in summary:
        lines += ["## RQ4 - defense frontier (network vantage)", "",
                  "| defense | advantage | bandwidth | latency |", "|---|---|---|---|"]
        for r in summary["RQ4_defenses"]:
            lines.append(f"| {r['defense']} | {r['advantage']:.3f} | "
                         f"+{r['bandwidth_overhead']*100:.0f}% | "
                         f"+{r['latency_overhead']*100:.0f}% |")
        lines.append("")
    if "RQ6_measured" in summary:
        lines += ["## RQ6 - measured harness", "",
                  "| vantage | advantage (mean +/- std) | MI (bits) |", "|---|---|---|"]
        for v, s in summary["RQ6_measured"].items():
            lines.append(f"| {v} | {s['mean']:.3f} +/- {s['std']:.3f} | "
                         f"{s['mi_bits_at_reference_seed']:.2f} |")
        for step, s in summary.get("RQ6_ordering_steps", {}).items():
            lines.append(f"- {step}: holds in {s['holds_in_seeds']}/{s['n_seeds']} seeds "
                         f"(mean gap {s['mean_gap']:.3f}, p={s['p_value_one_sided']:.2g})")
        lines.append("")
    if "RQ6_openworld" in summary:
        lines += ["## Open-world (out-of-taxonomy other)", "",
                  "| monitored base rate | precision | recall | F1 |", "|---|---|---|---|"]
        for r in summary["RQ6_openworld"]:
            lines.append(f"| {r['monitored_base_rate']:.0%} | {r['precision']:.2f} | "
                         f"{r['recall']:.2f} | {r['f1']:.2f} |")
        lines.append("")
    if "RQ7_taxonomy_sweep" in summary:
        lines += ["## RQ7 - taxonomy size", "",
                  "| \\|T\\| | H(C) | advantage | MI (bits) |", "|---|---|---|---|"]
        for r in summary["RQ7_taxonomy_sweep"]:
            lines.append(f"| {r['num_classes']} | {r['entropy_bits']:.2f} | "
                         f"{r['advantage']:.3f} | {r['mi_bits']:.2f} |")
        lines.append("")
    if "RQ8_live" in summary:
        lines += [f"## RQ8 - live captured traffic ({summary['RQ8_live']['capture_mode']})",
                  "",
                  "| seed | network | platform | providers | size-only | timing-only |",
                  "|---|---|---|---|---|---|"]
        for e in summary["RQ8_live"]["per_seed"]:
            lines.append(f"| {e['seed']} | {e['network']:.3f} | {e['platform']:.3f} | "
                         f"{e['providers']:.3f} | {e['size_only']:.3f} | "
                         f"{e['timing_only']:.3f} |")
        lines.append("")
    if "RQ1b_coarse_secret" in summary:
        lines += ["## Granularity - coarse (4-way family) vs fine (12-way class)", "",
                  "| vantage | class Adv | class MI | family Adv | family MI |",
                  "|---|---|---|---|---|"]
        for v, r in summary["RQ1b_coarse_secret"].items():
            lines.append(f"| {v} | {r['class_advantage']:.3f} | {r['class_mi_bits']:.2f} | "
                         f"{r['family_advantage_reestimated']:.3f} | {r['family_mi_bits']:.2f} |")
        lines.append("")
    if "RQ2b_feature_importance" in summary:
        lines += ["## RQ2 - feature importance (corroborates the ablation)", ""]
        for k, v in sorted(summary["RQ2b_feature_importance"].items(),
                           key=lambda kv: -kv[1]):
            lines.append(f"- {k}: {v:.1%}")
        lines.append("")
    if "separability_sweep" in summary:
        sep = summary["separability_sweep"]
        knob = sep["knob"]
        lines += [f"## Separability sweep (knob: {knob})", "",
                  f"| {knob} | advantage | MI (bits) |", "|---|---|---|"]
        for r in sep["rows"]:
            lines.append(f"| {r[knob]:g} | {r['advantage']:.3f} | {r['mi_bits']:.2f} |")
        lines.append(f"\nadvantage monotone along the separability knob: "
                     f"{sep.get('advantage_monotone')}")
        fam = sep.get("family_structure_knob")
        if fam:
            lines += ["", "Companion curve --- the *family-structure* knob "
                      "(signature up-weight), which decreases advantage because the "
                      "signature is shared by three classes:", "",
                      "| signature up-weight | advantage | MI (bits) |", "|---|---|---|"]
            for r in fam["rows"]:
                lines.append(f"| {r['signature_upweight']:g} | {r['advantage']:.3f} | "
                             f"{r['mi_bits']:.2f} |")
        lines.append("")
    if "generator_draw_variance" in summary:
        lines += ["## Variance across generator draws", "",
                  "| vantage | advantage (mean +/- std) |", "|---|---|"]
        for v, s_ in summary["generator_draw_variance"].items():
            lines.append(f"| {v} | {s_['mean']:.3f} +/- {s_['std']:.3f} |")
        lines.append("")
    if "prior_sensitivity" in summary:
        lines += ["## Prior sensitivity (uniform vs Zipf)", "",
                  "| prior | " + " | ".join(summary["prior_sensitivity"]["uniform"]) + " |",
                  "|---|" + "---|" * len(summary["prior_sensitivity"]["uniform"])]
        for prior_name, vals in summary["prior_sensitivity"].items():
            lines.append(f"| {prior_name} | "
                         + " | ".join(f"{a:.3f}" for a in vals.values()) + " |")
        lines.append("")
    if "dp_sweep" in summary:
        lines += ["## DP shaping: epsilon -> advantage", "",
                  "| epsilon | adjacency | advantage | ceiling (d=1) | bandwidth |",
                  "|---|---|---|---|---|"]
        for r in summary["dp_sweep"]["rows"]:
            lines.append(f"| {r['epsilon']:g} | {r['adjacency']} | {r['advantage']:.3f} | "
                         f"{r['ceiling_distance_1']:.3f} | "
                         f"+{r['bandwidth_overhead']*100:.0f}% |")
        lines.append(f"\nCeiling violations: {summary['dp_sweep']['ceiling_violations']}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="outputs/scale_study")
    ap.add_argument("--generator-config", default="configs/default.yaml")
    ap.add_argument("--measured-config", default="configs/measured.yaml")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of stages to run")
    ap.add_argument("--skip", default=None, help="comma-separated stages to skip")
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the measured harness")
    ap.add_argument("--live-per-class", type=int, default=70)
    ap.add_argument("--openworld-total", type=int, default=24000)
    ap.add_argument("--validation-per-class", type=int, default=300)
    ap.add_argument("--draws", type=int, default=8,
                    help="independent generator profile draws for the variance check")
    ap.add_argument("--tax-sizes", type=int, nargs="+", default=[8, 12, 16, 24])
    ap.add_argument("--no-tls", action="store_true",
                    help="run the live capture over plaintext TCP (records the mode)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny sizes for an end-to-end check (NOT a paper-scale run)")
    args = ap.parse_args()

    gen_cfg = load_config(args.generator_config)
    meas_cfg = load_config(args.measured_config)
    if args.quick:
        gen_cfg = replace(gen_cfg, traces_per_class=25, bootstrap=20)
        meas_cfg = replace(meas_cfg, traces_per_class=10, bootstrap=20)
        args.seeds = 2
        args.live_per_class = 8
        args.openworld_total = 1200
        args.validation_per_class = 60
        args.tax_sizes = [8, 12]
        args.draws = 2

    seed_everything(gen_cfg.seed)
    out_dir = Path(args.output)
    tax = reference_taxonomy()
    plan = _stage_plan(gen_cfg, meas_cfg, args)

    stages = list(plan)
    if args.only:
        stages = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in stages if s not in plan]
        if unknown:
            raise SystemExit(f"Unknown stage(s) {unknown}; choose from {list(plan)}.")
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",")}
        stages = [s for s in stages if s not in skip]

    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    prov = capture_provenance(
        gen_cfg.seed, tax.content_hash(), tax.name,
        {"generator": gen_cfg.to_dict(), "measured": meas_cfg.to_dict(),
         "args": vars(args)},
        note=("QUICK end-to-end check, NOT a paper-scale run" if args.quick
              else "paper-scale study"),
    )

    results: dict[str, Any] = {}
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    for stage in stages:
        logger.info("=== stage %s ===", stage)
        t0 = time.time()
        try:
            result = plan[stage]()
        except Exception as exc:  # noqa: BLE001 - a failed stage must not lose the rest
            failures[stage] = f"{type(exc).__name__}: {exc}"
            logger.exception("stage %s FAILED", stage)
            continue
        timings[stage] = time.time() - t0
        results[stage] = result
        dump_json(
            {"provenance": prov.to_dict(), "result": result,
             "status": ("quick_check_not_paper_scale" if args.quick
                        else "real_computation_at_paper_scale"),
             "elapsed_s": timings[stage]},
            out_dir / f"{stage}.json",
        )
        logger.info("stage %s done in %.1f s", stage, timings[stage])

    elapsed = time.time() - started
    meta = {
        "platform": platform.platform(),
        "started_utc": started_utc,
        "elapsed_s": elapsed,
        "seed": gen_cfg.seed,
        "taxonomy_name": tax.name,
        "taxonomy_hash": tax.content_hash(),
        "generator_traces_per_class": gen_cfg.traces_per_class,
        "measured_traces_per_class": meas_cfg.traces_per_class,
        "live_traces_per_class": args.live_per_class,
        "seeds": args.seeds,
        "quick": args.quick,
        "stage_timings_s": timings,
        "failed_stages": failures,
    }
    summary = _summarize(results)
    dump_json({"provenance": prov.to_dict(), "meta": meta, "summary": summary,
               "status": ("quick_check_not_paper_scale" if args.quick
                          else "real_computation_at_paper_scale")},
              out_dir / "summary.json")
    (out_dir / "report.md").write_text(_report_md(summary, meta), encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir), "stages_run": list(results),
                      "failed": failures, "elapsed_s": round(elapsed, 1)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
