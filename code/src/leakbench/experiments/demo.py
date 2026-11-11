"""Synthetic smoke-test demo orchestrator.

Runs a small, fast end-to-end pass of the LeakBench harness on the calibrated
generator and writes result files + plots, EVERY ONE labeled

    "synthetic smoke-test output, not a paper result".

This is the reproducible quick-start. It exercises: taxonomy -> generator ->
observation maps (3 vantages) -> estimator bank -> per-vantage leakage profile;
feature ablation (RQ2); disparate-vulnerability subgroups (RQ3); defense frontier
(RQ4); and estimator validation vs the analytic channel (RQ5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leakbench.experiments.config import ExperimentConfig
from leakbench.experiments.plotting import (
    plot_ablation,
    plot_defenses,
    plot_validation,
    plot_vantage,
)
from leakbench.experiments.runners import (
    run_ablation,
    run_defenses,
    run_subgroup,
    run_validation,
    run_vantage,
)
from leakbench.taxonomy import reference_taxonomy
from leakbench.utils.io import dump_json
from leakbench.utils.logging import get_logger
from leakbench.utils.provenance import capture_provenance
from leakbench.utils.seeding import seed_everything

logger = get_logger("leakbench.demo")

SYNTHETIC_LABEL = "synthetic smoke-test output, not a paper result"


def run_synthetic_demo(cfg: ExperimentConfig | None = None) -> Path:
    """Run the demo and return the output directory."""
    cfg = cfg or ExperimentConfig(
        name="synthetic_demo", traces_per_class=40, bootstrap=50,
        output_dir="outputs/synthetic_demo",
    )
    seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    tax = reference_taxonomy()
    prov = capture_provenance(cfg.seed, tax.content_hash(), tax.name, cfg.to_dict(),
                              note=SYNTHETIC_LABEL)

    def _write(stage: str, result: dict[str, Any]) -> None:
        payload = {
            "provenance": prov.to_dict(),
            "result": result,
            "status": SYNTHETIC_LABEL,
            "warning": "These numbers are from the calibrated generator, not a "
                       "deployed system. Do not cite as paper results.",
        }
        dump_json(payload, out_dir / f"{stage}.json")

    logger.info("Demo: %d classes, %d traces/class, seed %d",
                cfg.num_classes, cfg.traces_per_class, cfg.seed)

    vantage = run_vantage(cfg)
    _write("vantage", vantage)
    plot_vantage(vantage, out_dir / "plots" / "vantage.png", synthetic=True)

    ablation = run_ablation(cfg)
    _write("ablation", ablation)
    plot_ablation(ablation, out_dir / "plots" / "ablation.png", synthetic=True)

    subgroup = run_subgroup(cfg)
    _write("subgroup", subgroup)

    defenses = run_defenses(cfg)
    _write("defenses", defenses)
    plot_defenses(defenses, out_dir / "plots" / "defenses.png", synthetic=True)

    validation = run_validation(cfg, n_per_class=150)
    _write("validation", validation)
    plot_validation(validation, out_dir / "plots" / "validation.png", synthetic=True)

    # A compact human-readable summary.
    summary = {
        "label": SYNTHETIC_LABEL,
        "entropy_bits": vantage["entropy_bits"],
        "advantage_by_vantage": vantage["advantage_by_vantage"],
        "refinement_warnings": vantage["refinement_warnings"],
        "ablation_advantage": {k: v["advantage"] for k, v in ablation["ablations"].items()},
        "subgroup_by_family": subgroup["by_family"],
        "subgroup_by_length": subgroup["by_length_band"],
        "defense_frontier": [
            {"defense": r["defense"], "advantage": r["advantage"],
             "bandwidth_overhead": r["bandwidth_overhead"]}
            for r in defenses["rows"]
        ],
        "estimator_max_mi_error_bits": validation["max_mi_abs_error"],
    }
    dump_json(summary, out_dir / "summary.json")
    logger.info("Demo outputs written to %s", out_dir)
    return out_dir
