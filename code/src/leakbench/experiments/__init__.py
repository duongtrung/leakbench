"""Experiment runners (RQ1-RQ8) and plotting.

Each runner writes its result to a JSON/CSV file under an output directory; plots
and tables are generated *from* those files, never hard-coded (result-integrity
policy). Every result file carries run provenance and a status flag.
"""

from leakbench.experiments.config import ExperimentConfig, load_config
from leakbench.experiments.runners import (
    build_dataset,
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

__all__ = [
    "ExperimentConfig",
    "load_config",
    "build_dataset",
    "run_vantage",
    "run_ablation",
    "run_subgroup",
    "run_defenses",
    "run_validation",
    "run_measured",
    "run_measured_multiseed",
    "run_wire_sweep",
    "run_attacker_robustness",
    "run_openworld",
    "run_projections",
    "run_dp_sweep",
    "run_learning_curve",
    "run_coarse_secret",
    "run_feature_importance",
    "run_separability_sweep",
    "run_scalability",
    "run_generator_variance",
    "run_prior_sensitivity",
    "run_estimator_agreement",
    "run_overlap_sweep",
    "run_bank_comparison",
    "run_sequence_attacker",
    "run_mixing_sweep",
]
