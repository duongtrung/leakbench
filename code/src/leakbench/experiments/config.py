"""Experiment configuration (YAML-backed, with typed defaults).

A config fully determines a run: taxonomy, trace source, sizes, estimator bank,
bootstrap, seed, and output directory. Configs are captured into every result file's
provenance so a run is reproducible and diff-able.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from leakbench.utils.seeding import REFERENCE_SEED


@dataclass
class ExperimentConfig:
    """Typed experiment configuration."""

    name: str = "default"
    source: str = "generator"           # generator | measured | live
    num_classes: int = 12
    traces_per_class: int = 600
    seed: int = REFERENCE_SEED
    signature_upweight: float = 0.9     # generator separability knob
    overlap: float = 0.5                # measured-harness class overlap
    bank: tuple[str, ...] = ("random_forest", "knn")
    bootstrap: int = 200
    test_size: float = 0.4
    vantages: tuple[str, ...] = ("network", "platform", "providers")
    defenses: tuple[str, ...] = ("none", "pad-512", "shuffle", "constant-rate",
                                 "cover", "pad-max", "full")
    prior: str = "uniform"              # uniform | zipf
    output_dir: str = "outputs/experiment"
    # data.root for real datasets placed manually on EC2 (never auto-downloaded).
    data_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bank"] = list(self.bank)
        d["vantages"] = list(self.vantages)
        d["defenses"] = list(self.defenses)
        return d


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Unknown keys are ignored with the typed defaults filled in, so a partial config
    is valid.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    # Flatten a nested 'data:' block if present (data.root).
    if isinstance(raw.get("data"), dict):
        raw["data_root"] = raw["data"].get("root", raw.get("data_root"))
        raw.pop("data")
    valid = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in valid}
    for key in ("bank", "vantages", "defenses"):
        if key in filtered and isinstance(filtered[key], list):
            filtered[key] = tuple(filtered[key])
    return ExperimentConfig(**filtered)
