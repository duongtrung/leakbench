"""Run provenance capture (paper §Design Stage 5: reports keyed by
(taxonomy hash, framework version, defense params, seed) so they are reproducible
and diff-able)."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from leakbench._version import __version__


@dataclass
class RunProvenance:
    """Metadata stamped onto every result file so a number is never asserted
    without a record of how it was produced."""

    leakbench_version: str
    python_version: str
    platform: str
    seed: int
    taxonomy_hash: str
    taxonomy_name: str
    config: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_provenance(
    seed: int,
    taxonomy_hash: str,
    taxonomy_name: str,
    config: dict[str, Any] | None = None,
    note: str = "",
) -> RunProvenance:
    return RunProvenance(
        leakbench_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        seed=seed,
        taxonomy_hash=taxonomy_hash,
        taxonomy_name=taxonomy_name,
        config=config or {},
        note=note,
    )
