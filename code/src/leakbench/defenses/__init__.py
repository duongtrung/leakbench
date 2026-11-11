"""Trace-level defense transforms --- Contribution C4.

Defenses act on the trace *before* observation (paper §Design Stage 3), exactly as
a real padding/cover layer would act on the wire, so the same estimator scores a
system with and without protection on one axis.

Registry (paper Table "defense", ported from website fingerprinting):

| Defense       | WF analogue     | targets  |
|---------------|-----------------|----------|
| none          | ---             | ---      |
| pad-512       | cell padding    | size     |
| shuffle       | reordering      | order    |
| constant-rate | Tamaraw timing  | timing   |
| cover         | FRONT           | counts   |
| pad-max       | const. size     | size     |
| full          | Tamaraw+cover   | all      |
| gen-shaping   | Surakav         | all (learned decoy schedule) |
| dp-shaping    | NetShaper       | all (DP release schedule)    |
| mixing        | mix network     | all (shared-gateway anonymity set) |

The last two go beyond the paper's shipped artifact: the paper describes generative
shaping as future work and the DP/advantage connection as its key open problem, and
LeakBench's evaluation path accepts any trace->trace defense. See
``defenses/generative.py`` and ``defenses/dp_shaping.py`` for their exact scope.
"""

from leakbench.defenses.base import Defense, DefenseResult, get_defense, list_defenses
from leakbench.defenses.dp_shaping import DPShapingDefense, dp_advantage_ceiling
from leakbench.defenses.generative import GenerativeShapingDefense
from leakbench.defenses.mixing import MixingDefense
from leakbench.defenses.transforms import (
    ConstantRateDefense,
    CoverCallDefense,
    FullDefense,
    NoneDefense,
    Pad512Defense,
    PadMaxDefense,
    ShuffleDefense,
)

__all__ = [
    "Defense",
    "DefenseResult",
    "get_defense",
    "list_defenses",
    "NoneDefense",
    "Pad512Defense",
    "ShuffleDefense",
    "ConstantRateDefense",
    "CoverCallDefense",
    "PadMaxDefense",
    "FullDefense",
    "GenerativeShapingDefense",
    "MixingDefense",
    "DPShapingDefense",
    "dp_advantage_ceiling",
]
