"""LeakBench: a reproducible benchmark for task leakage from LLM agent tool-call
metadata.

Benchmark harness for *"What the Tools See: Measuring Task Leakage from LLM Agent
Tool-Call Metadata at Three Web Vantages"*.

The package is organized around the paper's five-stage harness:

    taxonomy  ->  trace source  ->  observation (+ defense)  ->  estimator bank  ->  report

Sub-packages
------------
- ``leakbench.data``        : trace schema, calibrated generator, measured harness, wire models, adapters
- ``leakbench.observables`` : vantage observation maps and prior-attack projections
- ``leakbench.defenses``    : trace-level defense transforms (padding, cover, shuffle, ...)
- ``leakbench.estimators``  : attacker bank + the standardized leakage-estimation algorithm
- ``leakbench.metrics``     : information-theoretic quantities (MI, g-leakage), open-world metrics
- ``leakbench.validation``  : the analytic ground-truth channel used to validate the estimator
- ``leakbench.experiments`` : RQ runners and plotting
- ``leakbench.utils``       : seeding, logging, IO, provenance
"""

from leakbench._version import __version__

__all__ = ["__version__"]
