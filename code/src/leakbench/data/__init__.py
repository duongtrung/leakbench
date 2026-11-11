"""Trace sources and the shared record schema.

Sources (all behind the same :class:`~leakbench.data.schemas.Trace` schema):
- :class:`~leakbench.data.generator.CalibratedGenerator` (RQ1-RQ5, RQ7)
- :class:`~leakbench.data.measured.MeasuredHarness` (RQ6)
- :class:`~leakbench.data.live_capture.LiveCaptureHarness` (RQ8)
- :mod:`leakbench.data.adapters` (framework adapters; stubs + dummy)
"""

from leakbench.data.generator import CalibratedGenerator
from leakbench.data.measured import MeasuredHarness
from leakbench.data.schemas import ToolCall, Trace, TraceDataset

__all__ = [
    "CalibratedGenerator",
    "MeasuredHarness",
    "ToolCall",
    "Trace",
    "TraceDataset",
]
