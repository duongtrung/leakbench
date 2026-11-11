"""Logging for experiment scripts (paper code-quality rule: logging, not print)."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "leakbench", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stderr."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        root = logging.getLogger("leakbench")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger
