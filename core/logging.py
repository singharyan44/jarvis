#!/usr/bin/env python3
"""
Logging configuration for JARVIS.

Provides a pre-configured logger instance and helper to reconfigure
log level/format at runtime (e.g., for testing).
"""

from __future__ import annotations

import logging
import sys


# Default format matches the original script
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LEVEL = logging.INFO


# Module-level logger instance — imported by other modules
log = logging.getLogger("jarvis")


def configure_logging(
    level: int = _DEFAULT_LEVEL,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    stream: sys._io.TextIOWrapper | None = None,
) -> None:
    """
    Configure the root logger and the 'jarvis' logger.

    Call this once at application startup (or in tests to adjust verbosity).
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]  # replace any existing handlers

    # Ensure our module logger propagates to root
    log.setLevel(level)
    log.propagate = True


# Initial configuration on import
configure_logging()