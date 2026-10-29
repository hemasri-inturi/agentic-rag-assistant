"""Structured-ish logging setup (stdlib only, no extra dependency)."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single, readable log format for the whole service."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (call ``configure_logging`` once at startup)."""
    return logging.getLogger(name)
