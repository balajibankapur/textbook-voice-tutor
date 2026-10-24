"""Logging setup. Import `configure_logging` once at app entry."""
from __future__ import annotations

import logging
import sys

from config.settings import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def configure_logging(level: str | None = None) -> None:
    log_level = (level or settings.log_level).upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(log_level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(log_level)
