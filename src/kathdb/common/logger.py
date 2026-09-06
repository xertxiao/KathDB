"""Project-wide logging: levels ``info``, ``interact``, ``warning``, ``error``."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, cast

__all__ = ["get_logger", "logger", "configure_logger", "KathDBLogger"]

_LOGGER_NAME = "kathdb"
_LOG_LEVEL_ENV = "KATHDB_LOG_LEVEL"
_DEFAULT_LEVEL = logging.INFO
_INTERACT_LEVEL = logging.INFO + 5
_SUPPORTED_LEVELS = {
    "info": logging.INFO,
    "interact": _INTERACT_LEVEL,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_RESET_COLOR = "\033[0m"
_COLOR_MAP = {
    _INTERACT_LEVEL: "\033[92m",  # green
    logging.WARNING: "\033[93m",  # yellow
    logging.ERROR: "\033[91m",  # red
}


class KathDBFormatter(logging.Formatter):
    """Formatter: bare, colored message for ``interact`` records."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        if record.levelno == _INTERACT_LEVEL:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{self.formatException(record.exc_info)}"
            color = _COLOR_MAP.get(record.levelno, "")
            return f"{color}{message}{_RESET_COLOR}" if color else message
        formatted = super().format(record)
        color = _COLOR_MAP.get(record.levelno)
        return f"{color}{formatted}{_RESET_COLOR}" if color else formatted


class KathDBLogger(logging.Logger):
    """Logger with a dedicated ``interact`` level for user dialogues."""

    def interact(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(_INTERACT_LEVEL):
            self._log(_INTERACT_LEVEL, msg, args, **kwargs)


logging.addLevelName(_INTERACT_LEVEL, "INTERACT")
logging.setLoggerClass(KathDBLogger)


def _coerce_level(level: Optional[str | int]) -> int:
    """Convert an incoming level to one of the supported logging levels."""
    if level is None:
        return _DEFAULT_LEVEL
    if isinstance(level, int):
        return level
    coerced = level.strip().lower()
    if coerced.isdigit():
        return int(coerced)
    return _SUPPORTED_LEVELS.get(coerced, _DEFAULT_LEVEL)


def configure_logger(*, level: Optional[str | int] = None) -> KathDBLogger:
    """Initialize (once) and return the KathDB logger; ``level`` defaults to
    ``KATHDB_LOG_LEVEL`` or INFO."""
    resolved_level = _coerce_level(level or os.getenv(_LOG_LEVEL_ENV))
    root_logger = cast(KathDBLogger, logging.getLogger(_LOGGER_NAME))
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            KathDBFormatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
        root_logger.propagate = False
    root_logger.setLevel(resolved_level)
    return root_logger


def get_logger(
    name: Optional[str] = None, *, level: Optional[str | int] = None
) -> KathDBLogger:
    """Return the shared KathDB logger or a child logger bound to ``name``."""
    parent = configure_logger(level=level)
    if not name:
        return parent
    return cast(KathDBLogger, parent.getChild(name))


logger = get_logger()
