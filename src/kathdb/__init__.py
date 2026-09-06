"""Public API surface for the KathDB package (heavy imports are deferred to first use)."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "KathDB", "KathDBConfig"]


def __getattr__(name: str):
    if name == "KathDB":
        from .kathdb import KathDB

        return KathDB
    if name == "KathDBConfig":
        from .config import KathDBConfig

        return KathDBConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
