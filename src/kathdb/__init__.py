"""Public API surface for the KathDB package."""

from .config import KathDBConfig
from .kathdb import KathDB

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "KathDB",
    "KathDBConfig",
]
