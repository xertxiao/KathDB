"""Lazy function library: ``from kathdb.fn import <name>`` resolves the folder
``<name>/scripts`` under ``pre_built_fn/`` or ``generated_fn/``."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_PREBUILT_FN_DIR: Path = _PKG_ROOT / "pre_built_fn"
# Must match the directory FunctionManager saves to (same env var, set by KathDB
# before the worker spawns), or a selected function is not importable in the worker.
_env_generated_fn_dir = os.environ.get("KATHDB_GENERATED_FN_DIR")
_GENERATED_FN_DIR: Path = (
    Path(_env_generated_fn_dir).expanduser().resolve()
    if _env_generated_fn_dir
    else _PKG_ROOT / "generated_fn"
)


def _discover_function_names() -> set[str]:
    """Return the set of valid function sub-folder names across both dirs."""
    names: set[str] = set()
    for base in (_PREBUILT_FN_DIR, _GENERATED_FN_DIR):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if (
                child.is_dir()
                and not child.name.startswith("_")
                and (child / "scripts").is_dir()
            ):
                names.add(child.name)
    return names


def __getattr__(name: str):
    """Lazily import a function from its ``scripts`` sub-package."""
    if name.startswith("_"):
        raise AttributeError(name)
    if name not in _discover_function_names():
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Try built-in dir first, then generated dir.
    for base in (_PREBUILT_FN_DIR, _GENERATED_FN_DIR):
        scripts_init = base / name / "scripts" / "__init__.py"
        if scripts_init.is_file():
            mod_name = f"kathdb.fn.{name}.scripts"
            spec = importlib.util.spec_from_file_location(
                mod_name,
                scripts_init,
                submodule_search_locations=[str(scripts_init.parent)],
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                obj = getattr(mod, name, None)
                if obj is not None:
                    globals()[name] = obj
                    return obj

    raise ImportError(f"{__name__}.{name}.scripts does not export {name!r}")


def __dir__():
    return list(_discover_function_names())
