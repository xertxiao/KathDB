"""Function library: discovery, loading, rendering, saving and eviction."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)

__all__ = ["FunctionManager", "FunctionRecord"]


@dataclass
class FunctionRecord:
    """In-memory metadata for a saved function, used for reusability tracking."""

    name: str
    atom_count: int
    member_atoms: tuple[str, ...]
    usage_count: int = 0


class FunctionManager:
    """Manages the built-in (``pre_built_fn/``) and generated (``generated_fn/``)
    function directories. Eviction (``max_functions``, least-used first) applies
    to the generated directory only.
    """

    def __init__(
        self,
        builtin_fn_dir: Path | None = None,
        generated_fn_dir: Path | None = None,
        max_functions: int = 10,
        sources: tuple[str, ...] = ("builtin", "generated"),
    ) -> None:
        pkg_root = Path(__file__).resolve().parent.parent
        self._builtin_fn_dir = builtin_fn_dir or pkg_root / "pre_built_fn"
        # KATHDB_GENERATED_FN_DIR is also read by ``kathdb.fn`` inside the worker;
        # both sides must resolve the same directory.
        self._generated_fn_dir = Path(
            generated_fn_dir
            or os.environ.get("KATHDB_GENERATED_FN_DIR")
            or pkg_root / "generated_fn"
        )
        self._generated_fn_dir.mkdir(parents=True, exist_ok=True)
        self._max_functions = max_functions
        self._sources = tuple(sources)
        self._usage_path = self._generated_fn_dir / "_usage.json"
        self._records: dict[str, FunctionRecord] = {}
        self._reuse_opportunity: bool = False
        self._load_records()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_functions(
        self,
        sources: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Scan the selected directories (``"builtin"`` / ``"generated"``; ``None`` =
        configured sources) and return ``{name: {"fn_md", "df_params"}}``.
        Built-in wins on name collisions; the filesystem is re-scanned every call.
        """
        if sources is None:
            sources = self._sources
        dirs: list[Path] = []
        if "builtin" in sources:
            dirs.append(self._builtin_fn_dir)
        if "generated" in sources:
            dirs.append(self._generated_fn_dir)

        catalog: dict[str, dict[str, Any]] = {}
        for fn_dir in dirs:
            if not fn_dir.is_dir():
                continue
            for child in sorted(fn_dir.iterdir()):
                if not child.is_dir() or child.name.startswith("_"):
                    continue
                if child.name in catalog:
                    continue  # built-in takes precedence
                fn_md = self._read_file(child / "fn.md")
                if not fn_md:
                    continue
                df_params = self._resolve_df_params(child.name)
                if df_params is None:
                    continue
                catalog[child.name] = {
                    "fn_md": fn_md,
                    "df_params": df_params,
                }
        return catalog

    def _resolve_df_params(self, name: str) -> tuple[str, ...] | None:
        """DataFrame-param names from the CONTRACT + typed signature (AST, no
        import); None (no CONTRACT) excludes the function."""
        from .fn_contract import derive_df_params

        parsed = self._parse_contract(name)
        return derive_df_params(parsed) if parsed is not None else None

    # ------------------------------------------------------------------
    # Usage tracking & eviction
    # ------------------------------------------------------------------

    def _load_usage(self) -> dict[str, int]:
        """Load per-function usage counts from ``_usage.json``."""
        if self._usage_path.is_file():
            try:
                data = json.loads(self._usage_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if isinstance(v, int)}
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt _usage.json; resetting counts.")
        return {}

    def _save_usage(self, usage: dict[str, int]) -> None:
        """Persist usage counts to ``_usage.json``."""
        try:
            self._usage_path.write_text(
                json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError:
            logger.warning("Failed to write _usage.json", exc_info=True)

    def _records_path(self) -> Path:
        return self._generated_fn_dir / "_records.json"

    def _load_records(self) -> None:
        """Load persisted FunctionRecords from ``_records.json``."""
        path = self._records_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for name, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                self._records[name] = FunctionRecord(
                    name=name,
                    atom_count=entry.get("atom_count", 0),
                    member_atoms=tuple(entry.get("member_atoms", ())),
                    usage_count=entry.get("usage_count", 0),
                )
            if any(r.usage_count > 0 for r in self._records.values()):
                self._reuse_opportunity = True
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt _records.json; starting fresh.", exc_info=True)

    def _save_records(self) -> None:
        """Persist FunctionRecords to disk."""
        data = {
            name: {
                "atom_count": r.atom_count,
                "member_atoms": list(r.member_atoms),
                "usage_count": r.usage_count,
            }
            for name, r in self._records.items()
        }
        try:
            self._records_path().write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError:
            logger.warning("Failed to write _records.json", exc_info=True)

    def record_usage(self, name: str) -> None:
        """Increment the usage counter for *name*."""
        usage = self._load_usage()
        usage[name] = usage.get(name, 0) + 1
        self._save_usage(usage)
        if name in self._records:
            self._records[name].usage_count += 1
            self._reuse_opportunity = True
            self._save_records()

    def record_save(
        self,
        name: str,
        atom_count: int,
        member_atoms: tuple[str, ...],
    ) -> None:
        """Record a newly saved function in the in-memory registry."""
        self._records[name] = FunctionRecord(
            name=name,
            atom_count=atom_count,
            member_atoms=member_atoms,
        )
        self._save_records()

    def reconcile_records(self) -> None:
        """Drop records for functions whose directories no longer exist."""
        if not self._generated_fn_dir.is_dir():
            return
        existing = {
            child.name
            for child in self._generated_fn_dir.iterdir()
            if child.is_dir() and not child.name.startswith("_")
        }
        stale = [name for name in self._records if name not in existing]
        if stale:
            for name in stale:
                del self._records[name]
            self._save_records()

    def get_reuse_rate(self) -> float | None:
        """Fraction of saved functions reused; None before any reuse opportunity."""
        if not self._records or not self._reuse_opportunity:
            return None
        n_reused = sum(1 for r in self._records.values() if r.usage_count > 0)
        return n_reused / len(self._records)

    def remove_function(self, name: str) -> bool:
        """Delete a generated function (built-ins are never removed); True if it existed."""
        fn_dir = self._generated_fn_dir / name
        if not fn_dir.is_dir():
            return False
        shutil.rmtree(fn_dir)
        usage = self._load_usage()
        if usage.pop(name, None) is not None:
            self._save_usage(usage)
        if self._records.pop(name, None) is not None:
            self._save_records()
        return True

    def clear_generated(self) -> list[str]:
        """Delete every generated function; returns the removed names."""
        names = sorted(
            p.name for p in self._generated_fn_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
        return [n for n in names if self.remove_function(n)]

    def evict_least_used(self) -> list[str]:
        """Evict least-used generated functions down to ``max_functions``; returns their names."""
        if not self._generated_fn_dir.is_dir():
            return []

        usage = self._load_usage()

        gen_fns: list[str] = [
            child.name
            for child in sorted(self._generated_fn_dir.iterdir())
            if child.is_dir() and not child.name.startswith("_")
        ]

        if len(gen_fns) <= self._max_functions:
            return []

        gen_fns.sort(key=lambda n: usage.get(n, 0))

        to_evict = gen_fns[: len(gen_fns) - self._max_functions]
        evicted: list[str] = []
        for name in to_evict:
            fn_dir = self._generated_fn_dir / name
            try:
                shutil.rmtree(fn_dir)
                usage.pop(name, None)
                evicted.append(name)
                logger.info("Evicted least-used generated function '%s'.", name)
            except OSError:
                logger.warning("Failed to evict function '%s'.", name, exc_info=True)

        if evicted:
            self._save_usage(usage)

        return evicted

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _resolve_fn_dir(self, name: str) -> Path | None:
        """Return the directory for *name*, checking built-in first."""
        for base in (self._builtin_fn_dir, self._generated_fn_dir):
            candidate = base / name
            if candidate.is_dir():
                return candidate
        return None

    def _import_module_from_file(self, module_name: str, file_path: Path):
        """Import a module from an arbitrary file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    def load_function(self, name: str):
        """Import and return the callable from ``<dir>/{name}/scripts/fn.py``."""
        fn_dir = self._resolve_fn_dir(name)
        if fn_dir is None:
            raise ImportError(f"Function '{name}' not found in any function directory")
        fn_path = fn_dir / "scripts" / "fn.py"
        mod = self._import_module_from_file(f"kathdb.fn.{name}.scripts.fn", fn_path)
        candidates = [
            attr
            for attr in dir(mod)
            if not attr.startswith("_") and callable(getattr(mod, attr))
        ]
        if not candidates:
            raise ImportError(f"No callable found in {fn_path}")
        return getattr(mod, candidates[0])

    def read_function_file(self, fn_name: str, relative_path: str) -> str | None:
        """Read a file from the function folder, returning *None* if missing."""
        fn_dir = self._resolve_fn_dir(fn_name)
        if fn_dir is None:
            return None
        path = fn_dir / relative_path
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    def render_functions_summary(
        self,
        sources: tuple[str, ...] | None = None,
    ) -> str:
        """Render the description preamble + ``## Output`` of every discovered function."""
        catalog = self.discover_functions(sources=sources)
        if not catalog:
            return ""
        parts: list[str] = ["## Available Functions"]
        for i, (name, entry) in enumerate(catalog.items(), 1):
            parts.append(f"\n{'=' * 64}")
            parts.append(f"{i}) {name}")
            parts.append("=" * 64)
            parts.append(self._extract_description_and_output(entry["fn_md"]))
        return "\n".join(parts)

    def render_functions_full_docs(self, names: list[str] | None = None) -> str:
        """Render full ``fn.md`` of ``names`` (or all) under a ``## Relevant Functions`` header."""
        catalog = self.discover_functions()

        if names is not None:
            entries: list[tuple[str, dict[str, Any]]] = []
            for n in names:
                entry = catalog.get(n)
                if entry is not None:
                    entries.append((n, entry))
                else:
                    fn_dir = self._resolve_fn_dir(n)
                    if fn_dir is not None:
                        fn_md = self._read_file(fn_dir / "fn.md")
                        if fn_md:
                            entries.append((n, {"fn_md": fn_md}))
        else:
            entries = list(catalog.items())

        if not entries:
            return ""

        parts: list[str] = [
            "## Relevant Functions",
            "These are KathDB functions available for this node. "
            "You can use them directly or add additional logic around them, "
            "or write your own code — whichever fits the operation.",
            "DEFAULT TO WRITING YOUR OWN CODE. Use a listed function ONLY if, after "
            "reading its full doc — especially `## Behavior`, the worked "
            "input->output trace — you are CERTAIN it produces exactly the logic "
            "this node needs. A function that is almost right but differs in one "
            "behavioral detail (pairs items once vs every combination; counts vs "
            "excludes a catch-all label; caps vs returns all rows) yields a wrong "
            "answer that looks plausible — worse than fresh code. If you are not "
            "sure, do not use it; reusing only a PART of it (e.g. its per-row "
            "classification loop) while writing your own logic around it is often "
            "the right middle ground.",
            "To use one, import inside your function body (with your other imports):",
            "  from kathdb.fn import <function_name>",
            "If the function has a `model` parameter, ALWAYS pass it explicitly "
            "with the same model string your own code would use — never rely on "
            "its default (a stale default can point at a provider this "
            "deployment cannot reach, and per-row error handling then turns "
            "every call into a silent empty result).",
            "",
            "Some functions wrap remote LLM calls; others are relational helpers. "
            "You DO NOT need to set API keys—KathDB configures them when applicable.",
            "",
            "Functions referenced here (each followed by full fn.md):",
        ]
        for i, (name, _) in enumerate(entries, 1):
            parts.append(f"  {i}) {name}")
        parts.append("")

        for i, (name, entry) in enumerate(entries, 1):
            parts.append(f"{'=' * 64}")
            parts.append(f"{i}) {name}  (import: from kathdb.fn import {name})")
            parts.append("=" * 64)
            parts.append(entry["fn_md"].strip())
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Writing (saving new functions)
    # ------------------------------------------------------------------

    def save_function(
        self,
        fn_spec: dict,
        code: str,
        *,
        fn_md: str | None = None,
        spec_py: str | None = None,
    ) -> None:
        """Save ``code`` as ``generated_fn/<name>/scripts/fn.py``; ``fn.md`` is
        rendered from its CONTRACT (``fn_md`` is used only when there is none).
        Returns the saved name (suffixed on collision) or None."""
        from .fn_contract import parse_fn_source, render_fn_md

        name = fn_spec.get("name", "").strip().replace(" ", "_").lower()
        if not name:
            logger.warning("Cannot save function: empty name in fn_spec")
            return None

        # On a name collision add a numeric suffix and rename every occurrence in
        # the code so the dir name, the def and the reuse-import stay consistent.
        if self._resolve_fn_dir(name) is not None:
            base, i = name, 1
            while self._resolve_fn_dir(f"{base}_{i}") is not None:
                i += 1
            new_name = f"{base}_{i}"
            code = re.sub(rf"\b{re.escape(base)}\b", new_name, code)
            logger.info(
                "Function '%s' exists; saving as '%s' (no dedup).", base, new_name
            )
            name = new_name

        final_md: str | None = None
        if "CONTRACT" in code:
            try:
                final_md = render_fn_md(parse_fn_source(code, name))
            except ValueError as exc:
                logger.warning(
                    "save_function('%s'): CONTRACT parse failed (%s); not saving",
                    name,
                    exc,
                )
                return None
        else:
            final_md = fn_md
        if not final_md:
            logger.warning(
                "save_function('%s'): no CONTRACT in code and no fn_md given; not saving",
                name,
            )
            return None

        fn_dir = self._generated_fn_dir / name
        try:
            scripts_dir = fn_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            (fn_dir / "__init__.py").write_text("")
            (scripts_dir / "__init__.py").write_text(
                f"from .fn import {name}\n\n__all__ = [{name!r}]\n"
            )
            (scripts_dir / "fn.py").write_text(code)
            (fn_dir / "fn.md").write_text(final_md)

            logger.info("Saved new generated function '%s' at %s", name, fn_dir)
            return name
        except Exception:
            logger.warning("Failed to save function '%s'", name, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Signature extraction & spec building
    # ------------------------------------------------------------------

    @staticmethod
    def extract_function_signature(code: str, fn_name: str) -> list[dict]:
        """Return ``[{"name", "default"}]`` for every parameter of ``fn_name`` (AST)."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                args = node.args
                num_args = len(args.args)
                num_defaults = len(args.defaults)
                params: list[dict] = []
                for i, arg in enumerate(args.args):
                    default_index = i - (num_args - num_defaults)
                    if default_index >= 0:
                        default_node = args.defaults[default_index]
                        default = ast.unparse(default_node)
                    else:
                        default = None
                    params.append({"name": arg.arg, "default": default})
                return params
        return []

    # ------------------------------------------------------------------
    # Contract (scripts/fn.py is the single source of truth)
    # ------------------------------------------------------------------

    def _parse_contract(self, name: str):
        """Parse <fn>/scripts/fn.py into a ParsedFn, or None if no CONTRACT."""
        from .fn_contract import parse_fn_source

        code = self.read_function_file(name, "scripts/fn.py")
        if not code or "CONTRACT" not in code:
            return None
        try:
            return parse_fn_source(code, name)
        except ValueError:
            return None

    def has_contract(self, name: str) -> bool:
        """True if the function declares a module-level CONTRACT dict."""
        return self._parse_contract(name) is not None

    def validate_function(self, name: str) -> list[str]:
        """Signature/CONTRACT drift issues (empty == consistent; ``["no CONTRACT"]`` if none)."""
        from .fn_contract import validate_contract

        parsed = self._parse_contract(name)
        if parsed is None:
            return ["no CONTRACT"]
        return validate_contract(parsed)

    def regenerate_fn(self, name: str) -> list[str]:
        """(Re)generate ``<fn>/fn.md`` from the CONTRACT;
        returns drift issues (``["no CONTRACT"]`` when there is no CONTRACT)."""
        from .fn_contract import parse_fn_source, render_fn_md

        fn_dir = self._resolve_fn_dir(name)
        if fn_dir is None:
            raise ImportError(f"Function '{name}' not found in any function directory")
        code = self.read_function_file(name, "scripts/fn.py")
        if not code or "CONTRACT" not in code:
            return ["no CONTRACT"]
        parsed = parse_fn_source(code, name)
        issues = self.validate_function(name)
        (fn_dir / "fn.md").write_text(render_fn_md(parsed), encoding="utf-8")
        if issues:
            logger.warning("regenerate_fn(%s): contract issues %s", name, issues)
        return issues

    def semantic_params(self, name: str) -> list:
        """Semantic (non-DataFrame, non-system) params of *name*; ``[]`` without a CONTRACT."""
        from .fn_contract import semantic_params as _semantic_params

        parsed = self._parse_contract(name)
        return _semantic_params(parsed) if parsed is not None else []

    def regenerate_builtin(self) -> dict[str, list[str]]:
        """Run :meth:`regenerate_fn` on every built-in function with a CONTRACT."""
        out: dict[str, list[str]] = {}
        if not self._builtin_fn_dir.is_dir():
            return out
        for child in sorted(self._builtin_fn_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            issues = self.regenerate_fn(child.name)
            if issues != ["no CONTRACT"]:
                out[child.name] = issues
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a text file, returning empty string if missing."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    @staticmethod
    def _extract_description_and_output(fn_md: str) -> str:
        """Keep the preamble, ``## Output`` and ``## Selection Guidance`` of ``fn_md``."""
        lines = fn_md.split("\n")
        preamble: list[str] = []
        output_section: list[str] = []
        selection_guidance: list[str] = []
        current_section: str | None = None
        _KEEP_SECTIONS = {"Output", "Selection Guidance"}

        for line in lines:
            if line.startswith("## "):
                current_section = line[3:].strip()
                if current_section in _KEEP_SECTIONS:
                    (
                        output_section
                        if current_section == "Output"
                        else selection_guidance
                    ).append(line)
                continue
            if current_section is None:
                preamble.append(line)
            elif current_section == "Output":
                output_section.append(line)
            elif current_section == "Selection Guidance":
                selection_guidance.append(line)

        parts = preamble
        if output_section:
            parts = parts + [""] + output_section
        if selection_guidance:
            parts = parts + [""] + selection_guidance
        return "\n".join(parts).strip()
