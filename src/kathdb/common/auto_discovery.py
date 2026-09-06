"""Folder-based auto-discovery: infer modalities from file extensions and column
contents and register everything via :meth:`DBContext.register_table`."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_object_dtype, is_string_dtype

from .logger import get_logger
from .view_schema import Modality

if TYPE_CHECKING:  # pragma: no cover
    from .context import DBContext

logger = get_logger(__name__)

__all__ = [
    "EXTENSION_TO_MODALITY",
    "TABULAR_EXTENSIONS",
    "DiscoveredTable",
    "AutoDiscovery",
]


EXTENSION_TO_MODALITY: dict[str, Modality] = {
    ".jpg": Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".png": Modality.IMAGE,
    ".gif": Modality.IMAGE,
    ".bmp": Modality.IMAGE,
    ".webp": Modality.IMAGE,
    ".mp4": Modality.VIDEO,
    ".mov": Modality.VIDEO,
    ".avi": Modality.VIDEO,
    ".mkv": Modality.VIDEO,
    ".webm": Modality.VIDEO,
    ".wav": Modality.AUDIO,
    ".mp3": Modality.AUDIO,
    ".flac": Modality.AUDIO,
    ".ogg": Modality.AUDIO,
    ".m4a": Modality.AUDIO,
    ".txt": Modality.TEXT,
    ".md": Modality.TEXT,
}

TABULAR_EXTENSIONS: set[str] = {".csv", ".parquet", ".tsv", ".jsonl", ".json"}

# Table-name suffix per modality when one folder mixes modalities.
_MODALITY_GROUP_SUFFIX: dict[Modality, str] = {
    Modality.IMAGE: "images",
    Modality.VIDEO: "videos",
    Modality.AUDIO: "audio",
    Modality.TEXT: "documents",
}


def _modality_for_file(p: Path) -> Modality | None:
    """Return the modality for *p* based on its file extension, or None."""
    return EXTENSION_TO_MODALITY.get(p.suffix.lower())


def _snake_case(name: str) -> str:
    """Lowercase a string and convert non-alphanumeric runs to single underscores."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    if not s:
        s = "table"
    if s[0].isdigit():
        s = f"t_{s}"
    return s


@dataclass
class DiscoveredTable:
    """One planned registration (DataFrame + metadata for view expansion)."""

    table_name: str
    df: DataFrame
    column_modalities: dict[str, Modality] = field(default_factory=dict)
    description: str | None = None
    source_paths: list[Path] = field(default_factory=list)


class AutoDiscovery:
    """Walks a folder and registers discovered files into ``ctx``; ``llm_assist``
    lets the catalog's LLM (if any) name uninformatively named tables."""

    _COLUMN_SAMPLE_ROWS: int = 20
    _COLUMN_HIT_THRESHOLD: float = 0.8

    def __init__(self, ctx: "DBContext", *, llm_assist: bool = True) -> None:
        self._ctx = ctx
        self._llm = ctx._llm if llm_assist else None
        self._used_names: set[str] = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def discover(
        self,
        root: str | Path,
        *,
        recursive: bool = True,
    ) -> list[str]:
        """Walk *root*, plan all discovered tables, register them, return names."""
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Discovery root does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Discovery root is not a directory: {root_path}")

        self._used_names = set(self._ctx.list_tables())

        plan = self._plan(root_path, recursive=recursive)
        registered: list[str] = []
        for dt in plan:
            try:
                self._ctx.register_table(
                    dt.df,
                    dt.table_name,
                    column_modalities=dt.column_modalities or None,
                    description=dt.description,
                )
                registered.append(dt.table_name)
                logger.info(
                    "Auto-registered table '%s' (%d rows, modalities=%s) from %d source(s)",
                    dt.table_name,
                    len(dt.df),
                    {k: v.value for k, v in dt.column_modalities.items()},
                    len(dt.source_paths),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to register auto-discovered table '%s': %s",
                    dt.table_name,
                    exc,
                )
        return registered

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self, root: Path, *, recursive: bool) -> list[DiscoveredTable]:
        """Build the full registration plan for *root*."""
        plan: list[DiscoveredTable] = []
        for directory in self._iter_dirs(root, recursive=recursive):
            self._plan_directory(directory, plan)
        return plan

    def _iter_dirs(self, root: Path, *, recursive: bool) -> Iterable[Path]:
        """Yield directories to scan, starting with *root*."""
        yield root
        if not recursive:
            return
        for child in sorted(root.rglob("*")):
            if child.is_dir():
                yield child

    def _plan_directory(self, directory: Path, plan: list[DiscoveredTable]) -> None:
        """Add tabular and media tables for the *direct* children of *directory*."""
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            logger.warning("Skipping unreadable directory: %s", directory)
            return

        media_groups: dict[Modality, list[Path]] = {}
        for entry in entries:
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext in TABULAR_EXTENSIONS:
                tabular = self._plan_tabular(entry)
                if tabular is not None:
                    plan.append(tabular)
                continue
            modality = _modality_for_file(entry)
            if modality is None:
                continue
            media_groups.setdefault(modality, []).append(entry)

        if not media_groups:
            return

        multi_modality = len(media_groups) > 1
        for modality, files in media_groups.items():
            dt = self._build_media_table(directory, modality, files, multi_modality)
            if dt is not None:
                plan.append(dt)

    # ------------------------------------------------------------------
    # Tabular files
    # ------------------------------------------------------------------

    def _plan_tabular(self, path: Path) -> DiscoveredTable | None:
        """Read a tabular file from disk and plan its registration."""
        try:
            df = self._read_tabular(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable tabular file %s: %s", path, exc)
            return None
        if df is None or df.empty:
            logger.info("Skipping empty tabular file: %s", path)
            return None

        column_modalities = self._scan_columns_for_media(df, base_dir=path.parent)
        table_name = self._reserve_name(self._table_name_from_path(path))
        return DiscoveredTable(
            table_name=table_name,
            df=df,
            column_modalities=column_modalities,
            description=None,
            source_paths=[path],
        )

    @staticmethod
    def _read_tabular(path: Path) -> DataFrame | None:
        """Dispatch on file extension to the right pandas reader."""
        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".jsonl":
            return pd.read_json(path, lines=True)
        if ext == ".json":
            return pd.read_json(path)
        return None

    def _scan_columns_for_media(
        self,
        df: DataFrame,
        *,
        base_dir: Path,
    ) -> dict[str, Modality]:
        """Inspect string columns; tag any whose values resolve to media files."""
        result: dict[str, Modality] = {}
        for col in df.columns:
            series = df[col]
            if not (is_object_dtype(series) or is_string_dtype(series)):
                continue
            sample = series.dropna()
            if sample.empty:
                continue
            sample = sample.head(self._COLUMN_SAMPLE_ROWS)
            modality = self._infer_column_modality(sample, base_dir=base_dir)
            if modality is not None:
                result[col] = modality
        return result

    def _infer_column_modality(
        self,
        sample: pd.Series,
        *,
        base_dir: Path,
    ) -> Modality | None:
        """Return the dominant media modality for *sample*, or None."""
        total = 0
        hits: dict[Modality, int] = {}
        for value in sample:
            if not isinstance(value, str):
                continue
            v = value.strip()
            if not v or v.startswith(("http://", "https://")):
                continue
            total += 1
            ext = Path(v).suffix.lower()
            modality = EXTENSION_TO_MODALITY.get(ext)
            if modality is None:
                continue
            candidate = Path(v)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            if not candidate.exists():
                continue
            hits[modality] = hits.get(modality, 0) + 1

        if total == 0 or not hits:
            return None
        modality, count = max(hits.items(), key=lambda kv: kv[1])
        if count / total < self._COLUMN_HIT_THRESHOLD:
            return None
        return modality

    # ------------------------------------------------------------------
    # Media folder tables
    # ------------------------------------------------------------------

    def _build_media_table(
        self,
        directory: Path,
        modality: Modality,
        files: list[Path],
        multi_modality: bool,
    ) -> DiscoveredTable | None:
        """Construct a DiscoveredTable for a group of sibling media files."""
        if not files:
            return None

        rows = []
        for f in files:
            row = {
                "path": str(f.resolve()),
                "filename": f.name,
                "relative_path": str(f.relative_to(directory)),
            }
            if modality is Modality.TEXT:
                row["content"] = self._read_text_file(f)
            rows.append(row)
        df = pd.DataFrame(rows)

        if modality is Modality.TEXT:
            column_modalities = {"content": Modality.TEXT}
        else:
            column_modalities = {"path": modality}

        base_name = self._snake_dir_name(directory)
        if multi_modality:
            base_name = f"{base_name}_{_MODALITY_GROUP_SUFFIX[modality]}"
        table_name = self._reserve_name(base_name)

        return DiscoveredTable(
            table_name=table_name,
            df=df,
            column_modalities=column_modalities,
            description=None,
            source_paths=files,
        )

    @staticmethod
    def _read_text_file(p: Path, *, max_chars: int = 1_000_000) -> str:
        """Read a text file with permissive decoding, capped at *max_chars*."""
        try:
            data = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read text file %s: %s", p, exc)
            return ""
        if len(data) > max_chars:
            data = data[:max_chars]
        return data

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def _table_name_from_path(self, path: Path) -> str:
        """Suggest a table name for a tabular *path*, using the LLM if helpful."""
        stem = path.stem
        snake = _snake_case(stem)
        if self._is_uninformative_name(snake):
            llm_name = self._llm_suggest_name(path)
            if llm_name:
                snake = llm_name
        return snake

    def _snake_dir_name(self, directory: Path) -> str:
        """Snake-case a directory name; if root, fall back to ``data``."""
        name = directory.name or "data"
        return _snake_case(name)

    @staticmethod
    def _is_uninformative_name(name: str) -> bool:
        """Heuristic: names like 'data', 'data1', 'untitled', 'sheet1' are weak."""
        if not name:
            return True
        weak_prefixes = ("data", "untitled", "sheet", "table", "file", "export", "tmp")
        return any(name == p or name.startswith(p) for p in weak_prefixes)

    def _reserve_name(self, base: str) -> str:
        """Return a table name guaranteed not to collide with prior reservations."""
        candidate = base
        i = 2
        while candidate in self._used_names:
            candidate = f"{base}_{i}"
            i += 1
        self._used_names.add(candidate)
        return candidate

    # ------------------------------------------------------------------
    # LLM-assisted naming
    # ------------------------------------------------------------------

    def _llm_suggest_name(self, path: Path) -> str | None:
        """Ask the LLM for a short snake_case table name for *path*. Returns None on failure."""
        if self._llm is None:
            return None
        try:
            df = self._read_tabular(path)
        except Exception:  # noqa: BLE001
            df = None
        sample_columns: list[str] = list(df.columns) if df is not None else []
        sample_rows: list[dict] = []
        if df is not None and not df.empty:
            sample_rows = df.head(3).to_dict(orient="records")
            for row in sample_rows:
                for k, v in list(row.items()):
                    try:
                        json.dumps(v)
                    except TypeError:
                        row[k] = str(v)

        prompt = (
            "Suggest a short snake_case table name for the following tabular file.\n"
            "Rules: lowercase, words separated by underscores, no extension, no path,\n"
            "max 3 words, return ONLY the name on a single line.\n\n"
            f"File path: {path}\n"
            f"Columns: {sample_columns}\n"
            f"Sample rows: {json.dumps(sample_rows, default=str)[:500]}\n"
        )
        try:
            response = self._llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM name suggestion failed for %s: %s", path, exc)
            return None
        text = getattr(response, "content", response)
        if not isinstance(text, str):
            return None
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        cleaned = _snake_case(first_line)
        if not cleaned or self._is_uninformative_name(cleaned):
            return None
        return cleaned
