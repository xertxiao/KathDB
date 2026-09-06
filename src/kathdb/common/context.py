"""DuckDB-backed catalog for KathDB table and schema management."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import duckdb
from pandas import DataFrame

from langchain_core.language_models import BaseChatModel

from .logger import get_logger
from .response_schemas import TableDescriptionResponse
from .schema_descriptions import SCHEMA_COLUMN_DESCRIPTIONS
from .utils import invoke_structured_with_retry, sample_dataframe
from .view_schema import (
    Modality,
    ViewSource,
    _IMAGE_VIEWS,
    _VIDEO_VIEWS,
    _TEXT_VIEWS,
)

logger = get_logger(__name__)

__all__ = ["DBContext"]

_METADATA_PREFIX = "_kdb_"


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier. Every name interpolated into SQL must go through
    this (identifiers cannot be bound as ``?`` parameters; names may come from LLM output)."""
    return '"' + str(name).replace('"', '""') + '"'


class DBContext:
    """DuckDB-backed catalog (``db_path`` is created if missing). ``llm``, when
    given, auto-generates table/column descriptions that are not supplied."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_path: str | Path,
        *,
        llm: BaseChatModel | None = None,
        skip_views: bool = False,
    ) -> None:
        self._db_path = str(db_path)
        self._llm = llm
        self._skip_views = skip_views
        self.conn = duckdb.connect(self._db_path)
        self._setup_metadata_tables()
        # Writes are held in a transaction until save() commits.
        self.conn.execute("BEGIN TRANSACTION")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        llm: BaseChatModel | None = None,
        skip_views: bool = False,
    ) -> DBContext:
        """Open an existing DuckDB catalog file."""
        return cls(db_path=path, llm=llm, skip_views=skip_views)

    def save(self, path: str | Path) -> None:
        """Commit, checkpoint, and optionally save another copy."""
        path = Path(path)
        path_str = str(path)
        self.conn.execute("COMMIT")
        self.conn.execute("CHECKPOINT")
        if path_str != self._db_path:
            shutil.copy2(self._db_path, path_str)
        self.conn.execute("BEGIN TRANSACTION")

    def close(self) -> None:
        """Close the connection, discarding any uncommitted changes."""
        try:
            self.conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        self.conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_table(
        self,
        df: DataFrame,
        table_name: str,
        column_modalities: Dict[str, Modality] | None = None,
        description: str | None = None,
        column_descriptions: Dict[str, str] | None = None,
    ) -> None:
        """Register ``df`` as table ``table_name``. ``column_modalities``
        (``{column: Modality}``) triggers canonical view expansion; missing
        descriptions are LLM-generated when an LLM is configured."""
        if table_name.startswith(_METADATA_PREFIX):
            raise ValueError(
                f"Table name {table_name!r} must not begin with "
                f"{_METADATA_PREFIX!r} (reserved for metadata tables)."
            )
        working_df = df.copy()

        if column_modalities:
            for col, modality in column_modalities.items():
                if col not in working_df.columns:
                    raise KeyError(
                        f"Column '{col}' not found in DataFrame "
                        f"for table '{table_name}'"
                    )
                self._set_column_modality(table_name, col, modality)
                if not self._skip_views:
                    if modality is Modality.IMAGE:
                        self._expand_image(working_df, table_name, col)
                    elif modality is Modality.VIDEO:
                        self._expand_video(working_df, table_name, col)
                    elif modality is Modality.TEXT:
                        self._expand_text(working_df, table_name, col)

        self._store_table(table_name, working_df)

        if description is not None:
            self._set_description(table_name, description)
        elif self._llm is not None:
            self._generate_description(table_name)
        else:
            self._set_description(table_name, f"User-provided table `{table_name}`.")

        # Explicit column descriptions are applied last so they win over generated ones.
        if column_descriptions:
            columns = self.get_columns(table_name)
            for col, col_desc in column_descriptions.items():
                if col in columns:
                    self._set_column_description(table_name, col, col_desc)

    def register_empty_typed_table(
        self,
        table_name: str,
        columns: list[tuple[str, str]],
        description: str | None = None,
        column_descriptions: Dict[str, str] | None = None,
    ) -> None:
        """Register an empty table from ordered ``(column_name, duckdb_type)`` pairs."""
        if table_name.startswith(_METADATA_PREFIX):
            raise ValueError(
                f"Table name {table_name!r} must not begin with "
                f"{_METADATA_PREFIX!r} (reserved for metadata tables)."
            )
        self.conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(table_name)}")
        col_names = tuple(c for c, _ in columns)
        dtypes_map = {c: dt for c, dt in columns}
        self._create_empty_view_table(table_name, col_names, dtypes_map)

        if description is not None:
            self._set_description(table_name, description)
        else:
            self._set_description(table_name, f"Table `{table_name}`.")

        if column_descriptions:
            stored_cols = self.get_columns(table_name)
            for col, col_desc in column_descriptions.items():
                if col in stored_cols:
                    self._set_column_description(table_name, col, col_desc)

    def load_table(self, name: str, n: int | None = None) -> DataFrame:
        """Load a table as a DataFrame; with *n*, a seeded reservoir sample of *n* rows."""
        if n is not None:
            return self.conn.execute(
                f"SELECT * FROM {_quote_ident(name)} "
                f"USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE (17)"
            ).df()
        return self.conn.execute(f"SELECT * FROM {_quote_ident(name)}").df()

    def has_table(self, name: str) -> bool:
        """Return True if a user table (not metadata) with *name* exists."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema = 'main'",
            [name],
        ).fetchone()
        return result is not None and result[0] > 0

    def list_tables(self) -> list[str]:
        """List user tables (excludes ``_kdb_`` metadata tables)."""
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' "
            "AND NOT starts_with(table_name, ?) "
            "ORDER BY table_name",
            [_METADATA_PREFIX],
        ).fetchall()
        return [r[0] for r in rows]

    def get_columns(self, name: str) -> list[str]:
        """Return column names for *name* in ordinal order."""
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = 'main' "
            "ORDER BY ordinal_position",
            [name],
        ).fetchall()
        return [r[0] for r in rows]

    def execute(self, query: str, params: list | None = None):
        """Execute a SQL statement and return the raw DuckDB result."""
        if params is not None:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def describe_table(
        self,
        name: str,
        *,
        cell_char_limit: int = 80,
        sample_n: int = 3,
    ) -> str | None:
        """Human-readable description of *name* (schema, PK, descriptions, sample
        rows); None if the table does not exist."""
        if not self.has_table(name):
            return None

        columns = self.get_columns(name)
        if not columns:
            return None
        dtypes = self._get_dtypes(name)
        primary_key = self._get_primary_key(name)
        table_desc = self._get_description(name)

        count = self.conn.execute(
            f"SELECT COUNT(*) FROM {_quote_ident(name)}"
        ).fetchone()[0]

        desc = (
            f"Table `{name}` ({count} rows) with columns: {', '.join(columns)} and dtypes: "
            f"{', '.join(f'{c}: {dtypes[c]}' for c in columns)}."
        )
        if primary_key:
            desc += f" Primary key: ({', '.join(primary_key)})."
        if table_desc:
            desc += f" Description: {table_desc.strip()}"

        vs = self.get_view_source(name) if self.is_view(name) else None
        if vs is not None:
            desc += (
                f"\nView source: derived from {vs.source_table}.{vs.source_column}"
                f" ({vs.modality.value} data)"
            )

        col_modalities = self._get_table_column_modalities(name)
        if col_modalities:
            mod_lines = "\n".join(
                f"  {c}: {col_modalities[c]}" for c in columns if c in col_modalities
            )
            if mod_lines:
                desc += f"\nColumn modalities:\n{mod_lines}"

        col_descs = self._get_column_descriptions(name)
        if col_descs:
            col_lines = "\n".join(
                f"  {c}: {col_descs[c]}" for c in columns if c in col_descs
            )
            if col_lines:
                desc += f"\nColumn descriptions:\n{col_lines}"

        try:
            if count > 0:
                sample_df = self.conn.execute(
                    f"SELECT * FROM {_quote_ident(name)} LIMIT {sample_n}"
                ).df()
                if cell_char_limit and cell_char_limit > 0:
                    sample_df = sample_df.copy().map(
                        lambda value: (
                            ("" if value is None else str(value))[: cell_char_limit - 3]
                            + (
                                "..."
                                if value is not None
                                and len(str(value)) > cell_char_limit
                                else ""
                            )
                            if cell_char_limit > 3
                            else ("" if value is None else str(value))[:cell_char_limit]
                        )
                    )
                serialized = sample_df.to_csv(index=False).strip()
                if serialized:
                    desc += f" Sample rows:\n{serialized}"
        except Exception:  # noqa: BLE001
            pass

        return desc

    def describe_all_tables(
        self, *, cell_char_limit: int = 80, compact_views: bool = True
    ) -> list[str]:
        """Description of every user table; view tables as one line when *compact_views*."""
        descriptions: list[str] = []
        for name in self.list_tables():
            if compact_views and self.is_view(name):
                vs = self.get_view_source(name)
                cols = self.get_columns(name)
                line = (
                    f"View `{name}`: from {vs.source_table}.{vs.source_column} "
                    f"({vs.modality.value}), columns: {', '.join(cols)}"
                )
                descriptions.append(line)
            else:
                text = self.describe_table(name, cell_char_limit=cell_char_limit)
                if text is not None:
                    descriptions.append(text)
        return descriptions

    def inspect(
        self,
        name: str | None = None,
        *,
        sample_n: int = 3,
        cell_char_limit: int = 40,
    ) -> None:
        """Print an overview of the catalog, or of one table when *name* is given."""
        SEP = "─"
        tables = self.list_tables()

        if name is not None:
            if not self.has_table(name):
                print(f"Table '{name}' not found.")
                return
            cols = self.get_columns(name)
            dtypes = self._get_dtypes(name)
            count = self.conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(name)}"
            ).fetchone()[0]
            pk = self._get_primary_key(name)
            desc = self._get_description(name)
            view_src = self.get_view_source(name)

            print(f"\n{SEP * 60}")
            print(f"  Table: {name}  ({count} rows)")
            print(SEP * 60)
            if desc:
                for line in desc.strip().splitlines()[:3]:
                    print(f"  {line}")
                print()
            if pk:
                print(f"  PK: ({', '.join(pk)})")
            if view_src:
                print(
                    f"  View source: {view_src.source_table}"
                    f".{view_src.source_column} ({view_src.modality.value})"
                )
            print()
            max_col_len = max(len(c) for c in cols) if cols else 0
            max_type_len = max(len(dtypes[c]) for c in cols) if cols else 0
            col_descs = self._get_column_descriptions(name)
            for c in cols:
                col_desc = col_descs.get(c, "")
                desc_suffix = f"  {col_desc}" if col_desc else ""
                print(f"  {c:<{max_col_len}}  {dtypes[c]:<{max_type_len}}{desc_suffix}")
            print()
            if count > 0:
                sample_df = self.conn.execute(
                    f"SELECT * FROM {_quote_ident(name)} LIMIT {sample_n}"
                ).df()
                if cell_char_limit and cell_char_limit > 0:
                    sample_df = sample_df.copy().map(
                        lambda v: (
                            str(v)[: cell_char_limit - 3] + "..."
                            if v is not None and len(str(v)) > cell_char_limit
                            else ("" if v is None else str(v))
                        )
                    )
                print(f"  Sample ({min(count, sample_n)} of {count} rows):")
                for line in sample_df.to_string(index=False).splitlines():
                    print(f"  {line}")
            print(SEP * 60)
            return

        if not tables:
            print("No user tables.")
            return

        max_name_len = max(len(t) for t in tables)
        row_counts = {}
        for t in tables:
            row_counts[t] = self.conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(t)}"
            ).fetchone()[0]

        print(f"\n{SEP * 60}")
        print(f"  Catalog: {len(tables)} tables  " f"(db: {self._db_path})")
        print(SEP * 60)
        for t in tables:
            cols = self.get_columns(t)
            tag = ""
            if self.is_view(t):
                tag = " [view]"
            print(
                f"  {t:<{max_name_len}}  {row_counts[t]:>6} rows  "
                f"cols=({', '.join(cols)}){tag}"
            )
        print(SEP * 60)

    def is_view(self, name: str) -> bool:
        """Return True if *name* is an auto-expanded multimodal view."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM _kdb_view_sources WHERE view_name = ?",
            [name],
        ).fetchone()
        return result is not None and result[0] > 0

    def get_view_source(self, name: str) -> ViewSource | None:
        """Return the :class:`ViewSource` for a view, or *None*."""
        result = self.conn.execute(
            "SELECT source_table, source_column, modality "
            "FROM _kdb_view_sources WHERE view_name = ?",
            [name],
        ).fetchone()
        if result is None:
            return None
        return ViewSource(
            source_table=result[0],
            source_column=result[1],
            modality=Modality(result[2]),
        )

    def _get_table_column_modalities(self, name: str) -> dict[str, str]:
        """Return column modalities for a specific table as ``{column: modality_str}``."""
        rows = self.conn.execute(
            "SELECT column_name, modality FROM _kdb_column_modalities "
            "WHERE table_name = ?",
            [name],
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_execution_context(self, names: list[str] | None = None) -> dict[str, Any]:
        """``{name: DataFrame}`` for *names* (all user tables when None)."""
        if names is None:
            names = self.list_tables()
        ctx: dict[str, Any] = {}
        for name in names:
            if self.has_table(name):
                ctx[name] = self.load_table(name)
        return ctx

    def discover(
        self,
        root: str | Path,
        *,
        recursive: bool = True,
        llm_assist: bool | None = None,
    ) -> list[str]:
        """Auto-register every tabular file and media folder under *root* (see
        :class:`~.auto_discovery.AutoDiscovery`); ``llm_assist`` defaults to
        whether an LLM is configured."""
        from .auto_discovery import AutoDiscovery

        use_llm = (self._llm is not None) if llm_assist is None else llm_assist
        return AutoDiscovery(self, llm_assist=use_llm).discover(
            root, recursive=recursive
        )

    def import_execution_results(self, ctx: dict[str, Any], names: list[str]) -> None:
        """Store the DataFrames ``ctx[name]`` for *names* as tables."""
        for name in names:
            if name in ctx:
                value = ctx[name]
                if isinstance(value, DataFrame):
                    self._store_table(name, value)

    # ------------------------------------------------------------------
    # Private: table storage
    # ------------------------------------------------------------------

    def _store_table(self, name: str, df: DataFrame) -> None:
        """Store or replace a DataFrame as a DuckDB table."""
        self.conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(name)}")
        self.conn.execute(f"CREATE TABLE {_quote_ident(name)} AS SELECT * FROM df")
        self._purge_stale_column_metadata(name)

    def _purge_stale_column_metadata(self, name: str) -> None:
        """Drop metadata rows for columns that no longer exist on *name*."""
        cols = self.get_columns(name)
        if cols:
            placeholders = ", ".join("?" for _ in cols)
            self.conn.execute(
                "DELETE FROM _kdb_column_descriptions "
                f"WHERE table_name = ? AND column_name NOT IN ({placeholders})",
                [name, *cols],
            )
            self.conn.execute(
                "DELETE FROM _kdb_column_modalities "
                f"WHERE table_name = ? AND column_name NOT IN ({placeholders})",
                [name, *cols],
            )
            self.conn.execute(
                "DELETE FROM _kdb_view_sources "
                f"WHERE source_table = ? AND source_column NOT IN ({placeholders})",
                [name, *cols],
            )
        else:  # pragma: no cover - a stored table always has >= 1 column
            self.conn.execute(
                "DELETE FROM _kdb_column_descriptions WHERE table_name = ?", [name]
            )
            self.conn.execute(
                "DELETE FROM _kdb_column_modalities WHERE table_name = ?", [name]
            )
            self.conn.execute(
                "DELETE FROM _kdb_view_sources WHERE source_table = ?", [name]
            )

    def _setup_metadata_tables(self) -> None:
        """Create the ``_kdb_*`` metadata tables if they do not exist."""
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS _kdb_view_sources ("
            "  view_name VARCHAR PRIMARY KEY,"
            "  source_table VARCHAR NOT NULL,"
            "  source_column VARCHAR NOT NULL,"
            "  modality VARCHAR NOT NULL"
            ")"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS _kdb_descriptions ("
            "  table_name VARCHAR PRIMARY KEY,"
            "  description VARCHAR NOT NULL"
            ")"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS _kdb_column_descriptions ("
            "  table_name VARCHAR NOT NULL,"
            "  column_name VARCHAR NOT NULL,"
            "  description VARCHAR NOT NULL,"
            "  PRIMARY KEY (table_name, column_name)"
            ")"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS _kdb_column_modalities ("
            "  table_name VARCHAR NOT NULL,"
            "  column_name VARCHAR NOT NULL,"
            "  modality VARCHAR NOT NULL,"
            "  PRIMARY KEY (table_name, column_name)"
            ")"
        )

    # ------------------------------------------------------------------
    # Private: schema introspection
    # ------------------------------------------------------------------

    def _get_dtypes(self, name: str) -> dict[str, str]:
        """Return ``{column: duckdb_type}`` for *name*."""
        rows = self.conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = 'main' "
            "ORDER BY ordinal_position",
            [name],
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def _get_primary_key(self, name: str) -> tuple[str, ...] | None:
        """Query DuckDB for the PRIMARY KEY constraint on *name*."""
        try:
            rows = self.conn.execute(
                "SELECT constraint_column_names FROM duckdb_constraints() "
                "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
                [name],
            ).fetchall()
            if rows and rows[0][0]:
                return tuple(rows[0][0])
        except Exception:  # noqa: BLE001
            pass
        return None

    # ------------------------------------------------------------------
    # Private: description management
    # ------------------------------------------------------------------

    def _set_description(self, name: str, description: str) -> None:
        """Set or update the human-readable description for a table."""
        self.conn.execute(
            "INSERT INTO _kdb_descriptions (table_name, description) VALUES (?, ?) "
            "ON CONFLICT DO UPDATE SET description = excluded.description",
            [name, description],
        )

    def _get_description(self, name: str) -> str | None:
        """Retrieve the stored description for a table."""
        result = self.conn.execute(
            "SELECT description FROM _kdb_descriptions WHERE table_name = ?",
            [name],
        ).fetchone()
        return result[0] if result else None

    def _set_column_description(
        self, table: str, column: str, description: str
    ) -> None:
        """Set or update the description for a single column."""
        self.conn.execute(
            "INSERT INTO _kdb_column_descriptions (table_name, column_name, description) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (table_name, column_name) DO UPDATE SET description = excluded.description",
            [table, column, description],
        )

    def _get_column_descriptions(self, table: str) -> dict[str, str]:
        """Return ``{column: description}`` for *table*."""
        rows = self.conn.execute(
            "SELECT column_name, description FROM _kdb_column_descriptions "
            "WHERE table_name = ?",
            [table],
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Private: LLM description generation
    # ------------------------------------------------------------------

    def _generate_description(
        self,
        name: str,
        *,
        llm: BaseChatModel | None = None,
        k: int = 5,
    ) -> str:
        """LLM-generate and store the table summary and column descriptions
        (identifier columns use the pre-written ``SCHEMA_COLUMN_DESCRIPTIONS``)."""
        llm = llm or self._llm
        if llm is None:
            raise ValueError("No LLM available for description generation")

        columns = self.get_columns(name)
        dtypes = self._get_dtypes(name)
        df = self.load_table(name)
        sample_df = sample_dataframe(df, k)
        sample_records = sample_df.to_dict(orient="records")

        identifier_columns = {c for c in columns if c in SCHEMA_COLUMN_DESCRIPTIONS}

        for c in columns:
            id_desc = SCHEMA_COLUMN_DESCRIPTIONS.get(c)
            if id_desc:
                self._set_column_description(name, c, id_desc)

        non_id_pairs = [(c, dtypes[c]) for c in columns if c not in identifier_columns]
        non_id_rows = [
            {k_: v for k_, v in row.items() if k_ not in identifier_columns}
            for row in sample_records
        ]

        table_summary: str = ""
        if non_id_pairs:
            result = self._llm_describe(
                llm=llm,
                table_name=name,
                dtype_pairs=non_id_pairs,
                sample_rows=non_id_rows,
            )
            table_summary = result["table_summary"]
            for col_name, col_desc in result["column_descriptions"].items():
                self._set_column_description(name, col_name, col_desc)

        if table_summary:
            self._set_description(name, table_summary)

        return table_summary

    @staticmethod
    def _llm_describe(
        *,
        llm: BaseChatModel,
        table_name: str,
        dtype_pairs: Iterable[tuple[str, str]],
        sample_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask the LLM for ``{"table_summary", "column_descriptions"}``."""
        col_overview = "\n".join(f"- {n}: {d}" for n, d in dtype_pairs)

        def _truncate(v: Any, limit: int = 200) -> Any:
            s = str(v)
            return s if len(s) <= limit else s[: limit - 3] + "..."

        formatted: list[str] = []
        for row in sample_rows:
            safe = {}
            for k_, v in row.items():
                try:
                    json.dumps(v)
                    safe[k_] = _truncate(v)
                except TypeError:
                    safe[k_] = f"<{type(v).__name__}>"
            formatted.append(json.dumps(safe, ensure_ascii=True))
        sample_text = "[\n" + "\n".join(formatted) + "\n]" if formatted else "[]"

        col_names = set(n for n, _ in dtype_pairs)

        prompt = (
            "You describe relational table schemas.\n\n"
            "Given the table metadata below, produce:\n"
            "- A single sentence summarizing what the table captures.\n"
            "- A short semantic description for each column.\n\n"
            "Rules:\n"
            "- Do NOT include data types in the column descriptions.\n"
            "- Base descriptions strictly on the provided metadata and sample rows.\n"
            "- Keep descriptions short and factual.\n\n"
            f"Table name: {table_name}\n"
            f"Column metadata:\n{col_overview}\n"
            f"Sample rows (at most {len(sample_rows)}, JSON format):\n{sample_text}\n"
        )

        response: TableDescriptionResponse = invoke_structured_with_retry(
            prompt,
            llm=llm,
            schema=TableDescriptionResponse,
        )

        column_descriptions = {
            cd.column_name: cd.description
            for cd in response.column_descriptions
            if cd.column_name in col_names
        }

        return {
            "table_summary": response.table_summary,
            "column_descriptions": column_descriptions,
        }

    # ------------------------------------------------------------------
    # Private: column modality management
    # ------------------------------------------------------------------

    def _set_column_modality(
        self, table_name: str, column_name: str, modality: Modality
    ) -> None:
        """Record a column's modality in ``_kdb_column_modalities``."""
        self.conn.execute(
            "INSERT INTO _kdb_column_modalities (table_name, column_name, modality) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (table_name, column_name) "
            "DO UPDATE SET modality = excluded.modality",
            [table_name, column_name, modality.value],
        )

    # ------------------------------------------------------------------
    # Private: view source management
    # ------------------------------------------------------------------

    def _register_view_source(
        self,
        view_name: str,
        source_table: str,
        source_column: str,
        modality: Modality,
    ) -> None:
        """Insert or update a view-source record in the metadata table."""
        self.conn.execute(
            "INSERT INTO _kdb_view_sources (view_name, source_table, source_column, modality) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT DO UPDATE SET "
            "source_table = excluded.source_table, "
            "source_column = excluded.source_column, "
            "modality = excluded.modality",
            [view_name, source_table, source_column, modality.value],
        )

    # ------------------------------------------------------------------
    # Private: multimodal view expansion
    # ------------------------------------------------------------------

    def _create_empty_view_table(
        self,
        view_name: str,
        columns: Tuple[str, ...],
        dtypes: Dict[str, str],
        primary_key: Tuple[str, ...] | None = None,
    ) -> None:
        """Create an empty DuckDB table with explicit column types and optional PK."""
        col_defs = []
        for col in columns:
            duckdb_type = dtypes.get(col, "VARCHAR")
            col_defs.append(f"{_quote_ident(col)} {duckdb_type}")
        if primary_key:
            pk_cols = ", ".join(_quote_ident(c) for c in primary_key)
            col_defs.append(f"PRIMARY KEY ({pk_cols})")
        cols_sql = ", ".join(col_defs)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_quote_ident(view_name)} ({cols_sql})"
        )

    def _expand_image(
        self,
        df: DataFrame,
        table_name: str,
        col: str,
    ) -> None:
        """Add an ``iid`` identifier column and create canonical image views."""
        df["iid"] = range(len(df))

        prefix = f"{table_name}_{col}_{Modality.IMAGE.value}"

        for suffix, columns, dtypes, summary, primary_key in _IMAGE_VIEWS:
            view_name = f"{prefix}_{suffix}"
            self._create_empty_view_table(view_name, columns, dtypes, primary_key)
            self._set_description(view_name, summary)
            for c in columns:
                col_desc = SCHEMA_COLUMN_DESCRIPTIONS.get(c)
                if col_desc:
                    self._set_column_description(view_name, c, col_desc)
            self._register_view_source(view_name, table_name, col, Modality.IMAGE)

    def _expand_video(
        self,
        df: DataFrame,
        table_name: str,
        col: str,
    ) -> None:
        """Add ``vid`` / ``fid`` identifier columns and create canonical video views."""
        df["vid"] = 0
        df["fid"] = range(len(df))

        prefix = f"{table_name}_{col}_{Modality.VIDEO.value}"

        for suffix, columns, dtypes, summary, primary_key in _VIDEO_VIEWS:
            view_name = f"{prefix}_{suffix}"
            self._create_empty_view_table(view_name, columns, dtypes, primary_key)
            self._set_description(view_name, summary)
            for c in columns:
                col_desc = SCHEMA_COLUMN_DESCRIPTIONS.get(c)
                if col_desc:
                    self._set_column_description(view_name, c, col_desc)
            self._register_view_source(view_name, table_name, col, Modality.VIDEO)

    def _expand_text(
        self,
        df: DataFrame,
        table_name: str,
        col: str,
    ) -> None:
        """Add identifier columns and create canonical text views."""
        df["did"] = range(len(df))

        prefix = f"{table_name}_{col}_{Modality.TEXT.value}"

        for suffix, columns, dtypes, summary, primary_key in _TEXT_VIEWS:
            view_name = f"{prefix}_{suffix}"
            self._create_empty_view_table(view_name, columns, dtypes, primary_key)
            self._set_description(view_name, summary)
            for c in columns:
                col_desc = SCHEMA_COLUMN_DESCRIPTIONS.get(c)
                if col_desc:
                    self._set_column_description(view_name, c, col_desc)
            self._register_view_source(view_name, table_name, col, Modality.TEXT)
