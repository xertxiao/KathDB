"""DBContext catalog regressions: identifier quoting, _kdb_ filtering,
stale-metadata purge on re-register, and repeatable sampling."""

from __future__ import annotations

import pandas as pd

from kathdb.common.context import DBContext, _quote_ident
from kathdb.common.view_schema import Modality


def _ctx(tmp_path, **kwargs) -> DBContext:
    return DBContext(tmp_path / "catalog.duckdb", **kwargs)


def test_quote_ident_plain_and_embedded_quote():
    assert _quote_ident("movies") == '"movies"'
    assert _quote_ident('weird"name') == '"weird""name"'
    assert _quote_ident('a""b') == '"a""""b"'


def test_list_tables_does_not_hide_xkdb_tables(tmp_path):
    ctx = _ctx(tmp_path, skip_views=True)
    ctx.register_table(pd.DataFrame({"a": [1]}), "xkdb_foo", description="d")
    tables = ctx.list_tables()
    assert "xkdb_foo" in tables
    assert not any(t.startswith("_kdb_") for t in tables)
    ctx.close()


def test_register_table_with_quote_in_name_is_safe(tmp_path):
    ctx = _ctx(tmp_path, skip_views=True)
    name = 'weird"name'
    ctx.register_table(pd.DataFrame({"a": [1, 2]}), name, description="d")
    assert ctx.has_table(name)
    # Exactly the intended table exists — no quote-mutated sibling table.
    assert ctx.list_tables() == [name]
    assert len(ctx.load_table(name)) == 2
    assert "2 rows" in ctx.describe_table(name)
    ctx.close()


def test_reregister_purges_stale_column_metadata(tmp_path):
    ctx = _ctx(tmp_path, skip_views=True)
    ctx.register_table(
        pd.DataFrame({"a": [1], "b": ["x"]}),
        "t",
        column_modalities={"b": Modality.TEXT},
        description="d",
        column_descriptions={"a": "col a", "b": "col b"},
    )
    # Re-register without column b: its metadata rows must not survive.
    ctx.register_table(
        pd.DataFrame({"a": [1]}),
        "t",
        description="d",
        column_descriptions={"a": "col a"},
    )
    desc_cols = [
        r[0]
        for r in ctx.execute(
            "SELECT column_name FROM _kdb_column_descriptions WHERE table_name = 't'"
        ).fetchall()
    ]
    assert desc_cols == ["a"]
    mod_cols = ctx.execute(
        "SELECT column_name FROM _kdb_column_modalities WHERE table_name = 't'"
    ).fetchall()
    assert mod_cols == []
    ctx.close()


def test_reregister_purges_stale_view_sources(tmp_path):
    ctx = _ctx(tmp_path, skip_views=False)
    ctx.register_table(
        pd.DataFrame({"a": [1], "b": ["some text"]}),
        "t",
        column_modalities={"b": Modality.TEXT},
        description="d",
        column_descriptions={"a": "col a", "b": "col b"},
    )
    assert (
        ctx.execute(
            "SELECT COUNT(*) FROM _kdb_view_sources WHERE source_table = 't'"
        ).fetchone()[0]
        > 0
    )
    ctx.register_table(pd.DataFrame({"a": [1]}), "t", description="d")
    assert (
        ctx.execute(
            "SELECT COUNT(*) FROM _kdb_view_sources WHERE source_table = 't'"
        ).fetchone()[0]
        == 0
    )
    ctx.close()


def test_load_table_sampling_is_repeatable(tmp_path):
    ctx = _ctx(tmp_path, skip_views=True)
    ctx.register_table(pd.DataFrame({"a": range(500)}), "big", description="d")
    s1 = ctx.load_table("big", 5)
    s2 = ctx.load_table("big", 5)
    assert len(s1) == 5
    assert s1.equals(s2)
    ctx.close()
