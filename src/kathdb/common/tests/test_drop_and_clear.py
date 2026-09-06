"""drop_table removes a table, its views and metadata; clear_generated empties the library."""

from __future__ import annotations

import pandas as pd

from kathdb.common.context import DBContext
from kathdb.common.function_manager import FunctionManager
from kathdb.common.view_schema import Modality


def test_drop_table_removes_views_and_metadata(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    ctx = DBContext(tmp_path / "c.duckdb", llm=None)
    ctx.register_table(
        pd.DataFrame({"id": [1], "image_path": [str(tmp_path / "a.jpg")]}),
        "products",
        column_modalities={"image_path": Modality.IMAGE},
    )
    assert any(ctx.is_view(n) for n in ctx.list_tables())
    ctx.drop_table("products")
    assert ctx.list_tables() == []
    for meta in ("_kdb_descriptions", "_kdb_column_descriptions", "_kdb_column_modalities", "_kdb_view_sources"):
        assert ctx.conn.execute(f"SELECT COUNT(*) FROM {meta}").fetchone()[0] == 0


def test_clear_generated_functions(tmp_path):
    gen = tmp_path / "gen"
    for name in ("fn_a", "fn_b"):
        (gen / name / "scripts").mkdir(parents=True)
        (gen / name / "scripts" / "fn.py").write_text(f"def {name}(df):\n    return df\n")
        (gen / name / "fn.md").write_text(f"# {name}\n\nDoes {name}.\n")
    fm = FunctionManager(builtin_fn_dir=tmp_path / "builtin", generated_fn_dir=gen)
    assert fm.remove_function("missing") is False
    assert fm.clear_generated() == ["fn_a", "fn_b"]
    assert [p for p in gen.iterdir() if p.is_dir()] == []
