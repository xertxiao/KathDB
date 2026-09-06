"""DBContext catalog behaviour: metadata tables hidden, explicit column descriptions win."""

from __future__ import annotations

import pandas as pd

from kathdb.common.context import DBContext


def test_list_tables_excludes_kdb_metadata(tmp_path):
    db = DBContext(str(tmp_path / "cat.duckdb"))
    try:
        db.register_table(pd.DataFrame({"x": [1, 2]}), "products")
        tables = db.list_tables()
        assert "products" in tables
        assert not any(t.startswith("_kdb_") for t in tables), tables
    finally:
        db.close()


def test_register_table_honors_column_descriptions_with_llm(tmp_path):
    db = DBContext(str(tmp_path / "cat.duckdb"), llm=object())
    db._generate_description = lambda name: None
    try:
        db.register_table(
            pd.DataFrame({"x": [1], "y": ["a"]}),
            "t",
            column_descriptions={"x": "the x column"},
        )
        descs = db._get_column_descriptions("t")
        assert descs.get("x") == "the x column"
    finally:
        db.close()
