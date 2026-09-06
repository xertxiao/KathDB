"""Tests for FunctionManager source filtering."""

from __future__ import annotations

from pathlib import Path

from kathdb.common.function_manager import FunctionManager


_FN_TEMPLATE = """\
import pandas as pd

CONTRACT = {{
    "purpose": "test function",
    "params": {{"df": "input rows", "label": "unused"}},
    "output": "df unchanged",
    "example": '{name}(df, label="")',
    "use_when": "never",
}}


def {name}(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    return df
"""


def _make_fn_dir(root: Path, name: str) -> None:
    fn_dir = root / name
    (fn_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (fn_dir / "fn.md").write_text(f"# {name}\n## Output\nreturns df\n")
    (fn_dir / "scripts" / "fn.py").write_text(_FN_TEMPLATE.format(name=name))
    (fn_dir / "scripts" / "__init__.py").write_text(f"from .fn import {name}\n")


def _fm(tmp_path: Path) -> FunctionManager:
    builtin = tmp_path / "pre_built"
    generated = tmp_path / "generated"
    builtin.mkdir()
    generated.mkdir()
    _make_fn_dir(builtin, "foo")
    _make_fn_dir(generated, "bar")
    return FunctionManager(builtin_fn_dir=builtin, generated_fn_dir=generated)


def test_discover_default_returns_both_sources(tmp_path: Path) -> None:
    fm = _fm(tmp_path)
    catalog = fm.discover_functions()
    assert set(catalog.keys()) == {"foo", "bar"}


def test_discover_builtin_only(tmp_path: Path) -> None:
    fm = _fm(tmp_path)
    catalog = fm.discover_functions(sources=("builtin",))
    assert set(catalog.keys()) == {"foo"}


def test_discover_generated_only(tmp_path: Path) -> None:
    fm = _fm(tmp_path)
    catalog = fm.discover_functions(sources=("generated",))
    assert set(catalog.keys()) == {"bar"}


def test_discover_empty_sources(tmp_path: Path) -> None:
    fm = _fm(tmp_path)
    assert fm.discover_functions(sources=()) == {}


def test_render_summary_respects_sources(tmp_path: Path) -> None:
    fm = _fm(tmp_path)
    full = fm.render_functions_summary()
    builtin_only = fm.render_functions_summary(sources=("builtin",))
    generated_only = fm.render_functions_summary(sources=("generated",))
    assert "foo" in full and "bar" in full
    assert "foo" in builtin_only and "bar" not in builtin_only
    assert "bar" in generated_only and "foo" not in generated_only
    assert fm.render_functions_summary(sources=()) == ""


def test_builtin_wins_on_name_collision(tmp_path: Path) -> None:
    builtin = tmp_path / "pre_built"
    generated = tmp_path / "generated"
    builtin.mkdir()
    generated.mkdir()
    _make_fn_dir(builtin, "shared")
    _make_fn_dir(generated, "shared")
    (generated / "shared" / "fn.md").write_text(
        "# shared\n## Output\nGENERATED COPY\n"
    )
    fm = FunctionManager(builtin_fn_dir=builtin, generated_fn_dir=generated)
    catalog = fm.discover_functions()
    assert "GENERATED COPY" not in catalog["shared"]["fn_md"]
