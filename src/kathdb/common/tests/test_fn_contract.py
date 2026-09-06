"""fn_contract: parsing, derivation, validation, and no fn.md drift for the shipped example."""

from __future__ import annotations

from pathlib import Path


import pytest

from kathdb.common import fn_contract as fc
from kathdb.common.function_manager import FunctionManager

_SAMPLE = """
import pandas as pd

CONTRACT = {
    "purpose": "Toy op.",
    "params": {"df": "rows", "label": "a tag"},
    "sys_params": ["model"],
    "output": "same df",
    "example": "toy(df, label='x')",
    "use_when": "always",
}


def toy(df: pd.DataFrame, label: str = "", model: str = "m") -> pd.DataFrame:
    return df
"""


def test_parse_and_derive():
    p = fc.parse_fn_source(_SAMPLE, "toy")
    assert [pp.name for pp in p.params] == ["df", "label", "model"]
    assert fc.derive_df_params(p) == ("df",)
    assert fc.validate_contract(p) == []


def test_signature_render():
    p = fc.parse_fn_source(_SAMPLE, "toy")
    assert fc.format_signature(p) == (
        "toy(df: pd.DataFrame, label: str = '', model: str = 'm') -> pd.DataFrame"
    )


def test_variadic_df_yields_sentinel():
    code = (
        "import pandas as pd\n"
        "CONTRACT = {'purpose':'p','output':'o','params':{'prompt':'x'},'use_when':'w'}\n"
        "def j(prompt: str, **dfs: pd.DataFrame):\n    return None\n"
    )
    p = fc.parse_fn_source(code, "j")
    assert fc.derive_df_params(p) == ("**",)


def test_utility_allows_no_df():
    code = (
        "CONTRACT = {'utility': True, 'purpose':'p','output':'o',"
        "'params':{'prompts':'list'},'sys_params':['model'],'use_when':'w'}\n"
        "def b(prompts: list, model: str = 'm'):\n    return prompts\n"
    )
    p = fc.parse_fn_source(code, "b")
    assert fc.derive_df_params(p) == ()
    assert fc.validate_contract(p) == []


def test_validate_flags_unknown_and_undocumented():
    code = (
        "import pandas as pd\n"
        "CONTRACT = {'purpose':'p','output':'o','params':{'ghost':'x'},'use_when':'w'}\n"
        "def f(df: pd.DataFrame, real: str = ''):\n    return df\n"
    )
    p = fc.parse_fn_source(code, "f")
    issues = fc.validate_contract(p)
    assert any("ghost" in i for i in issues)
    assert any("real" in i for i in issues)


def test_example_fn_no_drift():
    """The shipped example's on-disk fn.md equals the rendered one; no spec.py."""
    import kathdb

    name = "sem_map"
    fn_dir = Path(kathdb.__file__).parent / "pre_built_fn" / "_example" / name
    assert fn_dir.is_dir()
    code = (fn_dir / "scripts" / "fn.py").read_text()
    parsed = fc.parse_fn_source(code, name)
    assert fc.validate_contract(parsed) == [], f"{name} contract issues"
    assert (fn_dir / "fn.md").read_text() == fc.render_fn_md(
        parsed
    ), f"{name} fn.md drift"
    assert not (fn_dir / "spec.py").exists(), f"{name} still has a spec.py"
