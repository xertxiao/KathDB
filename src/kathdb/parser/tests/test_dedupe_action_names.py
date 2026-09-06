"""``_dedupe_action_names`` suffixes repeated action names so every ``FAONode.op`` is unique."""

from __future__ import annotations

from kathdb.parser.action import Action
from kathdb.parser.parser import ActionNLParser
from kathdb.plan_gen.plan_node import build_fao_dag


def _action(name: str, inputs: list[str] | None = None, output: str = "x", **kw):
    return Action(
        name=name, action="", inputs=inputs or ["t"], output=output, **kw
    )


def test_dedupe_identity_when_all_names_unique():
    sketch = [
        _action("filter_by_price", ["t"], "a"),
        _action("extract_brand_name", ["a"], "b"),
        _action("classify_image", ["b"], "c"),
    ]
    out = ActionNLParser._dedupe_action_names(sketch)
    assert [a.name for a in out] == [
        "filter_by_price",
        "extract_brand_name",
        "classify_image",
    ]
    # Helper does not mutate caller's objects.
    assert sketch[0].name == "filter_by_price"


def test_dedupe_suffixes_repeats_in_order():
    sketch = [
        _action("count_by_group", ["t"], "a"),
        _action("count_by_group", ["t"], "b"),
        _action("join_tables", ["a", "b"], "c"),
        _action("count_by_group", ["c"], "d"),
    ]
    out = ActionNLParser._dedupe_action_names(sketch)
    assert [a.name for a in out] == [
        "count_by_group",
        "count_by_group_2",
        "join_tables",
        "count_by_group_3",
    ]


def test_dedupe_handles_pre_existing_suffix_collision():
    """A pre-existing ``foo_2`` still yields unique names (``foo``, ``foo_2``, ``foo_2_2``)."""
    sketch = [_action("foo"), _action("foo"), _action("foo_2")]
    out = ActionNLParser._dedupe_action_names(sketch)
    names = [a.name for a in out]
    assert len(set(names)) == 3
    assert names == ["foo", "foo_2", "foo_2_2"]


def test_dedupe_preserves_other_fields():
    sketch = [
        _action("agg", ["t"], "a", op_kind="RELATIONAL-GROUP_BY_AGGREGATE", output_type="dataframe"),
        _action("agg", ["t"], "b", op_kind="RELATIONAL-GROUP_BY_AGGREGATE", output_type="dataframe"),
    ]
    out = ActionNLParser._dedupe_action_names(sketch)
    assert out[0].op_kind == "RELATIONAL-GROUP_BY_AGGREGATE"
    assert out[1].op_kind == "RELATIONAL-GROUP_BY_AGGREGATE"
    assert out[0].output == "a"
    assert out[1].output == "b"
    assert out[1].name == "agg_2"


def test_dedupe_skips_empty_names():
    sketch = [
        _action("", ["t"], "a"),
        _action("x", ["a"], "b"),
    ]
    out = ActionNLParser._dedupe_action_names(sketch)
    assert out[0].name == ""
    assert out[1].name == "x"


def test_build_fao_dag_after_dedupe_produces_unique_op_names():
    """After dedup, ``build_fao_dag`` yields distinct ``op`` names for every atom."""
    sketch = [
        _action(
            "filter_by_price", ["styles"], "budget",
            op_kind="RELATIONAL-FILTER", output_type="dataframe",
        ),
        _action(
            "count_by_group", ["budget"], "total_counts",
            op_kind="RELATIONAL-GROUP_BY_AGGREGATE", output_type="dataframe",
        ),
        _action(
            "count_by_group", ["budget"], "visual_counts",
            op_kind="RELATIONAL-GROUP_BY_AGGREGATE", output_type="dataframe",
        ),
        _action(
            "join_tables", ["total_counts", "visual_counts"], "combined",
            op_kind="RELATIONAL-JOIN", output_type="dataframe",
        ),
    ]
    deduped = ActionNLParser._dedupe_action_names(sketch)
    plan_entries = [
        {
            "name": a.name,
            "input": a.inputs,
            "output": [a.output],
            "op_kind": a.op_kind,
            "output_type": a.output_type,
        }
        for a in deduped
    ]
    root = build_fao_dag(plan_entries, ["styles"])
    atom_ops = [
        n.op
        for n in root.iter_preorder()
        if n is not root and n.op not in {"input_relation", "logical_plan"}
    ]
    assert len(atom_ops) == len(set(atom_ops)), (
        f"Atom op-names must be unique post-dedup: {atom_ops}"
    )
    assert "count_by_group" in atom_ops
    assert "count_by_group_2" in atom_ops
