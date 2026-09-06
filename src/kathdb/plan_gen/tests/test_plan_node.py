"""FAONode serialization / pretty-print tests."""

from __future__ import annotations

import pytest

from kathdb.plan_gen.plan_node import FAONode, build_fao_dag


def test_op_kind_rewrite_roundtrip():
    n = FAONode(
        op="join_formal",
        inputs=["text_classified", "image_classified"],
        outputs=["joined"],
        op_kind="RELATIONAL",
        op_kind_rewrite={
            "from": "SEMANTIC",
            "to": "RELATIONAL",
            "rationale": "both inputs collapse to {Formal, Non-Formal}",
            "evidence": "text_classified.label and image_classified.label "
            "value_constraint = closed enum",
        },
    )
    n.add_child(FAONode(op="input_relation", outputs=["text_classified"]))

    raw = n.to_dict()
    assert raw["op_kind"] == "RELATIONAL"
    assert raw["op_kind_rewrite"]["from"] == "SEMANTIC"
    assert raw["op_kind_rewrite"]["to"] == "RELATIONAL"

    restored = FAONode.from_dict(raw)
    assert restored.op_kind == "RELATIONAL"
    assert restored.op_kind_rewrite == n.op_kind_rewrite

    rendered = restored.pretty()
    assert "op_kind_rewrite: SEMANTIC -> RELATIONAL" in rendered


def test_op_kind_rewrite_default_is_none_and_omitted_from_dict():
    n = FAONode(op="filter_simple", op_kind="RELATIONAL")
    assert n.op_kind_rewrite is None
    raw = n.to_dict()
    assert "op_kind_rewrite" not in raw
    assert "op_kind_rewrite" not in n.pretty()


def test_build_fao_dag_rejects_duplicate_output():
    # A duplicate output would silently misroute consumers; build_fao_dag must fail loud.
    entries = [
        {"name": "step_a", "input": ["src"], "output": ["result"]},
        {"name": "step_b", "input": ["src"], "output": ["result"]},
    ]
    with pytest.raises(ValueError, match="Duplicate output relation"):
        build_fao_dag(entries, ["src"])


def test_build_fao_dag_allows_unique_outputs():
    entries = [
        {"name": "step_a", "input": ["src"], "output": ["a_out"]},
        {"name": "step_b", "input": ["a_out"], "output": ["b_out"]},
    ]
    root = build_fao_dag(entries, ["src"])
    assert root.op == "logical_plan"
    ops = {n.op for n in root.iter_preorder()}
    assert {"step_a", "step_b"} <= ops
