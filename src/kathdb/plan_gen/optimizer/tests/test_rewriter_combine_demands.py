"""``_combine_consumer_demands``: keep demands per escaping output, drop internal
ones, member-level rule for untagged demands."""

from __future__ import annotations

from ...plan_node import FAONode
from ..rewriter import _combine_consumer_demands


def _demand(consumer: str, output_relation: str | None, *cols: str) -> dict:
    d: dict = {
        "consumer": consumer,
        "required_columns": [
            {"name": c, "dtype": "VARCHAR", "reason": "r"} for c in cols
        ],
        "value_constraints": [],
    }
    if output_relation is not None:
        d["output_relation"] = output_relation
    return d


def test_combine_keeps_per_output_demands_tagged():
    b = FAONode(op="B", outputs=["B_out"])
    c = FAONode(op="C", outputs=["C_out"])
    b.consumer_demands = [_demand("D", "B_out", "x")]
    c.consumer_demands = [_demand("D", "C_out", "y")]

    combined = _combine_consumer_demands([b, c], ["B_out", "C_out"])

    by_rel = {d["output_relation"]: d for d in combined}
    assert set(by_rel) == {"B_out", "C_out"}
    assert by_rel["B_out"]["required_columns"][0]["name"] == "x"
    assert by_rel["C_out"]["required_columns"][0]["name"] == "y"


def test_combine_drops_demand_on_internal_output():
    # B produces B_out (escapes) and B_internal (consumed only inside the group).
    b = FAONode(op="B", outputs=["B_out", "B_internal"])
    b.consumer_demands = [
        _demand("D", "B_out", "x"),  # external → keep
        _demand("C", "B_internal", "secret"),  # internal → drop
    ]

    combined = _combine_consumer_demands([b], ["B_out"])

    assert [d["output_relation"] for d in combined] == ["B_out"]
    assert all(
        col["name"] != "secret"
        for d in combined
        for col in d["required_columns"]
    )


def test_combine_untagged_uses_member_escape_rule():
    # No output_relation → fall back to "kept iff a member output escapes".
    escaping = FAONode(op="B", outputs=["B_out"])
    escaping.consumer_demands = [_demand("D", None, "x")]
    internal = FAONode(op="C", outputs=["C_internal"])
    internal.consumer_demands = [_demand("B", None, "y")]

    combined = _combine_consumer_demands([escaping, internal], ["B_out"])

    names = {col["name"] for d in combined for col in d["required_columns"]}
    assert names == {"x"}  # internal member's demand excluded
