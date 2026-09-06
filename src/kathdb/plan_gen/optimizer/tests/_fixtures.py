"""Shared FAONode fixtures for optimizer unit tests."""

from __future__ import annotations

from ...plan_node import FAONode


def linear_chain(*ops: str, op_kinds: dict[str, str] | None = None) -> FAONode:
    """Build ``logical_plan`` → ops[-1] → ops[-2] → ... → ops[0].

    Each op reads its predecessor's output (``in_table`` for the head).
    ``op_kinds`` maps op-name → ``"SEMANTIC"`` or ``"RELATIONAL"``;
    omitted ops get ``None``.
    """
    op_kinds = op_kinds or {}
    nodes: list[FAONode] = []
    prev_out = "in_table"
    for op in ops:
        out = f"{op}_out"
        children = [nodes[-1]] if nodes else []
        nodes.append(
            FAONode(
                op=op,
                inputs=[prev_out],
                outputs=[out],
                children=children,
                op_kind=op_kinds.get(op),
            )
        )
        prev_out = out
    root = FAONode(op="logical_plan", children=[nodes[-1]] if nodes else [])
    return root


def diamond(op_kinds: dict[str, str] | None = None) -> FAONode:
    """Build the diamond A -> {B, C} -> D ({A, D} alone is non-convex)."""
    op_kinds = op_kinds or {}
    a = FAONode(
        op="A",
        inputs=["in_table"],
        outputs=["A_out"],
        children=[],
        op_kind=op_kinds.get("A"),
    )
    b = FAONode(
        op="B",
        inputs=["A_out"],
        outputs=["B_out"],
        children=[a],
        op_kind=op_kinds.get("B"),
    )
    c = FAONode(
        op="C",
        inputs=["A_out"],
        outputs=["C_out"],
        children=[a],
        op_kind=op_kinds.get("C"),
    )
    d = FAONode(
        op="D",
        inputs=["B_out", "C_out"],
        outputs=["D_out"],
        children=[b, c],
        op_kind=op_kinds.get("D"),
    )
    return FAONode(op="logical_plan", children=[d])
