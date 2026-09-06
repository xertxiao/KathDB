"""``build_fused_node`` parity with ``apply_merges``; diamond with a duplicated shared node."""

from __future__ import annotations

from ...plan_node import FAONode
from ..rewriter import apply_merges, build_fused_node
from ..types import MergeProposal
from ._fixtures import linear_chain


def test_build_fused_node_matches_apply_merges_io():
    root = linear_chain("A", "B", "C")
    by_op = {n.op: n for n in root.iter_preorder()}
    fused = build_fused_node([by_op["A"], by_op["B"]], root, fused_op_name="g_test")

    prop = MergeProposal(
        proposal_id="p",
        member_atoms=("A", "B"),
        fused_op_name="g_test",
        fused_description="",
        rationale="",
    )
    rewritten = apply_merges(root, [prop])
    in_tree = next(n for n in rewritten.iter_preorder() if n.op == "g_test")

    assert set(fused.inputs) == set(in_tree.inputs)
    assert set(fused.outputs) == set(in_tree.outputs)
    assert set(fused.member_atoms) == set(in_tree.member_atoms)
    assert fused.type == "GROUPED"


def test_build_fused_node_diamond_with_duplicated_shared_node():
    """With the shared `keep` duplicated into two instances, the fused node keeps
    `kept` internal and never emits a member as a child."""
    keep1 = FAONode(op="keep", inputs=["raw"], outputs=["kept"], children=[])
    keep2 = FAONode(op="keep", inputs=["raw"], outputs=["kept"], children=[])
    minid = FAONode(op="minid", inputs=["kept"], outputs=["minid_out"], children=[keep1])
    join = FAONode(
        op="join", inputs=["minid_out", "kept"], outputs=["joined"], children=[minid, keep2]
    )
    root = FAONode(op="logical_plan", children=[join])

    by_op = {n.op: n for n in root.iter_preorder()}  # picks ONE keep instance
    fused = build_fused_node(
        [by_op["keep"], by_op["minid"], by_op["join"]], root, fused_op_name="g_diamond"
    )

    assert all(c.op not in fused.member_atoms for c in fused.children)
    assert "kept" not in fused.inputs
    assert set(fused.inputs) == {"raw"}
    assert all(c.op != "keep" for c in fused.children)
