"""Convexity checks and the ``apply_merges`` splice must treat the duplicate
instances a ``to_dict``/``from_dict`` round trip creates for a diamond's shared
node as one logical node."""

from __future__ import annotations

from ...plan_node import FAONode
from ..rewriter import (
    _collect_nodes_by_op,
    apply_merges,
    compute_convex_closure,
    verify_fusable_subset,
)
from ..types import MergeProposal
from ._fixtures import diamond


def _roundtrip(root: FAONode) -> FAONode:
    return FAONode.from_dict(root.to_dict())


def _members(root: FAONode, *ops: str) -> list[FAONode]:
    by_op = _collect_nodes_by_op(root)
    return [by_op[o] for o in ops]


def _proposal(*ops: str, fused_op_name: str = "fused") -> MergeProposal:
    return MergeProposal(
        proposal_id="p_test",
        member_atoms=tuple(ops),
        fused_op_name=fused_op_name,
        fused_description="fused for test",
        rationale="test",
    )


def test_roundtrip_duplicates_the_shared_diamond_node():
    root = _roundtrip(diamond())
    assert sum(1 for n in root.iter_preorder() if n.op == "A") == 2


def test_non_convex_group_rejected_on_shared_object_dag():
    root = diamond()
    # C sits on the A -> D path outside the group.
    assert not verify_fusable_subset(_members(root, "A", "B", "D"), root)


def test_non_convex_group_rejected_after_roundtrip():
    """An id-only walk would miss the duplicate instance."""
    root = _roundtrip(diamond())
    assert not verify_fusable_subset(_members(root, "A", "B", "D"), root)


def test_full_diamond_accepted_after_roundtrip():
    root = _roundtrip(diamond())
    assert verify_fusable_subset(_members(root, "A", "B", "C", "D"), root)


def test_sibling_pair_accepted_after_roundtrip():
    root = _roundtrip(diamond())
    assert verify_fusable_subset(_members(root, "B", "C"), root)


def test_convex_closure_dedupes_duplicate_instances():
    root = _roundtrip(diamond())
    closure = compute_convex_closure(_members(root, "A", "D"), root)
    assert sorted(n.op for n in closure) == ["B", "C"]


def test_apply_merges_splices_out_duplicate_member_instances():
    root = _roundtrip(diamond())
    out = apply_merges(root, [_proposal("A", "B", "C", "D")])
    ops = [n.op for n in out.iter_preorder()]
    # No member instance — including the duplicated A — may survive.
    for member in ("A", "B", "C", "D"):
        assert member not in ops
    fused = next(n for n in out.iter_preorder() if n.op == "fused")
    assert fused.inputs == ["in_table"]
    assert fused.outputs == ["D_out"]


def test_apply_merges_sibling_group_keeps_one_shared_producer():
    root = _roundtrip(diamond())
    out = apply_merges(root, [_proposal("B", "C", fused_op_name="fused_bc")])
    fused = next(n for n in out.iter_preorder() if n.op == "fused_bc")
    # The duplicated external producer A must be deduped to one instance.
    assert [c.op for c in fused.children] == ["A"]
    assert sorted(fused.outputs) == ["B_out", "C_out"]
