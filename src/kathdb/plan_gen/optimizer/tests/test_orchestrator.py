"""Tests for the optimizer entry point (selection -> rewrite -> trace)."""

from __future__ import annotations

from ..orchestrator import fused_op_name, run_optimizer
from ..types import GroupingConfig
from ._fixtures import linear_chain


class _FakeSelector:
    def __init__(self, proposed, n_candidates=1):
        self._proposed = proposed
        self.n_candidates = n_candidates

    def select(self, root):
        return self._proposed


def test_singleton_dag_short_circuits():
    root = linear_chain("A")
    rewritten, trace = run_optimizer(
        root=root, cfg=GroupingConfig(), selector=_FakeSelector(None, 0)
    )
    assert rewritten is root
    assert trace.n_atoms == 1
    assert trace.short_circuit_reason == "n_atoms_le_1"


def test_fused_group_is_applied_with_canonical_name():
    root = linear_chain("A", "B", "C", op_kinds={"A": "SEMANTIC", "B": "SEMANTIC"})
    rewritten, trace = run_optimizer(
        root=root,
        cfg=GroupingConfig(rank_k=4),
        selector=_FakeSelector((frozenset({"A", "B"}), frozenset({"C"})), 3),
    )
    fused = [n for n in rewritten.iter_preorder() if n.type == "GROUPED"]
    assert len(fused) == 1
    assert fused[0].op == fused_op_name({"A", "B"})
    assert set(fused[0].member_atoms) == {"A", "B"}
    assert root.to_dict() == trace.atomic_dag  # input tree untouched
    assert trace.n_candidates == 3
    assert trace.rank_k == 4
    assert [g["members"] for g in trace.fused_groups] == [["A", "B"]]


def test_no_fusion_returns_atomic_plan_with_reason():
    root = linear_chain("A", "B", "C")
    rewritten, trace = run_optimizer(
        root=root, cfg=GroupingConfig(), selector=_FakeSelector(None, 0)
    )
    assert rewritten is root
    assert trace.short_circuit_reason == "no_fusion_candidates"
    assert trace.fused_groups == []


def test_fused_op_name_is_order_independent_and_injective():
    assert fused_op_name(["A", "B"]) == fused_op_name(["B", "A"])
    assert fused_op_name(["A", "B"]) != fused_op_name(["A", "C"])
    assert fused_op_name(["step_0", "step_1"]) != fused_op_name(["step_0_step", "1"])


def test_trace_to_dict_shape():
    root = linear_chain("A", "B", op_kinds={"A": "SEMANTIC", "B": "SEMANTIC"})
    _, trace = run_optimizer(
        root=root,
        cfg=GroupingConfig(),
        selector=_FakeSelector((frozenset({"A", "B"}),), 1),
    )
    d = trace.to_dict()
    assert set(d) == {
        "n_atoms",
        "rank_k",
        "max_group_size",
        "n_candidates",
        "fused_groups",
        "short_circuit_reason",
        "atomic_dag",
        "final_dag",
    }


def test_ranker_rewrite_becomes_the_fused_rationale():
    from kathdb.plan_gen.optimizer.orchestrator import _proposals_from_partition

    partition = (frozenset({"a", "b"}), frozenset({"c"}))
    props = _proposals_from_partition(
        partition, {frozenset({"a", "b"}): "call the model per key and stop at the first hit"}
    )
    assert len(props) == 1
    assert props[0].rationale == "call the model per key and stop at the first hit"
    assert _proposals_from_partition(partition)[0].rationale == ""
