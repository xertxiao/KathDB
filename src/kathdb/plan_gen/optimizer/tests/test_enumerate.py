"""Tests for the convex-partition enumerator (list_rank full search space)."""

from __future__ import annotations

from ..dag import all_atoms_of, nodes_by_op
from ..enumerate import (
    build_atom_index,
    enumerate_convex_partitions,
    precompute_valid_groups,
)
from ..rewriter import verify_fusable_subset
from ._fixtures import diamond, linear_chain


def _is_valid(partition, root, atom_set):
    by_op = nodes_by_op(root)
    covered: set[str] = set()
    for g in partition:
        assert not (covered & g)
        covered |= set(g)
        if len(g) >= 2:
            assert verify_fusable_subset([by_op[m] for m in g], root)
    return covered == atom_set


def test_enumerate_linear_chain_three():
    # A->B->C path: convex partitions = contiguous segmentations = 2^(n-1) = 4.
    root = linear_chain(
        "A", "B", "C", op_kinds={"A": "SEMANTIC", "B": "SEMANTIC", "C": "SEMANTIC"}
    )
    parts, capped = enumerate_convex_partitions(root, max_group_size=5)
    assert capped is False
    assert len(parts) == 4
    # No duplicate partitions, each a valid convex cover.
    keys = {frozenset(p) for p in parts}
    assert len(keys) == len(parts)
    for p in parts:
        assert _is_valid(p, root, {"A", "B", "C"})


def test_enumerate_respects_max_group_size():
    root = linear_chain(
        "A", "B", "C", op_kinds={"A": "SEMANTIC", "B": "SEMANTIC", "C": "SEMANTIC"}
    )
    parts, _ = enumerate_convex_partitions(root, max_group_size=1)
    # Only singletons allowed -> exactly one partition (all atoms unfused).
    assert len(parts) == 1
    assert all(len(g) == 1 for g in parts[0])


def test_enumerate_hard_cap_fires():
    root = linear_chain(
        "A", "B", "C", op_kinds={"A": "SEMANTIC", "B": "SEMANTIC", "C": "SEMANTIC"}
    )
    parts, capped = enumerate_convex_partitions(root, max_group_size=5, hard_cap=2)
    assert capped is True
    assert len(parts) >= 2


def test_enumerate_prunes_relational_only_groups():
    # A[S] -> B[R] -> C[R]: a fused group with no semantic op can't cut tokens, so {B,C}
    # is never formed. The 4 contiguous segmentations lose {A}{B,C} -> 3 partitions, and
    # every fused group must contain the semantic atom A.
    root = linear_chain("A", "B", "C", op_kinds={"A": "SEMANTIC"})  # B,C relational
    parts, capped = enumerate_convex_partitions(root, max_group_size=5)
    assert capped is False
    keys = {frozenset(p) for p in parts}
    assert frozenset({frozenset({"A"}), frozenset({"B", "C"})}) not in keys
    assert len(parts) == 3
    for p in parts:
        for g in p:
            if len(g) >= 2:
                assert "A" in g  # no relational-only fused group survives


def test_enumerate_diamond_excludes_nonconvex():
    # In a diamond A->{B,C}->D, {A,D} is NOT convex; no partition may fuse it.
    root = diamond(
        op_kinds={"A": "SEMANTIC", "B": "SEMANTIC", "C": "SEMANTIC", "D": "SEMANTIC"}
    )
    parts, capped = enumerate_convex_partitions(root, max_group_size=5)
    assert not capped
    for p in parts:
        assert _is_valid(p, root, {"A", "B", "C", "D"})
        for g in p:
            assert g != frozenset({"A", "D"})


def test_enumerate_uncapped_group_size_allows_whole_chain():
    root = linear_chain("A", "B", "C", op_kinds={"A": "SEMANTIC"})
    parts, _ = enumerate_convex_partitions(root, max_group_size=None)
    assert (frozenset({"A", "B", "C"}),) in parts


# ---------------------------------------------------------------------------
# atom index / valid-group lattice
# ---------------------------------------------------------------------------


def test_build_atom_index_marks_semantic_atoms():
    root = linear_chain("A", "B", "C", op_kinds={"A": "SEMANTIC", "B": "SEMANTIC"})
    atoms, atom_to_bit, sem_mask, edge_pairs = build_atom_index(root)
    assert set(atoms) == {"A", "B", "C"}
    assert sem_mask & (1 << atom_to_bit["A"])
    assert sem_mask & (1 << atom_to_bit["B"])
    assert not sem_mask & (1 << atom_to_bit["C"])
    assert edge_pairs


def test_precompute_valid_groups_contains_singletons_and_chain():
    root = linear_chain("A", "B", "C")
    atoms, atom_to_bit, _, _ = build_atom_index(root)
    valid = precompute_valid_groups(root, atoms)
    for a in atoms:
        assert (1 << atom_to_bit[a]) in valid
    full = (1 << len(atoms)) - 1
    assert full in valid  # the whole convex chain is fusable


def test_precompute_valid_groups_cap_excludes_oversized():
    root = linear_chain("A", "B", "C")  # convex chain: {A,B,C} is fusable
    atoms, *_ = build_atom_index(root)
    full = precompute_valid_groups(root, atoms)
    capped = precompute_valid_groups(root, atoms, max_group_size=2)
    assert any(bin(m).count("1") == 3 for m in full)  # triple present uncapped
    assert all(bin(m).count("1") <= 2 for m in capped)  # never present capped
