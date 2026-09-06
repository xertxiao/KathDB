"""Tests for the atomic-DAG helpers."""

from __future__ import annotations

from ..dag import all_atoms_of, atomic_edges_of, nodes_by_op
from ._fixtures import diamond, linear_chain


def test_all_atoms_skips_synthetic():
    root = linear_chain("A", "B", "C")
    assert set(all_atoms_of(root)) == {"A", "B", "C"}


def test_atomic_edges_linear_chain():
    root = linear_chain("A", "B", "C")
    assert atomic_edges_of(root) == frozenset({("A", "B"), ("B", "C")})


def test_atomic_edges_diamond():
    root = diamond()
    assert atomic_edges_of(root) == frozenset(
        {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    )


def test_nodes_by_op_skips_synthetic_and_keeps_first():
    root = diamond()
    by_op = nodes_by_op(root)
    assert set(by_op) == {"A", "B", "C", "D"}
    assert by_op["D"].outputs == ["D_out"]
