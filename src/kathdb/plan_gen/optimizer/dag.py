"""Read-only helpers over the atomic plan DAG (``FAONode`` tree).

An *atom* is one plan node emitted by the parser; the synthetic ``logical_plan`` root
and ``input_relation`` leaves are not atoms.
"""

from __future__ import annotations

from ..plan_node import FAONode

__all__ = ["SYNTHETIC_OPS", "all_atoms_of", "atomic_edges_of", "nodes_by_op"]


SYNTHETIC_OPS = frozenset({"input_relation", "logical_plan"})


def all_atoms_of(root: FAONode) -> tuple[str, ...]:
    """Atom op-names in pre-order, each once, synthetic placeholders skipped."""
    out: list[str] = []
    seen: set[str] = set()
    for n in root.iter_preorder():
        if n is root or n.op in SYNTHETIC_OPS:
            continue
        if not n.op or n.op in seen:
            continue
        seen.add(n.op)
        out.append(n.op)
    return tuple(out)


def atomic_edges_of(root: FAONode) -> frozenset[tuple[str, str]]:
    """Data-flow edges between atoms: ``(u, v)`` means ``v`` consumes an output of ``u``."""
    edges: set[tuple[str, str]] = set()
    for n in root.iter_preorder():
        if n is root or n.op in SYNTHETIC_OPS or not n.op:
            continue
        for c in n.children:
            if c.op in SYNTHETIC_OPS or not c.op:
                continue
            edges.add((c.op, n.op))
    return frozenset(edges)


def nodes_by_op(root: FAONode) -> dict[str, FAONode]:
    """Map atom op-name -> first matching node in pre-order (synthetic skipped)."""
    out: dict[str, FAONode] = {}
    for n in root.iter_preorder():
        if n is root or n.op in SYNTHETIC_OPS or not n.op:
            continue
        if n.op not in out:
            out[n.op] = n
    return out
