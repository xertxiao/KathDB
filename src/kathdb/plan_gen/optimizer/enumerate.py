"""Enumerate every legal convex partition of an atomic plan.

A group is convex (no non-member on a data-flow path between two members) and holds
at most ``max_group_size`` atoms; a partition is a disjoint cover by such groups.
Enumeration assigns the lowest uncovered atom to each valid group containing it, so
every partition is reached once; ``hard_cap`` bounds the output.
"""

from __future__ import annotations

from ..plan_node import FAONode
from .dag import all_atoms_of, atomic_edges_of, nodes_by_op
from .rewriter import verify_fusable_subset
from .types import Partition

__all__ = ["build_atom_index", "enumerate_convex_partitions", "precompute_valid_groups"]


def build_atom_index(
    root: FAONode,
) -> tuple[tuple[str, ...], dict[str, int], int, list[tuple[int, int]]]:
    """Map atoms to bit positions.

    Returns ``(atoms, atom_to_bit, sem_mask, edge_pairs)`` where ``sem_mask`` has a
    bit set for every SEMANTIC atom and ``edge_pairs`` are the data-flow edges in
    bit coordinates.
    """
    atoms = all_atoms_of(root)
    atom_to_bit = {name: i for i, name in enumerate(atoms)}

    sem_mask = 0
    for node in root.iter_preorder():
        if node.op in atom_to_bit and (node.op_kind or "").startswith("SEMANTIC"):
            sem_mask |= 1 << atom_to_bit[node.op]

    edge_pairs = [
        (atom_to_bit[u], atom_to_bit[v])
        for u, v in atomic_edges_of(root)
        if u in atom_to_bit and v in atom_to_bit
    ]
    return atoms, atom_to_bit, sem_mask, edge_pairs


def precompute_valid_groups(
    root: FAONode, atoms: tuple[str, ...], max_group_size: int | None = None
) -> set[int]:
    """Bitmasks of every fusable (convex) group of atoms, singletons included."""
    n = len(atoms)
    by_op = nodes_by_op(root)
    valid: set[int] = {1 << i for i in range(n)}

    for mask in range(3, 1 << n):
        pop = bin(mask).count("1")
        if pop < 2:
            continue
        if max_group_size is not None and pop > max_group_size:
            continue
        members = [by_op[atoms[i]] for i in range(n) if mask & (1 << i)]
        if verify_fusable_subset(members, root):
            valid.add(mask)

    return valid


def _lowest_uncovered_bit(covered: int, full_mask: int) -> int:
    low = (~covered) & full_mask
    return (low & -low).bit_length() - 1


def enumerate_convex_partitions(
    root: FAONode,
    *,
    max_group_size: int | None = None,
    hard_cap: int = 200,
) -> tuple[list[Partition], bool]:
    """Return ``(partitions, capped)``.

    ``partitions`` lists every legal convex partition whose groups have at most
    ``max_group_size`` atoms (``None`` = no cap). ``capped`` is True when
    enumeration stopped at ``hard_cap``, i.e. the list is incomplete.

    A multi-atom group with no SEMANTIC atom is never formed: it has no model call
    to share, skip, or push a filter into, so fusing it cannot cut execution cost.
    """
    atoms, _atom_to_bit, sem_mask, _edges = build_atom_index(root)
    n = len(atoms)
    if n == 0:
        return [], False
    full_mask = (1 << n) - 1
    valid = precompute_valid_groups(root, atoms, max_group_size=max_group_size)
    valid = {g for g in valid if (g & (g - 1)) == 0 or (g & sem_mask)}

    groups_by_bit: list[list[int]] = [[] for _ in range(n)]
    for g in valid:
        for b in range(n):
            if g & (1 << b):
                groups_by_bit[b].append(g)

    out: list[Partition] = []
    capped = False

    def _group_frozen(g: int) -> frozenset[str]:
        return frozenset(atoms[i] for i in range(n) if g & (1 << i))

    def rec(covered: int, acc: list[frozenset[str]]) -> None:
        nonlocal capped
        if capped:
            return
        if covered == full_mask:
            out.append(tuple(acc))
            if len(out) >= hard_cap:
                capped = True
            return
        b = _lowest_uncovered_bit(covered, full_mask)
        for g in groups_by_bit[b]:
            if g & covered:
                continue
            acc.append(_group_frozen(g))
            rec(covered | g, acc)
            acc.pop()
            if capped:
                return

    rec(0, [])
    return out, capped
