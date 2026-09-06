"""Plan-tree rewriter: collapse a convex subgraph of atoms into one fused node.

A subgraph is convex iff no non-member sits on a data-flow path between two members
(fusing it keeps the DAG acyclic). The fused node's ``inputs`` are member inputs not
produced inside the group, ``outputs`` are member outputs consumed outside the group
(or not at all), ``children`` are the external producers, ``selected_functions`` the
members' union. A fork/rejoin (diamond) serialized as a tree duplicates the shared
node, so membership and consumer lookups are op-name-aware, not only id-based.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Sequence

from ...common.logger import get_logger
from ..plan_node import FAONode
from .types import MergeProposal

logger = get_logger(__name__)

__all__ = [
    "apply_merges",
    "build_fused_node",
    "verify_fusable_subset",
    "compute_convex_closure",
]


# ---------------------------------------------------------------------------
# Tree introspection helpers
# ---------------------------------------------------------------------------


def _collect_nodes_by_op(root: FAONode) -> dict[str, FAONode]:
    """Map op-name -> first matching node found in pre-order."""
    out: dict[str, FAONode] = {}
    for n in root.iter_preorder():
        if n.op and n.op not in out:
            out[n.op] = n
    return out


def _build_consumer_map(root: FAONode) -> dict[Any, list[FAONode]]:
    """Map node-id (and ``("op", op_name)``, aggregating duplicate instances) -> consumers.

    ``node.children`` are upstream producers, so a consumer is the parent.
    """
    consumers: dict[Any, list[FAONode]] = defaultdict(list)
    for n in root.iter_preorder():
        for c in n.children:
            consumers[id(c)].append(n)
            if c.op:
                consumers[("op", c.op)].append(n)
    return consumers


def _atom_nodes_by_id(root: FAONode) -> dict[int, FAONode]:
    """Map node-id -> FAONode for every atom (skip synthetic placeholders)."""
    out: dict[int, FAONode] = {}
    for n in root.iter_preorder():
        if n is root or n.op in {"input_relation", "logical_plan"}:
            continue
        out[id(n)] = n
    return out


def _strict_ancestors_and_descendants(
    members: Sequence[FAONode],
    root: FAONode,
) -> tuple[dict[int, FAONode], dict[int, FAONode]]:
    """Return (ancestors, descendants) of the member set as id -> node maps.

    Ancestors walk ``.children`` (upstream); descendants walk consumer edges, both
    id-keyed and op-keyed so duplicate instances of a node are reached. The maps may
    contain member ids when members are inter-reachable.
    """
    consumer_map = _build_consumer_map(root)

    descendants: dict[int, FAONode] = {}
    stack: list[FAONode] = list(members)
    while stack:
        cur = stack.pop()
        nxts = {id(c): c for c in consumer_map.get(id(cur), ())}
        if cur.op:
            for c in consumer_map.get(("op", cur.op), ()):
                nxts.setdefault(id(c), c)
        for nid, nxt in nxts.items():
            if nid in descendants:
                continue
            descendants[nid] = nxt
            stack.append(nxt)

    ancestors: dict[int, FAONode] = {}
    stack = list(members)
    while stack:
        cur = stack.pop()
        for c in cur.children:
            if id(c) in ancestors:
                continue
            ancestors[id(c)] = c
            stack.append(c)

    return ancestors, descendants


# ---------------------------------------------------------------------------
# Convexity check + convex closure
# ---------------------------------------------------------------------------


def verify_fusable_subset(
    members: Sequence[FAONode],
    root: FAONode,
) -> bool:
    """A subset is fusable iff fusing it keeps the DAG acyclic.

    That holds iff no non-member sits on a data-flow path between two
    members — i.e., no non-member is simultaneously a strict descendant
    of some member and a strict ancestor of another.
    """
    if len(members) < 2:
        return True
    ancestors, descendants = _strict_ancestors_and_descendants(members, root)
    return not _non_member_overlap(members, ancestors, descendants)


def _non_member_overlap(
    members: Sequence[FAONode],
    ancestors: dict[int, FAONode],
    descendants: dict[int, FAONode],
) -> dict[int, FAONode]:
    """Nodes that are both ancestor and descendant of the member set, minus members
    (membership is op-name-aware so duplicate member instances are not external)."""
    member_ids = {id(m) for m in members}
    member_ops = {m.op for m in members if m.op}
    return {
        nid: n
        for nid, n in descendants.items()
        if nid in ancestors
        and nid not in member_ids
        and not (n.op and n.op in member_ops)
    }


def compute_convex_closure(
    members: Sequence[FAONode],
    root: FAONode,
) -> list[FAONode]:
    """Return the non-member atoms on a path between two members, i.e. the atoms
    that must be added to make ``members`` convex; empty when already convex.
    Topologically sorted.
    """
    if len(members) < 2:
        return []
    ancestors, descendants = _strict_ancestors_and_descendants(members, root)
    intermediate = _non_member_overlap(members, ancestors, descendants)
    if not intermediate:
        return []
    atoms_by_id = _atom_nodes_by_id(root)
    # Dedup duplicate instances of one logical node by op-name.
    seen_ops: set[str] = set()
    intermediates: list[FAONode] = []
    for nid, n in intermediate.items():
        if nid not in atoms_by_id:
            continue
        if n.op:
            if n.op in seen_ops:
                continue
            seen_ops.add(n.op)
        intermediates.append(n)
    return _topo_sort_members(intermediates)


# ---------------------------------------------------------------------------
# Topological sort restricted to members
# ---------------------------------------------------------------------------


def _topo_sort_members(members: Sequence[FAONode]) -> list[FAONode]:
    """Kahn's-algorithm topological sort restricted to member-only edges.

    Edge m_i -> m_j iff m_j depends on m_i (i.e., m_i appears in
    ``m_j.children`` in LP convention). Tie-broken alphabetically by
    ``op`` name for deterministic output.
    """
    member_set = {id(m) for m in members}
    member_by_id = {id(m): m for m in members}

    deps: dict[int, set[int]] = {id(m): set() for m in members}
    for m in members:
        for c in m.children:
            if id(c) in member_set and id(c) != id(m):
                deps[id(m)].add(id(c))

    remaining = {mid: set(d) for mid, d in deps.items()}
    result: list[FAONode] = []
    while remaining:
        ready = [mid for mid, d in remaining.items() if not d]
        if not ready:
            ready = list(remaining.keys())
        ready.sort(key=lambda mid: member_by_id[mid].op)
        chosen = ready[0]
        result.append(member_by_id[chosen])
        del remaining[chosen]
        for d in remaining.values():
            d.discard(chosen)
    return result


# ---------------------------------------------------------------------------
# Fused-node I/O computation
# ---------------------------------------------------------------------------


def _compute_fused_io(
    ordered_members: Sequence[FAONode],
    consumer_map: dict[int, list[FAONode]],
) -> tuple[list[str], list[str], list[FAONode]]:
    """Return (external_inputs, external_outputs, external_producers).

    ``ordered_members`` should be in topological-within-subgraph order so
    the deduped lists reflect data flow.
    """
    member_ids = {id(m) for m in ordered_members}
    # Membership by id OR op-name: a diamond duplicates a member into two instances.
    member_ops = {m.op for m in ordered_members if m.op}

    def _is_member(n: FAONode) -> bool:
        return id(n) in member_ids or (bool(n.op) and n.op in member_ops)

    produced_by_members: set[str] = set()
    for m in ordered_members:
        for o in m.outputs:
            if o:
                produced_by_members.add(o)

    seen_in: set[str] = set()
    inputs: list[str] = []
    for m in ordered_members:
        for name in m.inputs:
            if not name or name in seen_in:
                continue
            if name in produced_by_members:
                continue
            seen_in.add(name)
            inputs.append(name)

    seen_out: set[str] = set()
    outputs: list[str] = []
    for m in ordered_members:
        for name in m.outputs:
            if not name or name in seen_out:
                continue
            # Consumers across all same-op instances, else a boundary output is dropped.
            _cand = {id(c): c for c in consumer_map.get(id(m), ())}
            if m.op:
                for c in consumer_map.get(("op", m.op), ()):
                    _cand.setdefault(id(c), c)
            consumers_of_name = [c for c in _cand.values() if name in c.inputs]
            external_consumer_exists = any(not _is_member(c) for c in consumers_of_name)
            if external_consumer_exists or not consumers_of_name:
                seen_out.add(name)
                outputs.append(name)

    seen_prod: set[int] = set()
    seen_prod_ops: set[str] = set()
    producers: list[FAONode] = []
    for m in ordered_members:
        for c in m.children:
            if _is_member(c):
                continue
            # A child producing a relation the group already produces is internal;
            # externalizing it would execute it twice.
            if any(o in produced_by_members for o in (c.outputs or ()) if o):
                continue
            # Duplicate instances of one external producer count once.
            if id(c) in seen_prod or (c.op and c.op in seen_prod_ops):
                continue
            seen_prod.add(id(c))
            if c.op:
                seen_prod_ops.add(c.op)
            producers.append(c)

    return inputs, outputs, producers


def _combine_selected_functions(members: Sequence[FAONode]) -> list[str]:
    """Union of members' selected_functions in the given member order, deduped."""
    seen: set[str] = set()
    out: list[str] = []
    for m in members:
        for fn in m.selected_functions or ():
            if fn and fn not in seen:
                seen.add(fn)
                out.append(fn)
    return out


def _combine_consumer_demands(
    ordered_members: Sequence[FAONode],
    fused_outputs: Sequence[str],
) -> list[dict]:
    """Collect consumer_demands targeting an output that escapes the subgraph.

    A demand tagged with ``output_relation`` is kept iff that relation is a fused
    output; an untagged demand is kept iff any of its member's outputs escape.
    """
    fused_output_set = {o for o in fused_outputs if o}
    seen_keys: set[str] = set()
    out: list[dict] = []
    for m in ordered_members:
        member_output_escapes = any(o in fused_output_set for o in m.outputs)
        for d in m.consumer_demands or ():
            if isinstance(d, dict) and d.get("output_relation"):
                if d["output_relation"] not in fused_output_set:
                    continue
            elif not member_output_escapes:
                continue
            key = repr(sorted(d.items())) if isinstance(d, dict) else repr(d)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(dict(d) if isinstance(d, dict) else d)
    return out


# ---------------------------------------------------------------------------
# Op-name uniqueness
# ---------------------------------------------------------------------------


def _unique_op_name(desired: str, taken: set[str]) -> str:
    """Pick an op name not already used in the rewrite pass."""
    if desired not in taken:
        return desired
    i = 2
    while f"{desired}_{i}" in taken:
        i += 1
    return f"{desired}_{i}"


# ---------------------------------------------------------------------------
# Fused-node construction
# ---------------------------------------------------------------------------


def build_fused_node(
    members: Sequence[FAONode],
    root: FAONode,
    *,
    fused_op_name: str,
    fused_description: str | None = None,
    rationale: str | None = None,
) -> FAONode:
    """Build one ``type="GROUPED"`` fused node for ``members`` without splicing it
    into the tree. ``members`` need not be sorted; ``root`` provides the consumer map.
    """
    ordered = _topo_sort_members(list(members))
    consumer_map = _build_consumer_map(root)
    fused_inputs, fused_outputs, fused_children = _compute_fused_io(
        ordered, consumer_map
    )
    fused_demands = _combine_consumer_demands(ordered, fused_outputs)
    fused_functions = _combine_selected_functions(ordered)

    member_descs = [(m.description or m.op or "").strip() for m in ordered]
    member_op_names = [m.op for m in ordered]

    # A member leaking through as an external producer would execute twice: fail loud.
    _member_op_set = {n for n in member_op_names if n}
    _leaked = sorted({c.op for c in fused_children if c.op in _member_op_set})
    if _leaked:
        raise ValueError(
            f"build_fused_node({fused_op_name}): fused child(ren) {_leaked} are also "
            f"group members — inconsistent expansion that would double-execute them. "
            "Likely a fork/rejoin (diamond) member whose duplicate instance leaked as an "
            "external producer; see _compute_fused_io membership handling."
        )

    summary = (fused_description or f"Fused: {' + '.join(member_op_names)}").strip()

    return FAONode(
        op=fused_op_name,
        description=summary,
        inputs=fused_inputs,
        outputs=fused_outputs,
        children=list(fused_children),
        selected_functions=fused_functions,
        type="GROUPED",
        consumer_demands=fused_demands,
        member_atoms=member_op_names,
        member_descriptions=member_descs,
        merge_rationale=(rationale or "").strip() or None,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_merges(
    root: FAONode,
    selected: Sequence[MergeProposal],
) -> FAONode:
    """Return a deep-copied tree with ``selected`` merges applied.

    Per proposal: locate members by op-name (skip if any is missing), reject
    non-convex subsets, build the fused node, and rewire every external consumer
    of a member to it (at most once per consumer). Idempotent.
    """
    out = copy.deepcopy(root)
    used_ops: set[str] = {n.op for n in out.iter_preorder() if n.op}

    for proposal in selected:
        by_op = _collect_nodes_by_op(out)
        members_raw = [by_op.get(name) for name in proposal.member_atoms]
        if any(m is None for m in members_raw):
            logger.info(
                "[grouping][rewrite] skipping %s: member(s) not found "
                "(already merged or stale).",
                proposal.proposal_id,
            )
            continue
        members: list[FAONode] = []
        seen_ids: set[int] = set()
        for m in members_raw:
            if id(m) in seen_ids:
                continue
            seen_ids.add(id(m))
            members.append(m)
        if len(members) < 2:
            logger.info(
                "[grouping][rewrite] skipping %s: fewer than 2 distinct "
                "members after dedup.",
                proposal.proposal_id,
            )
            continue

        if not verify_fusable_subset(members, out):
            logger.info(
                "[grouping][rewrite] skipping %s: non-convex subset (fusion "
                "would create a cycle).",
                proposal.proposal_id,
            )
            continue

        ordered = _topo_sort_members(members)

        fused_op = _unique_op_name(proposal.fused_op_name, used_ops)
        used_ops.add(fused_op)
        for m in ordered:
            used_ops.discard(m.op)

        fused = build_fused_node(
            ordered,
            out,
            fused_op_name=fused_op,
            fused_description=proposal.fused_description,
            rationale=proposal.rationale,
        )

        member_id_set = {id(m) for m in ordered}
        member_op_set = {m.op for m in ordered if m.op}

        def _is_member_instance(n: FAONode) -> bool:
            # Op-aware: a consumer may point at a member's duplicate instance.
            return id(n) in member_id_set or (bool(n.op) and n.op in member_op_set)

        non_members = [n for n in out.iter_preorder() if not _is_member_instance(n)]
        for n in non_members:
            if not any(_is_member_instance(c) for c in n.children):
                continue
            new_children: list[FAONode] = []
            fused_added = False
            for c in n.children:
                if _is_member_instance(c):
                    if not fused_added:
                        new_children.append(fused)
                        fused_added = True
                else:
                    new_children.append(c)
            n.children = new_children

        logger.info(
            "[grouping][rewrite] applied %s: %s -> %s",
            proposal.proposal_id,
            list(proposal.member_atoms),
            fused_op,
        )

    return out
