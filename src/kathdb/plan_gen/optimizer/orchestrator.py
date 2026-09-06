"""Optimizer entry point: pick a partition of the atomic plan and apply it.

1. Ask the :class:`~.list_rank.ListRankSelector` for the best partition of the
   atoms into fused groups (validated: every atom covered once, every group convex).
2. Turn each multi-atom group into a :class:`MergeProposal` and rewrite the plan
   tree with :func:`~.rewriter.apply_merges`.
3. Return the rewritten tree plus a :class:`GroupingTrace`.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from ...common.logger import get_logger
from ..plan_node import FAONode
from .dag import all_atoms_of
from .list_rank import ListRankSelector, select_partition
from .rewriter import apply_merges
from .types import GroupingConfig, GroupingTrace, MergeProposal, Partition

logger = get_logger(__name__)

__all__ = ["run_optimizer"]


def fused_op_name(members: Iterable[str]) -> str:
    """Stable, injective op-name for a fused group, keyed by its member set."""
    key = "\x00".join(sorted(members))
    return "g_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def _proposals_from_partition(partition: Partition) -> list[MergeProposal]:
    out: list[MergeProposal] = []
    for idx, group in enumerate(partition):
        if len(group) < 2:
            continue
        members = tuple(sorted(group))
        out.append(
            MergeProposal(
                proposal_id=f"group_{idx}",
                member_atoms=members,
                fused_op_name=fused_op_name(members),
                fused_description=f"Fusion of: {', '.join(members)}",
                rationale="Ranked lowest in total execution LLM tokens.",
            )
        )
    return out


def run_optimizer(
    *,
    root: FAONode,
    cfg: GroupingConfig,
    selector: ListRankSelector,
) -> tuple[FAONode, GroupingTrace]:
    """Run the grouping optimizer on ``root``; return ``(rewritten_plan, trace)``.

    The input tree is never mutated; when nothing is fused it is returned as-is.
    """
    n_atoms = len(all_atoms_of(root))
    trace = GroupingTrace(
        n_atoms=n_atoms, rank_k=cfg.rank_k, max_group_size=cfg.max_group_size
    )
    try:
        trace.atomic_dag = root.to_dict()
    except Exception as exc:  # noqa: BLE001 - telemetry only
        logger.warning("[optimizer] atomic_dag serialization failed: %s", exc)
    trace.final_dag = trace.atomic_dag

    logger.info(
        "[optimizer] start: n_atoms=%d rank_k=%d max_group_size=%s",
        n_atoms,
        cfg.rank_k,
        cfg.max_group_size,
    )
    if n_atoms <= 1:
        trace.short_circuit_reason = "n_atoms_le_1"
        logger.info("[optimizer] skipped: n_atoms=%d (need >1)", n_atoms)
        return root, trace

    selection = select_partition(root, selector, max_group_size=cfg.max_group_size)
    trace.n_candidates = selection.n_candidates
    trace.short_circuit_reason = selection.reason

    proposals = _proposals_from_partition(selection.partition)
    if not proposals:
        logger.info("[optimizer] no multi-atom groups; returning atomic plan.")
        return root, trace

    rewritten = apply_merges(root, proposals)
    trace.fused_groups = [
        {
            "id": p.proposal_id,
            "members": list(p.member_atoms),
            "rationale": p.rationale,
            "description": p.fused_description,
        }
        for p in proposals
    ]
    try:
        trace.final_dag = rewritten.to_dict()
    except Exception as exc:  # noqa: BLE001 - telemetry only
        logger.warning("[optimizer] final_dag serialization failed: %s", exc)

    logger.info("[optimizer] DONE: %d group(s) fused", len(proposals))
    return rewritten, trace
