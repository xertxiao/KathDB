"""Pure-data types for the plan-grouping optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["GroupingConfig", "GroupingTrace", "MergeProposal", "Partition"]


# A partition is a tuple of frozensets of atom op-names. Tuple + frozenset make it
# hashable, so partitions can be de-duplicated with a plain set.
Partition = tuple[frozenset[str], ...]


@dataclass(frozen=True)
class GroupingConfig:
    """The optimizer's knobs (the grouping subset of :class:`~kathdb.KathDBConfig`)."""

    enabled: bool = True
    # Ranking width: one LLM call ranks up to ``rank_k`` candidate partitions. With
    # more candidates a recursive tournament keeps the best of each ``rank_k``-chunk
    # and ranks the winners (``rank_k=2`` is pairwise comparison).
    rank_k: int = 10
    # Maximum number of atomic operators in one fused group. ``None`` = no cap.
    max_group_size: int | None = None


@dataclass(frozen=True)
class MergeProposal:
    """One fusion of a convex subgraph of atomic operators.

    Materialised into a fused ``FAONode`` by :func:`~.rewriter.apply_merges`.
    """

    proposal_id: str
    member_atoms: tuple[str, ...]
    fused_op_name: str
    fused_description: str
    rationale: str


@dataclass
class GroupingTrace:
    """What the optimizer did for one query. Telemetry only; nothing reads it back."""

    n_atoms: int
    rank_k: int = 0
    max_group_size: int | None = None
    # Number of candidate partitions that were ranked (0 when nothing was ranked).
    n_candidates: int = 0
    # The fused groups applied to the plan: {id, members, rationale, description}.
    fused_groups: list[dict[str, Any]] = field(default_factory=list)
    # Why no grouping happened (``None`` when a grouping was applied).
    short_circuit_reason: str | None = None
    atomic_dag: dict[str, Any] | None = None
    final_dag: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_atoms": self.n_atoms,
            "rank_k": self.rank_k,
            "max_group_size": self.max_group_size,
            "n_candidates": self.n_candidates,
            "fused_groups": list(self.fused_groups),
            "short_circuit_reason": self.short_circuit_reason,
            "atomic_dag": self.atomic_dag,
            "final_dag": self.final_dag,
        }
