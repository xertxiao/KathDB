"""Plan-grouping optimizer: fuse convex groups of atomic operators so the generated
code can push filters ahead of model calls, exit early, and share or cache calls.

``list_rank``: (1) code-generate every atom once on a sample (optionally profiled);
(2) enumerate every legal convex partition (groups <= ``max_group_size``; fusions
without a model call are dropped; very wide plans get LLM-proposed candidates);
(3) an LLM ranks up to ``rank_k`` candidates per call (tournament beyond that); the
winner is validated and applied. Entry point :func:`run_optimizer`, knobs
:class:`GroupingConfig`.
"""

from __future__ import annotations

from .orchestrator import run_optimizer
from .types import GroupingConfig, GroupingTrace, MergeProposal, Partition

__all__ = [
    "GroupingConfig",
    "GroupingTrace",
    "MergeProposal",
    "Partition",
    "run_optimizer",
]
