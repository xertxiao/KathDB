"""Code-gen cache shared between the grouping optimizer and the executor.

The optimizer's base-plan pass (:mod:`.base_plan_codegen`) code-generates every atomic
operator once and stores the result here. When the executor later walks the final
plan, every atom that was left unfused is served from this cache instead of paying
for a second code-gen call (``codegen.py`` checks ``cg_in["_grouping_cache"]``).
Fused groups are new operators, so they are always generated fresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .codegen_tree import FAOExecutableNode

__all__ = ["GroupingCache"]


@dataclass
class GroupingCache:
    # op-name -> (executable node, generated source)
    codegen: dict[str, tuple["FAOExecutableNode", str]] = field(default_factory=dict)
    codegen_hits: int = 0
    codegen_misses: int = 0
