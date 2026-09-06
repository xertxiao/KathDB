"""TypedDict states passed into / out of the codegen+exec pass."""

from __future__ import annotations

from pandas import DataFrame
from typing import TypedDict
from typing_extensions import NotRequired

from ...common.context import DBContext
from ...plan_gen.plan_node import FAONode
from .codegen_tree import FAOExecutableNode
from .grouping_cache import GroupingCache

__all__ = [
    "CodegenInState",
    "CodegenOutState",
]


class CodegenInState(TypedDict):
    """Input state for the codegen+exec pass."""

    q_in: str
    actions: list
    relation_context: DBContext
    input_rel_names: list[str]
    input_rel: list[DataFrame]
    logical_plan: FAONode
    output_name: NotRequired[str]
    # Code the grouping optimizer generated at plan time (unfused operators are not regenerated).
    _grouping_cache: NotRequired[GroupingCache]


class CodegenOutState(TypedDict):
    """Output state; ``code_tree`` drives the post-run save walk."""

    q_in: str
    actions: list
    relation_context: DBContext
    input_rel_names: list[str]
    input_rel: list[DataFrame]
    code_tree: FAOExecutableNode
    output_name: NotRequired[str]
