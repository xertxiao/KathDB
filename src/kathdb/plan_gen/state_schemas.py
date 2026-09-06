"""Input / output state of :class:`~kathdb.plan_gen.plan_generator.PlanGenerator`."""

from __future__ import annotations

from typing import Any, TypedDict

from pandas import DataFrame
from typing_extensions import NotRequired

from ..common.context import DBContext
from .plan_node import FAONode

__all__ = ["PlanGenInState", "PlanGenOutState"]


class PlanGenInState(TypedDict):
    """Input state: the parser output plus the catalog."""

    q_in: str
    actions: list[Any]
    relation_context: DBContext
    input_rel_names: list[str]
    input_rel: list[DataFrame]
    # ``atomic_root -> ListRankSelector``; injected by :class:`~kathdb.KathDB`.
    grouping_selector_factory: NotRequired[Any]


class PlanGenOutState(TypedDict, total=False):
    q_in: str
    actions: list[Any]
    relation_context: DBContext
    input_rel_names: list[str]
    input_rel: list[DataFrame]
    logical_plan: FAONode
    # Present only when the grouping optimizer ran.
    grouping_trace: dict[str, Any]
    # Per-stage token usage ("annotate") plus "_total".
    usage_by_model: dict[str, dict[str, Any]]
