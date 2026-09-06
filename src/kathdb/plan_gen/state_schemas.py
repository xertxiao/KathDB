"""State schemas for plan generation."""

from __future__ import annotations

from typing import Any, Annotated, TypedDict

from pandas import DataFrame
from typing_extensions import NotRequired

from ..common.context import DBContext
from .plan_node import FAONode

__all__ = [
    "PlanGenInState",
    "PlanGenState",
    "PlanGenOutState",
    "merge_usage_by_model",
]


def merge_usage_by_model(
    a: dict[str, dict[str, Any]] | None,
    b: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Reducer for ``usage_by_model``: per-stage keys are disjoint, so merge."""
    out: dict[str, dict[str, Any]] = dict(a or {})
    out.update(b or {})
    return out


class PlanGenInState(TypedDict):
    """Input state: the parser output plus the catalog."""

    q_in: str
    actions: list[Any]
    relation_context: DBContext
    input_rel_names: list[str]
    input_rel: list[DataFrame]
    # ``atomic_root -> ListRankSelector``; injected by :class:`~kathdb.KathDB`.
    grouping_selector_factory: NotRequired[Any]


class PlanGenState(TypedDict, total=False):
    q_in: str
    actions: list[Any]
    relation_context: DBContext
    input_rel_names: list[str]
    input_rel: list[DataFrame]
    logical_plan: FAONode
    grouping_selector_factory: Any
    # Populated by the optimizer when grouping is enabled.
    grouping_trace: dict[str, Any]
    # Per-stage token usage ("annotate", "group_actions"; "_total" added in arun).
    usage_by_model: Annotated[dict[str, dict[str, Any]], merge_usage_by_model]


class PlanGenOutState(TypedDict, total=False):
    q_in: str
    actions: list[Any]
    relation_context: DBContext
    input_rel_names: list[str]
    input_rel: list[DataFrame]
    logical_plan: FAONode
    grouping_trace: dict[str, Any]
    usage_by_model: dict[str, dict[str, Any]]
