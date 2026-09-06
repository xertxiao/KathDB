"""State schemas for KathDB common components, designed for LangGraph."""

from __future__ import annotations

from typing import Any
from typing_extensions import NotRequired, TypedDict

from .context import DBContext

__all__ = ["QueryInState", "QueryOutState"]


class QueryInState(TypedDict):
    """State representation for an NL question (input)."""

    q_in: str
    relation_context: DBContext
    input_rel_names: NotRequired[list[str]]


class QueryOutState(TypedDict):
    """State representation for an NL question (output)."""

    q_in: str
    actions: list[Any]
    relation_context: DBContext
    input_rel_names: list[str]
