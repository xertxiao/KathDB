"""State schemas for the NL parser components."""

from __future__ import annotations

from operator import add
from typing import Any, Literal, TypedDict, Annotated
from typing_extensions import NotRequired

from ..common.context import DBContext

__all__ = ["ParserState"]


class ParserState(TypedDict):
    """Parser state tracking HITL phases and conversation history."""

    q_in: str
    actions: list[Any]
    relation_context: DBContext
    reviews_messages: Annotated[list[str], add]
    clarification_messages: Annotated[list[str], add]
    clarification_status: NotRequired[Literal["clear", "clarify"]]
    clarification_options: NotRequired[list[dict[str, str]]]
    runtime_deferred: Annotated[list[str], add]
    reviews_count: int
    clarifications_count: int
