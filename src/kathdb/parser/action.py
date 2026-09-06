"""Action dataclass — the parser's typed output unit."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Action"]


@dataclass(slots=True)
class Action:
    """One parsed step of the query sketch (plan generation may fuse several into one FAONode)."""

    name: str
    action: str
    inputs: list[str]
    output: str
    output_type: str = "dataframe"
    op_kind: str | None = None
    selected_functions: list[str] = field(default_factory=list)
