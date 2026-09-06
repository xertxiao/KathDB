"""Pydantic response schemas for the NL parser module."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

CANONICAL_REL_OPS: frozenset[str] = frozenset(
    {
        "filter",
        "join",
        "group_by_aggregate",
        "project",
        "sort",
        "limit",
        "rename",
        "compute",
        "distinct",
        "union",
        "intersect",
        "difference",
        "window",
    }
)

_RELATIONAL_SUBTYPES: tuple[str, ...] = tuple(
    f"RELATIONAL-{op.upper()}" for op in sorted(CANONICAL_REL_OPS)
)

OpKindLiteral = Literal[
    "SEMANTIC",
    "RELATIONAL",
    "RELATIONAL-COMPUTE",
    "RELATIONAL-DIFFERENCE",
    "RELATIONAL-DISTINCT",
    "RELATIONAL-FILTER",
    "RELATIONAL-GROUP_BY_AGGREGATE",
    "RELATIONAL-INTERSECT",
    "RELATIONAL-JOIN",
    "RELATIONAL-LIMIT",
    "RELATIONAL-PROJECT",
    "RELATIONAL-RENAME",
    "RELATIONAL-SORT",
    "RELATIONAL-UNION",
    "RELATIONAL-WINDOW",
]

__all__ = [
    "ClarificationOption",
    "ClarificationResponse",
    "RefinedQueryResponse",
    "ActionItem",
    "ActionSketchResponse",
    "ActionItemWithFunctions",
    "ActionSketchWithFunctionsResponse",
    "PickedFunction",
    "PickFunctionsResponse",
]


class ClarificationOption(BaseModel):
    """A single option presented to the user during clarification."""

    label: str = Field(
        description="Sequential option label (e.g. 'A', 'B', 'C', 'D', ...)."
    )
    description: str = Field(
        description=(
            "A concrete meaning or interpretation for this option. "
            "Ground in relation schemas whenever possible "
            "(e.g., 'Measure popularity by sales rank (table: book_table; column: sales_rank)')."
        )
    )


class ClarificationResponse(BaseModel):
    """LLM response for clarification check."""

    status: Literal["clear", "clarify"] = Field(
        description=(
            "You must return exactly 'clear' if the query is unambiguous and ready "
            "to process, or 'clarify' if clarification is needed from the user."
        )
    )
    question: Optional[str] = Field(
        default=None,
        description=(
            "If status is 'clarify', provide the clarification question to ask the user "
            "(without inline options — those go in the 'options' field). "
            "Must be null if status is 'clear'."
        ),
    )
    options: Optional[list[ClarificationOption]] = Field(
        default=None,
        description=(
            "If status is 'clarify', provide at least 2 options labeled sequentially "
            "(A, B, C, ...). Each option must represent a substantively different "
            "interpretation of the ambiguous term — do not generate near-duplicate or "
            "overlapping options. Generate only as many options as there are genuinely "
            "distinct interpretations; do not pad. "
            "Must be null if status is 'clear'."
        ),
    )

    @model_validator(mode="after")
    def _check_options_count(self) -> "ClarificationResponse":
        if self.status == "clarify" and (self.options is None or len(self.options) < 2):
            raise ValueError(
                "At least 2 options are required when status is 'clarify'."
            )
        return self


class RefinedQueryResponse(BaseModel):
    """LLM response for query refinement after clarification."""

    refined_query: str = Field(
        description=(
            "The refined natural language query that incorporates the user's clarification. "
            "Must be a complete, unambiguous query string."
        )
    )


class ActionItem(BaseModel):
    """A single action in the query sketch."""

    name: str = Field(
        description=(
            "Query-specific snake_case label describing what this action "
            "does (e.g. filter_cheap_products, join_with_images, "
            "classify_sentiment). Should hint at the objective for this "
            "particular query."
        )
    )
    action: str = Field(
        description="Verb + Subject phrase for this query; concrete columns/filters when known."
    )
    inputs: list[str] = Field(
        description="1–2 input names (catalog entries or prior step outputs).",
    )
    output: str = Field(
        description="Unique name for this action's output.",
    )
    output_type: Literal[
        "dataframe", "int", "float", "string", "bool", "list", "dict"
    ] = Field(
        description=(
            "Python value-shape of this action's output: 'dataframe' for a "
            "table, or one of 'int'/'float'/'string'/'bool'/'list'/'dict' for "
            "a non-tabular value."
        ),
    )
    op_kind: OpKindLiteral = Field(
        description=(
            "Operator type from a fixed vocabulary. "
            "For RELATIONAL actions use EXACTLY one of: "
            "RELATIONAL-FILTER, RELATIONAL-JOIN, "
            "RELATIONAL-GROUP_BY_AGGREGATE, RELATIONAL-PROJECT, "
            "RELATIONAL-SORT, RELATIONAL-LIMIT, RELATIONAL-RENAME, "
            "RELATIONAL-COMPUTE, RELATIONAL-DISTINCT, RELATIONAL-UNION, "
            "RELATIONAL-INTERSECT, RELATIONAL-DIFFERENCE, "
            "RELATIONAL-WINDOW. "
            "Each maps to an extended relational algebra operator — a "
            "pure data transformation computable by DuckDB or pandas "
            "without any model inference. "
            "For SEMANTIC actions use 'SEMANTIC' — the action needs an "
            "ML/LLM/VLM inference call to compute its output (one model "
            "call per input row or per group); this is the dominant cost "
            "we optimize."
        ),
    )


class ActionSketchResponse(BaseModel):
    """LLM response containing the list of actions for a query sketch."""

    actions: list[ActionItem] = Field(description="Ordered list of actions.")


class ActionItemWithFunctions(ActionItem):
    """Action sketch entry that also carries pre-built function picks."""

    selected_functions: list[str] = Field(
        default_factory=list,
        description=(
            "Names of pre-built functions from `## Available Functions` whose "
            "documentation matches this action. Include every plausible "
            "match; downstream code generation will compose them. Leave "
            "empty when no listed function applies."
        ),
    )


class ActionSketchWithFunctionsResponse(BaseModel):
    """LLM response for the fused sketch + function-picking parser variant."""

    actions: list[ActionItemWithFunctions] = Field(
        description="Ordered list of actions, each annotated with picked functions."
    )


class PickedFunction(BaseModel):
    """A single pre-built function picked for an action."""

    function_name: str = Field(
        description="Exact function name as it appears in `## Available Functions`."
    )
    reasoning: str = Field(
        description="One short sentence: why this function fits the action."
    )


class PickFunctionsResponse(BaseModel):
    """LLM response for per-action pre-built function picking."""

    selected_functions: list[PickedFunction] = Field(
        default_factory=list,
        description=(
            "All pre-built functions whose documentation matches this action. "
            "Include every plausible match; downstream code generation will "
            "compose them. Leave empty when no listed function applies."
        ),
    )
