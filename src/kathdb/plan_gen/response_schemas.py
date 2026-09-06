"""Pydantic response schemas for the logical plan module."""

from __future__ import annotations

from pydantic import BaseModel, Field


__all__ = [
    "DemandedColumn",
    "ValueConstraint",
    "InputDemand",
    "DemandPropagationResponse",
    "QueryDemandResponse",
    "FinalOutputDemand",
    "NodeDemand",
    "OpKindDecision",
    "AllNodesDemandResponse",
]


class DemandedColumn(BaseModel):
    """A column required by downstream consumers."""

    name: str = Field(description="Column name required by consumers.")
    dtype: str = Field(description="Expected DuckDB type.")
    reason: str = Field(description="Why needed (which consumer, what for).")


class ValueConstraint(BaseModel):
    """A value-level constraint on a column from consumer demands."""

    column: str = Field(description="Column this applies to.")
    constraint: str = Field(
        description="e.g., \"one of ['positive','negative']\" or \"integer range 1-5\"."
    )


class InputDemand(BaseModel):
    """Demands that a node places on one of its input relations."""

    input_relation: str = Field(description="Input relation name.")
    required_columns: list[DemandedColumn] = Field(default_factory=list)
    value_constraints: list[ValueConstraint] = Field(default_factory=list)


class DemandPropagationResponse(BaseModel):
    """LLM response for demand propagation (intermediate nodes)."""

    input_demands: list[InputDemand] = Field(
        description="Demands this node places on each of its input relations."
    )


class QueryDemandResponse(BaseModel):
    """LLM response for extracting demands from the root NL query."""

    required_columns: list[DemandedColumn] = Field(default_factory=list)
    value_constraints: list[ValueConstraint] = Field(default_factory=list)


class FinalOutputDemand(BaseModel):
    """User-query-derived demands on the LP's final output relation."""

    required_columns: list[DemandedColumn] = Field(default_factory=list)
    value_constraints: list[ValueConstraint] = Field(default_factory=list)


class NodeDemand(BaseModel):
    """Demands that a single LP node places on its input relations."""

    node_id: str = Field(
        description=(
            "Stable identifier of the consumer node — the node's primary "
            "output relation name (unique by LP construction)."
        )
    )
    input_demands: list[InputDemand] = Field(default_factory=list)


class OpKindDecision(BaseModel):
    """Per-node op-kind classification derived from propagated demands."""

    node_id: str = Field(
        description=(
            "Stable identifier of the node — the node's primary output relation "
            "name (unique by LP construction)."
        )
    )
    parser_op_kind: str = Field(
        description=(
            "Current op_kind tag as seen in the prompt: 'SEMANTIC', "
            "'RELATIONAL-FILTER', 'RELATIONAL-JOIN', or another "
            "fine-grained RELATIONAL-* subtype, or 'UNSET' for GROUPED "
            "nodes that have not yet been tagged."
        )
    )
    chosen_op_kind: str = Field(
        description=(
            "Final base-kind decision: 'SEMANTIC' or 'RELATIONAL'. "
            "The parser may emit fine-grained subtypes "
            "(e.g. 'RELATIONAL-FILTER') but this field is always the "
            "base kind."
        )
    )
    rationale: str = Field(
        description="1-2 sentence explanation of the decision in this DAG context."
    )
    evidence: str = Field(
        description=(
            "Which producer outputs / value_constraints / demands you relied on "
            "(cite by node_id or column name)."
        )
    )


class AllNodesDemandResponse(BaseModel):
    """LLM response for one-shot demand propagation over the full LP DAG."""

    final_output_demand: FinalOutputDemand = Field(
        description="Demands derived from the NL query on the LP's final output relation."
    )
    node_demands: list[NodeDemand] = Field(
        default_factory=list,
        description="Per-node input-relation demands for every non-input-relation node.",
    )
    op_kind_decisions: list[OpKindDecision] = Field(
        default_factory=list,
        description=(
            "Per-node final op_kind classification derived from the propagated "
            "demands. Used to downgrade SEMANTIC nodes to RELATIONAL when "
            "value-domain constraints make the operation deterministic."
        ),
    )
