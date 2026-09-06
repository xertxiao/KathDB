"""Pydantic response schemas for the DBContext module."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "ColumnDescription",
    "TableDescriptionResponse",
]


class ColumnDescription(BaseModel):
    """A single column's semantic description."""

    column_name: str = Field(
        description="The exact column name as it appears in the table schema."
    )
    description: str = Field(
        description=(
            "A short, factual semantic description of what this column represents. "
            "Do NOT include data types."
        )
    )


class TableDescriptionResponse(BaseModel):
    """LLM response for table and column description generation."""

    table_summary: str = Field(
        description="A single sentence summarizing what the table captures."
    )
    column_descriptions: list[ColumnDescription] = Field(
        description=(
            "One entry per column, each with the exact column name and a short "
            "semantic description. Cover every column provided in the metadata."
        )
    )
