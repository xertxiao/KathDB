"""LLM-guided result persistence: which result tables are worth keeping in the catalog."""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ..common.logger import get_logger
from ..common.utils import invoke_structured_with_retry, sample_dataframe
from .codegen.codegen_tree import FAOExecutableNode, walk_nodes

logger = get_logger(__name__)

__all__ = ["decide_persistence"]


class _TablePersistenceReason(BaseModel):
    """Per-table reasoning for persistence."""

    table_name: str = Field(description="Name of the candidate table.")
    reason: str = Field(
        description="Why this table is worth persisting for future queries."
    )


class _PersistenceDecisionResponse(BaseModel):
    """LLM response for deciding which tables to persist."""

    tables_to_persist: list[_TablePersistenceReason] = Field(
        description=(
            "Tables worth persisting, each with a reason explaining future utility."
        )
    )
    reasoning: str = Field(
        description="Brief overall explanation of why these tables were selected."
    )


def decide_persistence(
    llm: BaseChatModel,
    *,
    nl_query: str,
    plan: FAOExecutableNode,
    result_ctx: dict[str, Any],
    input_rel_names: list[str],
    sample_rows: int = 5,
    skip_user_review: bool = False,
) -> list[str]:
    """Ask the LLM which result tables (not inputs) are worth persisting; the user
    approves each unless ``skip_user_review``. Returns the approved table names;
    returns ``[]`` when the LLM call fails."""
    new_keys = [
        k
        for k in result_ctx
        if k not in set(input_rel_names) and isinstance(result_ctx[k], pd.DataFrame)
    ]
    if not new_keys:
        return []

    table_summaries: list[str] = []
    table_info: dict[str, dict[str, Any]] = {}
    for name in new_keys:
        df = result_ctx[name]
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        sample = sample_dataframe(df, sample_rows)
        sample_str = sample.to_string(index=False, max_colwidth=60)
        table_summaries.append(
            f"Table: {name}\n"
            f"  Columns: {cols}\n"
            f"  Rows: {len(df)}\n"
            f"  Sample:\n{sample_str}"
        )
        table_info[name] = {
            "columns": cols,
            "row_count": len(df),
            "sample_str": sample_str,
        }

    fn_summaries: list[str] = []
    for node in walk_nodes(plan):
        fn = node.function
        fn_summaries.append(
            f"- {node.op}: {fn.name}"
            + (f" -- {fn.description}" if fn.description else "")
        )

    prompt = (
        "## System\n"
        "You are a database assistant deciding which intermediate/final tables "
        "from a query execution should be persisted to DuckDB for potential "
        "future reuse by similar workloads.\n\n"
        "## Original Query\n"
        f"{nl_query}\n\n"
        "## Functions Used\n"
        + "\n".join(fn_summaries)
        + "\n\n## Candidate Tables\n"
        + "\n\n".join(table_summaries)
        + "\n\n## Instructions\n"
        "Select only tables that would be useful for future similar queries. "
        "Do NOT persist tables that are trivially re-derivable or only relevant "
        "to this specific query. Prefer persisting tables that are expensive to "
        "compute (e.g. LLM-generated columns, aggregations over large datasets). "
        "For each selected table, explain why it is worth keeping."
    )

    try:
        response = invoke_structured_with_retry(
            prompt,
            llm=llm,
            schema=_PersistenceDecisionResponse,
        )
        valid_entries = [
            entry
            for entry in response.tables_to_persist
            if entry.table_name in new_keys
        ]
        logger.info(
            "Persistence decision: persist=%s, reasoning=%s",
            [e.table_name for e in valid_entries],
            response.reasoning,
        )
    except Exception:
        # Fail closed: never persist on an LLM error.
        logger.warning(
            "LLM persistence decision failed; persisting NOTHING.",
            exc_info=True,
        )
        return []

    if not valid_entries:
        return []

    if skip_user_review:
        return [e.table_name for e in valid_entries]

    # Present each table to the user for approval
    approved: list[str] = []
    accept_all = False

    logger.interact("\n" + "=" * 60 + "\n  TABLE PERSISTENCE REVIEW\n" + "=" * 60)

    for entry in valid_entries:
        name = entry.table_name
        info = table_info.get(name, {})

        logger.interact(
            "\n"
            + "-" * 60
            + f"\n  Table: {name}\n"
            + f"  Columns: {info.get('columns', '(unknown)')}\n"
            + f"  Rows: {info.get('row_count', '?')}\n"
            + f"\n  Reason to keep: {entry.reason}\n"
            + f"\n  Sample:\n{info.get('sample_str', '(no sample)')}\n"
            + "-" * 60
        )

        if accept_all:
            approved.append(name)
            logger.info("Auto-accepted table '%s' (accept all).", name)
            continue

        response_text = input(
            "\nPersist this table? "
            "(type 'yes' to persist, 'accept all' to persist remaining, "
            "or anything else to skip): "
        ).strip()
        logger.info("Table persistence feedback for '%s': %s", name, response_text)

        lower = response_text.lower()
        if lower in ("yes", "y"):
            approved.append(name)
        elif lower in ("accept all", "accept_all"):
            accept_all = True
            approved.append(name)

    return approved
