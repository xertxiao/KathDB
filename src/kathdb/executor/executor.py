"""Query executor: per-operator codegen + execution via :class:`CodeGenerator`,
then the function-save and table-persistence steps."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

import pandas as pd

from ..common.function_finalizer import finalize_with_llm
from ..common.function_manager import FunctionManager
from ..common.logger import get_logger
from ..common.utils import invoke_structured_with_retry, sample_dataframe
from ..executor.codegen import CodeGenerator, CodegenInState
from ..executor.codegen.codegen_tree import FAOExecutableNode, walk_nodes
from ..worker import WorkerClient, WorkerManager
from .error_handler import ExecutionErrorHandler

logger = get_logger(__name__)

__all__ = [
    "ExecutionErrorHandler",
    "Executor",
]


class Executor:
    """Runs a logical plan (codegen + execution per operator) and offers function saves afterwards."""

    def __init__(
        self,
        *,
        code_gen: CodeGenerator,
        auto_mode: bool = False,
        save_functions: bool = True,
        save_function_timeout_sec: float = 180.0,
        fn_manager: FunctionManager | None = None,
    ) -> None:
        self._code_gen = code_gen
        # auto_mode=True saves without asking the user first.
        self.auto_mode = auto_mode
        # save_functions=False disables saving entirely (reuse is unaffected).
        self.save_functions = save_functions
        self.save_function_timeout_sec = save_function_timeout_sec
        self._fn_manager = fn_manager or FunctionManager()
        self._save_threads: list = []
        # Code tree of the last run(); drives the post-run save walk.
        self._last_code_tree: FAOExecutableNode | None = None

    # ------------------------------------------------------------------
    # Function save logic
    # ------------------------------------------------------------------

    def _record_function_usage(self, node: FAOExecutableNode) -> None:
        """Record usage of any reused functions after successful execution."""
        metadata = node.metadata or {}
        selected = metadata.get("selected_functions") or []
        code = node.function.str_impl or ""
        for fn_name in selected:
            if f"from kathdb.fn import {fn_name}" in code:
                self._fn_manager.record_usage(fn_name)

    def _should_save_function(self, node: FAOExecutableNode) -> bool:
        """Save unless saving is off, the node has no op, or codegen said ``new_fn_worth_saving=False``."""
        if not self.save_functions or not node.op:
            return False
        worth = (node.metadata or {}).get("new_fn_worth_saving")
        if worth is False:
            logger.info(
                "Skipping function save for '%s': LLM new_fn_worth_saving=False.",
                node.op,
            )
        return worth is not False

    def _offer_function_save(self, node: FAOExecutableNode) -> None:
        """Prompt user (or auto-accept) to save a function after execution."""
        if not self._should_save_function(node):
            return

        metadata = node.metadata or {}
        selected = metadata.get("selected_functions") or []
        description = metadata.get("fn_description", "")
        input_rel_names = metadata.get("input_rel_names") or []

        user_accepted = False
        if self.auto_mode:
            user_accepted = True
            logger.info("auto_mode: auto-accepting function save for '%s'.", node.op)
        else:
            logger.interact(
                "\n"
                + "=" * 60
                + "\n  FUNCTION SAVE REVIEW\n"
                + "=" * 60
                + f"\n  Function: {node.op}\n"
                + f"  Description: {description or '(none)'}\n"
                + "-" * 60
            )
            response = input(
                "\nSave this function for future reuse? "
                "(type 'accept' to save, or anything else to discard): "
            ).strip()
            logger.info("Function save review feedback: %s", response)
            user_accepted = response.lower().startswith("accept")

        if user_accepted:
            member_atoms = metadata.get("member_atoms") or []
            save_info = {
                "fn_name": node.op,
                "description": description,
                "code": node.function.str_impl or "",
                "input_rel_names": list(input_rel_names),
                "outputs": list(node.outputs) if node.outputs else [],
                "selected_functions": list(selected),
                "member_atoms": list(member_atoms),
                "atom_count": len(member_atoms),
            }
            self._launch_background_save(save_info)
        else:
            logger.info("Function not saved for '%s' (user declined).", node.op)

    def _launch_background_save(self, save_info: dict) -> None:
        """Run the finalize-and-save step in a daemon thread."""
        import threading as _threading

        t = _threading.Thread(
            target=self._background_save_worker,
            args=(save_info,),
            daemon=True,
        )
        t.start()
        self._save_threads.append(t)

    def _background_save_worker(self, save_info: dict) -> None:
        """Finalize the code with the LLM (canonical name, typed params, CONTRACT),
        smoke-test it, and write it to ``generated_fn/``."""
        fn_name = save_info["fn_name"]
        code = save_info["code"]

        existing_names: list[str] = []
        try:
            existing_names = list(self._fn_manager.discover_functions().keys())
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not enumerate existing functions for collision check.",
                exc_info=True,
            )

        record, canonical_name, code_out = finalize_with_llm(
            fn_name=fn_name,
            code=code,
            llm=self._code_gen.generation_llm,
            existing_names=existing_names,
        )
        if record.status == "success":
            from kathdb.common.fn_smoke import run_smoke

            smoke_ok, smoke_detail = run_smoke(
                code_out, canonical_name, record.extra.get("smoke", "")
            )
            record.extra["smoke_result"] = smoke_detail
            if not smoke_ok:
                logger.warning(
                    "Save-time smoke test REJECTED '%s' (canonical='%s'): %s",
                    fn_name,
                    canonical_name,
                    smoke_detail,
                )
                return
        if record.status == "failed":
            logger.warning("Function save skipped for '%s': %s", fn_name, record.error)
            return
        if record.status == "success":
            try:
                fn_spec = {
                    "name": canonical_name,
                    "description": save_info.get("description") or "",
                    "inputs": [],
                    "output": {
                        "type": "pd.DataFrame",
                        "description": f"Output of {canonical_name}",
                    },
                }
                saved_name = self._fn_manager.save_function(fn_spec, code_out)
                if saved_name:
                    canonical_name = saved_name  # may be suffixed on name collision
                self._fn_manager.record_save(
                    name=canonical_name,
                    atom_count=save_info.get("atom_count", 0),
                    member_atoms=tuple(save_info.get("member_atoms", ())),
                )
                logger.info(
                    "Background save completed: '%s' -> '%s'.",
                    fn_name,
                    canonical_name,
                )
                evicted = self._fn_manager.evict_least_used()
                if evicted:
                    logger.info(
                        "Evicted %d least-used functions: %s",
                        len(evicted),
                        evicted,
                    )
            except Exception:  # noqa: BLE001 - a failed save must not fail the query
                logger.warning(
                    "Function save: writer failed for '%s' (canonical='%s').",
                    fn_name,
                    canonical_name,
                    exc_info=True,
                )

    def wait_for_saves(self) -> None:
        """Join pending save threads; one still alive after ``save_function_timeout_sec`` + 30 s is abandoned."""
        per_thread_ceiling = float(self.save_function_timeout_sec) + 30.0
        for t in self._save_threads:
            t.join(timeout=per_thread_ceiling)
            if t.is_alive():
                logger.warning(
                    "wait_for_saves: abandoning background save thread %s "
                    "after %.0fs — save is considered failed.",
                    t.name,
                    per_thread_ceiling,
                )
        self._save_threads.clear()

    # ------------------------------------------------------------------
    # Layered codegen + execution
    # ------------------------------------------------------------------

    def run(
        self,
        cg_in: CodegenInState,
        *,
        worker: WorkerClient | None = None,
        worker_manager: WorkerManager | None = None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Code-generate and execute ``cg_in["logical_plan"]``; return the execution context.

        The context holds the input relations plus every materialized output.
        Prefer *worker_manager* (parallel execution, poisoned workers replaced);
        a static *worker* runs sequentially.
        """
        cg_out = self._code_gen.run(
            cg_in, worker=worker, worker_manager=worker_manager, config=config
        )
        root_node: FAOExecutableNode | None = cg_out.get("code_tree")  # type: ignore[arg-type]
        self._last_code_tree = root_node

        result_ctx: dict[str, Any] = {
            name: df for name, df in zip(cg_in["input_rel_names"], cg_in["input_rel"])
        }
        if root_node is not None and root_node.metadata:
            materialized = root_node.metadata.pop("_layered_materialized_outputs", None)
            if materialized:
                result_ctx.update(materialized)
        logger.info(
            "Executor: layered codegen+exec finished; result_ctx has %d table(s)",
            len(result_ctx),
        )
        return result_ctx

    def walk_and_offer_saves(self, plan: FAOExecutableNode) -> None:
        """Record library-function usage and offer a save for each unique node (call after the query is done)."""
        self._fn_manager.reconcile_records()
        seen: set[int] = set()
        save_offers = 0
        for node in walk_nodes(plan):
            nid = id(node)
            if nid in seen:
                continue
            seen.add(nid)
            if not node.outputs or not getattr(node, "function", None):
                continue
            self._record_function_usage(node)
            self._offer_function_save(node)
            save_offers += 1
        logger.info("Executor: offered function-save for %d node(s)", save_offers)


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
