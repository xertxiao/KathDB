"""KathDB end-to-end facade.

``KathDB`` wraps the three pipeline stages behind one API::

    Parser  ->  PlanGenerator  ->  Executor

* **Parser** turns the natural-language query into an ordered list of atomic
  actions (one SEMANTIC or RELATIONAL operator each), optionally steered by the
  function library.
* **PlanGenerator** builds the operator DAG, annotates it with the columns each
  consumer needs (demand propagation), and runs the grouping optimizer, which
  fuses convex groups of operators so the generated code can push filters ahead
  of model calls and stop early.
* **Executor** generates Python for each operator (reusing library functions where
  they fit) and runs it in a sandboxed worker, layer by layer.
"""

from __future__ import annotations

import time

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pandas import DataFrame

from .common.context import DBContext
from .common.cost_tracker import (
    CostTracker,
    attach_handler,
    capture_into,
    new_stage_handler,
    read_inference_log_file,
)
from .common.function_manager import FunctionManager
from .common.logger import configure_logger, get_logger
from .common.view_schema import Modality
from .config import KathDBConfig, make_llm
from .executor.codegen import CodeGenerator
from .executor.codegen.grouping_cache import GroupingCache
from .executor.executor import Executor, decide_persistence
from .parser import ActionNLParser, ActionNLParserWithFunctions, BaseParser
from .plan_gen import PlanGenerator
from .plan_gen.optimizer import GroupingConfig
from .plan_gen.optimizer.list_rank import make_selector_factory
from .worker import WorkerManager

logger = get_logger(__name__)

__all__ = ["KathDB"]

# The basic settings ``KathDB(...)`` accepts directly (see ``KathDBConfig``).
_BASIC_SETTINGS = (
    "planner_model",
    "ai_op_model",
    "human_in_the_loop",
    "logical_rewrite",
    "phy_opt",
    "prebuilt_functions",
    "generated_functions",
    "max_generated_functions",
    "worker_env",
    "num_executor_workers",
)


class KathDB:
    """Unified entry point for the KathDB multimodal database system.

    Example usage::

        db = KathDB("my_catalog.db", planner_model="anthropic/claude-opus-5")
        db.register_table(df, "products", column_modalities={"image": Modality.IMAGE})
        result = db.query("Which products have a red logo?")
        db.close()

    Or as a context manager::

        with KathDB("my_catalog.db", human_in_the_loop=True) as db:
            db.register_table(df, "products")
            result = db.query("Which products have a red logo?")

    The keyword arguments are the basic settings of :class:`KathDBConfig`; advanced
    settings are edited in ``config.py`` or passed via ``config=``.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        planner_model: str | None = None,
        ai_op_model: str | None = None,
        human_in_the_loop: bool | None = None,
        logical_rewrite: bool | None = None,
        phy_opt: bool | None = None,
        prebuilt_functions: bool | None = None,
        generated_functions: bool | None = None,
        max_generated_functions: int | None = None,
        worker_env: str | None = None,
        num_executor_workers: int | None = None,
        config: KathDBConfig | None = None,
    ) -> None:
        self._config = config or KathDBConfig()
        overrides = {
            k: v for k, v in locals().items() if k in _BASIC_SETTINGS and v is not None
        }
        self._config.update(**overrides)
        self._config.validate()

        if self._config.log_level:
            configure_logger(level=self._config.log_level)
        self._export_worker_env()

        # -- Catalog ----------------------------------------------------------
        self._ctx = DBContext(
            db_path,
            llm=make_llm(
                self._config.planner_model, temperature=self._config.llm_temperature
            ),
        )

        # -- Function library (one instance shared by every stage) -------------
        self._fn_manager = self._build_fn_manager()

        # -- Worker -----------------------------------------------------------
        self._worker_mgr = self._build_worker_manager()

        # -- Pipeline components ----------------------------------------------
        self._parser = self._build_parser()
        self._plan_gen = self._build_plan_gen()
        self._executor = self._build_executor()

        self._last_grouping_trace: dict[str, Any] | None = None
        self._last_cost: CostTracker | None = None
        self._last_result_name: str | None = None
        # Worker-side LLM usage log; must be bound before the first ``get_worker()``.
        self._inference_log_path: str | None = None

    # ------------------------------------------------------------------
    # Environment handed to the worker subprocess
    # ------------------------------------------------------------------

    def _export_worker_env(self) -> None:
        """Publish config the worker subprocess reads from the environment.

        ``kathdb.fn`` inside the worker resolves the saved-function directory from
        ``KATHDB_GENERATED_FN_DIR``; script staging uses ``KATHDB_RUNTIME_DIR``.
        """
        if self._config.generated_fn_dir:
            os.environ["KATHDB_GENERATED_FN_DIR"] = str(
                Path(self._config.generated_fn_dir).expanduser().resolve()
            )
        if self._config.runtime_dir:
            os.environ["KATHDB_RUNTIME_DIR"] = str(self._config.runtime_dir)

    def _ensure_inference_log(self) -> None:
        """Bind ``KATHDB_INFERENCE_LOG_PATH`` (read by the worker at spawn) so the
        worker records per-call usage; an externally-set path is respected."""
        existing = os.environ.get("KATHDB_INFERENCE_LOG_PATH")
        if existing:
            self._inference_log_path = existing
            return
        if self._inference_log_path is None:
            path = os.path.join(
                tempfile.gettempdir(),
                f"kathdb_inference_{os.getpid()}_{id(self)}.jsonl",
            )
            os.environ["KATHDB_INFERENCE_LOG_PATH"] = path
            self._inference_log_path = path

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    def _fn_sources(self) -> tuple[str, ...]:
        cfg = self._config
        return tuple(
            name
            for name, on in (
                ("builtin", cfg.prebuilt_functions),
                ("generated", cfg.generated_functions),
            )
            if on
        )

    def _build_fn_manager(self) -> FunctionManager:
        cfg = self._config
        return FunctionManager(
            generated_fn_dir=(
                Path(cfg.generated_fn_dir).expanduser() if cfg.generated_fn_dir else None
            ),
            max_functions=cfg.max_generated_functions,
            sources=self._fn_sources(),
        )

    def _build_parser(self) -> BaseParser:
        cfg = self._config
        llm = cfg.get_llm("parser")
        function_reuse = bool(self._fn_sources())
        kwargs: dict[str, Any] = dict(
            clarification_llm=llm,
            sketch_llm=llm,
            revision_llm=llm,
            max_clarifications=cfg.max_parser_clarifications,
            max_revisions=cfg.max_parser_revisions,
            auto_mode=not cfg.human_in_the_loop,
            function_reuse=function_reuse,
            fn_manager=self._fn_manager,
        )
        if cfg.parser_type == "action" or not function_reuse:
            parser: BaseParser = ActionNLParser(**kwargs)
        else:
            parser = ActionNLParserWithFunctions(
                fn_coarsening=cfg.parser_type.endswith("with_coarsening"), **kwargs
            )
        parser.compile()
        return parser

    def _build_plan_gen(self) -> PlanGenerator:
        cfg = self._config
        gen = PlanGenerator(
            lp_llm=cfg.get_llm("plan_gen"),
            max_retries=cfg.max_plan_gen_retries,
            demand_propagation=cfg.demand_propagation,
            demand_propagation_one_shot_max_actions=(
                cfg.demand_propagation_one_shot_max_actions
            ),
            grouping_cfg=GroupingConfig(
                enabled=cfg.logical_rewrite,
                rank_k=cfg.grouping_rank_k,
                max_group_size=cfg.grouping_max_group_size,
            ),
            fn_manager=self._fn_manager,
        )
        gen.compile()
        return gen

    def _build_executor(self) -> Executor:
        cfg = self._config
        code_gen = CodeGenerator(
            generation_llm=cfg.get_llm("executor_generation"),
            diagnosis_llm=cfg.get_llm("executor_diagnosis"),
            revision_llm=cfg.get_llm("executor_revision"),
            max_retries=cfg.max_codegen_retries,
            distinct_value_sample_k=cfg.distinct_value_sample_k,
            max_concurrent_generations=cfg.codegen_concurrency,
            ai_op_model=cfg.ai_op_model,
            ai_op_temperature=cfg.ai_op_temperature,
            image_detail_low=cfg.image_quality_low_ai_op,
            phy_opt=cfg.phy_opt,
            fn_manager=self._fn_manager,
        )
        code_gen.compile()
        return Executor(
            code_gen=code_gen,
            auto_mode=not cfg.human_in_the_loop,
            save_functions=cfg.generated_functions,
            save_function_timeout_sec=cfg.save_function_timeout_sec,
            fn_manager=self._fn_manager,
        )

    def _build_worker_manager(self) -> WorkerManager:
        cfg = self._config
        req = (
            Path(cfg.requirements_path)
            if cfg.requirements_path
            else Path(__file__).parent / "worker" / "requirements.txt"
        )
        return WorkerManager(
            conda_env_name=cfg.worker_env,
            requirements_path=req,
            connect_timeout_s=cfg.worker_connect_timeout_s,
            exec_timeout_s=cfg.worker_exec_timeout_s,
            max_workers=cfg.num_executor_workers,
        )

    # ------------------------------------------------------------------
    # Data registration
    # ------------------------------------------------------------------

    def register_table(
        self,
        df: DataFrame,
        table_name: str,
        *,
        column_modalities: Dict[str, Modality] | None = None,
        description: str | None = None,
        column_descriptions: Dict[str, str] | None = None,
    ) -> None:
        """Register a DataFrame as a table in the catalog.

        ``column_modalities`` marks columns holding paths to images / videos / long
        text; each such column gets an auto-populated multimodal view the planner
        can reference.
        """
        self._ctx.register_table(
            df,
            table_name,
            column_modalities=column_modalities,
            description=description,
            column_descriptions=column_descriptions,
        )
        logger.info("Registered table '%s' (%d rows).", table_name, len(df))

    def register_csv(
        self,
        path: str | Path,
        table_name: str,
        *,
        column_modalities: Dict[str, Modality] | None = None,
        description: str | None = None,
        **read_csv_kwargs: Any,
    ) -> None:
        df = pd.read_csv(path, **read_csv_kwargs)
        self.register_table(
            df,
            table_name,
            column_modalities=column_modalities,
            description=description,
        )

    def register_parquet(
        self,
        path: str | Path,
        table_name: str,
        *,
        column_modalities: Dict[str, Modality] | None = None,
        description: str | None = None,
    ) -> None:
        df = pd.read_parquet(path)
        self.register_table(
            df,
            table_name,
            column_modalities=column_modalities,
            description=description,
        )

    def discover(
        self,
        root: str | Path,
        *,
        recursive: bool = True,
        llm_assist: bool | None = None,
    ) -> list[str]:
        """Auto-register every tabular / media file under ``root``."""
        names = self._ctx.discover(root, recursive=recursive, llm_assist=llm_assist)
        logger.info("Auto-discovered %d table(s) under %s: %s", len(names), root, names)
        return names

    # ------------------------------------------------------------------
    # Catalog introspection (thin wrappers around DBContext)
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        return self._ctx.list_tables()

    def has_table(self, name: str) -> bool:
        return self._ctx.has_table(name)

    def load_table(self, name: str, *, n: int | None = None) -> DataFrame:
        return self._ctx.load_table(name, n=n)

    def inspect(
        self,
        name: str | None = None,
        *,
        sample_n: int = 3,
        cell_char_limit: int = 40,
    ) -> None:
        self._ctx.inspect(name, sample_n=sample_n, cell_char_limit=cell_char_limit)

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def query(
        self,
        nl_query: str,
        *,
        input_tables: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the full NL-to-result pipeline.

        Returns the execution context: ``{relation_name: DataFrame}`` holding the
        input tables plus every intermediate and final relation the plan produced.
        """
        self._ensure_inference_log()
        # Warm-start the worker.
        self._worker_mgr.get_worker()

        rel_names = (
            input_tables if input_tables is not None else self._ctx.list_tables()
        )
        if not rel_names:
            raise RuntimeError("No tables registered. Call register_table() first.")

        # Parent-side stages attach a usage handler; worker-side model calls are read
        # from the inference log at stage boundaries.
        tracker = CostTracker()
        read_inference_log_file(self._inference_log_path, reset=True)

        # 1 -- Parser ---------------------------------------------------------
        logger.info("=== Stage 1/3: Parsing ===")
        with capture_into(tracker, "parser"):
            parser_out = self._parser.run(
                {
                    "q_in": nl_query,
                    "relation_context": self._ctx,
                    "input_rel_names": rel_names,
                }
            )
        q_in = parser_out["q_in"]
        actions = parser_out["actions"]
        input_rel_names = parser_out["input_rel_names"]

        # 2 -- Plan generator -------------------------------------------------
        logger.info("=== Stage 2/3: Plan Gen ===")
        input_dfs = [self._ctx.load_table(n) for n in input_rel_names]

        # The optimizer code-generates (and optionally profiles) the atomic plan at plan
        # time; ``grouping_cache`` keeps that code so unfused operators are not regenerated.
        grouping_cache: GroupingCache | None = None
        selector_factory = None
        if self._config.logical_rewrite:
            grouping_cache = GroupingCache()
            selector_factory = make_selector_factory(
                code_gen=self._executor._code_gen,
                rc=self._ctx,
                nl_query=q_in,
                grouping_cache=grouping_cache,
                sample_rows=self._config.grouping_sample_rows,
                rank_k=self._config.grouping_rank_k,
                max_group_size=self._config.grouping_max_group_size,
                profile=self._config.grouping_base_plan_profiling,
                worker_manager=self._worker_mgr,
            )

        with capture_into(tracker, "plan_gen"):
            plan_out = self._plan_gen.run(
                {
                    "q_in": q_in,
                    "actions": actions,
                    "relation_context": self._ctx,
                    "input_rel_names": input_rel_names,
                    "input_rel": input_dfs,
                    "grouping_selector_factory": selector_factory,
                }
            )
        self._last_grouping_trace = plan_out.get("grouping_trace")

        # ``capture_into`` charged the whole stage to ``plan_gen``; move the optimizer's
        # calls (parent-side and worker-side profiling) into ``grouping``.
        grouping_um = (plan_out.get("usage_by_model") or {}).get("group_actions") or {}
        tracker.record_usage_metadata("plan_gen", grouping_um, sign=-1)
        tracker.record_usage_metadata("grouping", grouping_um)
        gi, go, gc = read_inference_log_file(self._inference_log_path, reset=True)
        tracker.record_inference_totals("grouping", gi, go, gc)

        logger.info("Plan:\n%s", plan_out["logical_plan"].pretty())

        # 3 -- Executor (layered codegen + eval, interleaved per topo layer) -
        logger.info("=== Stage 3/3: Executor (codegen + eval) ===")
        cg_in: dict[str, Any] = {
            "q_in": plan_out["q_in"],
            "actions": plan_out["actions"],
            "relation_context": plan_out["relation_context"],
            "input_rel_names": plan_out["input_rel_names"],
            "input_rel": plan_out["input_rel"],
            "logical_plan": plan_out["logical_plan"],
        }
        if grouping_cache is not None:
            cg_in["_grouping_cache"] = grouping_cache
        # Codegen runs in the executor's thread pool, so its usage is captured through a
        # config-attached handler; execution usage comes from the worker inference log.
        codegen_cb = new_stage_handler()
        t_exec = time.perf_counter()
        result_ctx = self._executor.run(
            cg_in,
            worker_manager=self._worker_mgr,
            config=attach_handler(None, codegen_cb),
        )
        tracker.add_wall_time("execution", time.perf_counter() - t_exec)
        tracker.record_usage_metadata("codegen", codegen_cb.usage_metadata)
        ei, eo, ec = read_inference_log_file(self._inference_log_path, reset=True)
        tracker.record_inference_totals("execution", ei, eo, ec)
        self._last_cost = tracker
        logger.info(self._last_cost.summary())

        # Generated code tree, for the persistence and function-save walks.
        code_tree = self._executor._last_code_tree
        self._last_result_name = (
            code_tree.outputs[-1] if code_tree is not None and code_tree.outputs else None
        )

        # LLM-guided persistence: ask which result tables are worth keeping.
        tables_to_persist = decide_persistence(
            make_llm(self._config.planner_model, temperature=self._config.llm_temperature),
            nl_query=nl_query,
            plan=code_tree,
            result_ctx=result_ctx,
            input_rel_names=input_rel_names,
            skip_user_review=not self._config.human_in_the_loop,
        )
        if tables_to_persist:
            self._ctx.import_execution_results(result_ctx, tables_to_persist)
            logger.info("Persisted tables (user-approved): %s", tables_to_persist)

        if code_tree is not None:
            self._executor.walk_and_offer_saves(code_tree)
            self._executor.wait_for_saves()

        return result_ctx

    @property
    def config(self) -> KathDBConfig:
        """Current settings (read-only view; change them with :meth:`configure`)."""
        return self._config

    @property
    def worker_env_name(self) -> str | None:
        """Conda env the worker runs in; pass it as ``worker_env`` to reuse it."""
        return self._worker_mgr.env_name

    def last_result_name(self) -> str | None:
        """Name of the final relation produced by the most recent ``.query()``."""
        return self._last_result_name

    def last_result(self, result_ctx: dict[str, Any] | None = None) -> DataFrame | None:
        """The final relation of the most recent ``.query()`` (``None`` if it produced none).

        Pass the dict ``.query()`` returned; without it the relation is read from the
        catalog, where it exists only if it was persisted.
        """
        name = self._last_result_name
        if name is None:
            return None
        if result_ctx is not None and isinstance(result_ctx.get(name), DataFrame):
            return result_ctx[name]
        return self._ctx.load_table(name) if self._ctx.has_table(name) else None

    def last_grouping_trace(self) -> dict[str, Any] | None:
        """The grouping optimizer's trace for the most recent ``.query()``.

        ``None`` before the first query or when ``logical_rewrite`` is off. See
        :class:`kathdb.plan_gen.optimizer.GroupingTrace`.
        """
        return self._last_grouping_trace

    def last_cost(self) -> CostTracker | None:
        """Per-stage token/USD accounting of the most recent ``.query()``.

        Stages: ``parser``, ``plan_gen``, ``grouping``, ``codegen``, ``execution``.
        ``None`` before the first query. See :class:`kathdb.common.cost_tracker.CostTracker`.
        """
        return self._last_cost

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, **overrides: Any) -> None:
        """Update any setting (basic or advanced) and rebuild the pipeline components.

        The catalog (``DBContext``) is not rebuilt, so its LLM does not refresh live.
        """
        changed = self._config.update(**overrides)
        if not changed:
            return

        self._export_worker_env()
        self._fn_manager = self._build_fn_manager()
        self._parser = self._build_parser()
        self._plan_gen = self._build_plan_gen()
        self._executor = self._build_executor()

        if changed & {
            "worker_env",
            "num_executor_workers",
            "requirements_path",
            "remove_worker_env_on_close",
            "worker_connect_timeout_s",
            "worker_exec_timeout_s",
            "generated_fn_dir",
            "runtime_dir",
        }:
            self._worker_mgr.shutdown(
                remove_env=self._config.remove_worker_env_on_close,
            )
            self._worker_mgr = self._build_worker_manager()

        if "log_level" in changed and self._config.log_level:
            configure_logger(level=self._config.log_level)

        logger.info("Configuration updated: %s", changed)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Commit the catalog to disk."""
        self._ctx.save(path or self._ctx._db_path)

    def close(self) -> None:
        self._worker_mgr.shutdown(
            remove_env=self._config.remove_worker_env_on_close,
        )
        try:
            self._ctx.close()
        except Exception as exc:
            logger.warning("Error closing DBContext: %s", exc)

    def __enter__(self) -> KathDB:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"KathDB(tables={self._ctx.list_tables()})"
