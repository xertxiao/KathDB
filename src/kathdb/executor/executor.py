"""Query executor: schedules a logical plan's operators by data dependency, code-
generates each with :class:`~kathdb.executor.codegen.CodeGenerator`, runs it on a
leased worker (diagnosing + regenerating on failure), then offers function saves."""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

import pandas as pd
from langchain_core.runnables.config import RunnableConfig

from ..common.function_manager import FunctionManager
from ..common.logger import get_logger
from ..executor.codegen import CodeGenerator, CodegenInState
from ..executor.codegen.codegen import _build_consumer_demands_map
from ..executor.codegen.codegen_tree import FAOExecutionError, FAOExecutableNode, walk_nodes
from ..plan_gen.plan_node import FAONode, topo_layers
from ..worker import (
    KathDBWorkerError,
    KathDBWorkerExecuteError,
    KathDBWorkerInstallError,
    KathDBWorkerLoadError,
    WorkerClient,
    WorkerManager,
)
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

        # Set by run().
        self._worker: WorkerClient | None = None
        self._worker_manager: WorkerManager | None = None
        # Error handler, built lazily on first failure.
        self._error_handler: ExecutionErrorHandler | None = None
        self._handler_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Run: dependency-driven codegen + execution
    # ------------------------------------------------------------------

    def run(
        self,
        cg_in: CodegenInState,
        *,
        worker: WorkerClient | None = None,
        worker_manager: WorkerManager | None = None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Code-generate and execute every operator of ``cg_in["logical_plan"]``;
        return the execution context (input relations plus every materialized output).

        Operators run as soon as their inputs are materialized, up to
        ``worker_manager.max_workers`` at a time; a static *worker* runs sequentially.
        Prefer *worker_manager*: poisoned workers are replaced between attempts.
        """
        if worker is None and worker_manager is None:
            raise ValueError("Executor.run requires worker or worker_manager")
        self._worker = worker
        self._worker_manager = worker_manager
        self._code_gen.config = config
        root = cg_in["logical_plan"]

        materialized_outputs: dict[str, pd.DataFrame] = {
            name: df
            for name, df in zip(cg_in["input_rel_names"], cg_in["input_rel"])
            if isinstance(df, pd.DataFrame)
        }
        result_ctx: dict[str, Any] = {
            name: df for name, df in zip(cg_in["input_rel_names"], cg_in["input_rel"])
        }

        layers = topo_layers(root)
        nodes: list[FAONode] = [n for layer in layers for n in layer]
        if not nodes:
            logger.warning("Executor: no processable nodes; nothing to do.")
            self._last_code_tree = None
            return result_ctx
        n_slots = 1 if worker_manager is None else max(1, getattr(worker_manager, "max_workers", 1))
        logger.info(
            "[run] starting: %d node(s) in %d dependency level(s), up to %d in parallel",
            len(nodes),
            len(layers),
            n_slots,
        )

        depth_of = {id(n): d for d, layer in enumerate(layers) for n in layer}
        siblings_of = {
            id(n): [(s.op, s.description) for s in layer if s is not n]
            for layer in layers
            for n in layer
        }
        # "consumer of X" (the downstream-op hint) and "producer of X" (readiness).
        parent_action_for: dict[str, str | None] = {}
        producer_of: dict[str, FAONode] = {}
        for n in nodes:
            for inp in n.inputs:
                parent_action_for.setdefault(inp, n.description or n.op)
            for out in n.outputs:
                producer_of.setdefault(out, n)
        consumed = set(parent_action_for)
        consumer_demands_map = _build_consumer_demands_map(root)

        produced_pp: dict[str, FAOExecutableNode] = {}
        state_lock = threading.Lock()
        remaining: list[FAONode] = list(nodes)
        running: dict[Any, FAONode] = {}
        last_sink: FAOExecutableNode | None = None

        def is_ready(n: FAONode) -> bool:
            return all(
                inp in materialized_outputs
                for inp in n.inputs
                if inp in producer_of
            )

        t0 = time.time()
        pool = ThreadPoolExecutor(max_workers=n_slots)
        try:
            while remaining or running:
                free = n_slots - len(running)
                ready = [n for n in remaining if is_ready(n)][: max(0, free)]
                for n in ready:
                    remaining.remove(n)
                    with state_lock:
                        snapshot = dict(materialized_outputs)
                        deps = [produced_pp[i] for i in n.inputs if i in produced_pp]
                    fut = pool.submit(
                        self._run_node,
                        n,
                        snapshot,
                        deps,
                        siblings_of[id(n)],
                        next(
                            (parent_action_for[o] for o in n.outputs if o in parent_action_for),
                            None,
                        ),
                        cg_in,
                        consumer_demands_map,
                        depth_of[id(n)],
                    )
                    running[fut] = n
                    logger.info(
                        "[run] dispatched %s (level %d; %d running, %d waiting)",
                        n.op,
                        depth_of[id(n)],
                        len(running),
                        len(remaining),
                    )
                if not running:
                    stuck = [n.op for n in remaining]
                    raise RuntimeError(
                        f"Plan cannot progress: no ready operator among {stuck} "
                        "(missing producer or cycle)"
                    )
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    n = running.pop(fut)
                    plan_node, outputs = fut.result()  # re-raises a node's failure
                    with state_lock:
                        materialized_outputs.update(outputs)
                        for out in plan_node.outputs:
                            produced_pp[out] = plan_node
                    if not any(o in consumed for o in n.outputs):
                        last_sink = plan_node
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

        logger.info("[run] DONE: %d node(s) in %.1fs", len(nodes), time.time() - t0)
        self._last_code_tree = last_sink
        result_ctx.update(materialized_outputs)
        logger.info(
            "Executor: codegen+exec finished; result_ctx has %d table(s)", len(result_ctx)
        )
        return result_ctx

    def _run_node(
        self,
        node: FAONode,
        materialized: dict[str, pd.DataFrame],
        deps: list[FAOExecutableNode],
        sibling_meta: list[tuple[str, str | None]],
        parent_action: str | None,
        cg_in: CodegenInState,
        consumer_demands_map: dict[str, list[dict]],
        depth: int,
    ) -> tuple[FAOExecutableNode, dict[str, pd.DataFrame]]:
        """One operator: codegen, then execution on a leased worker (runs in a
        scheduler thread; ``materialized`` is the snapshot taken when it became ready)."""
        plan_node, _code = self._code_gen.generate(
            node,
            materialized,
            sibling_meta,
            parent_action,
            cg_in,
            consumer_demands_map,
            depth,
        )
        if deps:
            plan_node.replace_children(deps)
        exec_ctx: dict[str, Any] = dict(materialized)
        plan_node = self._execute_with_regen(plan_node, exec_ctx, layer_idx=depth)
        outputs = {o: exec_ctx[o] for o in plan_node.outputs if o in exec_ctx}
        return plan_node, outputs

    # ------------------------------------------------------------------
    # Execute one node with diagnosis-driven recovery
    # ------------------------------------------------------------------

    def _get_error_handler(self) -> ExecutionErrorHandler:
        """Lazily build (and cache) an execution error handler."""
        with self._handler_lock:
            if self._error_handler is None:
                self._error_handler = ExecutionErrorHandler(
                    diagnosis_llm=self._code_gen.diagnosis_llm,
                    regenerate_fn=self._code_gen._regenerate_node,
                    fn_manager=self._fn_manager,
                    max_retries=self._code_gen.max_retries,
                )
            return self._error_handler

    def _lease_worker(self, current: WorkerClient | None) -> WorkerClient:
        """Worker for the next attempt: keep the leased one, or replace it if a prior
        attempt killed it. A manager without ``acquire`` is re-asked per attempt."""
        mgr = self._worker_manager
        if mgr is None:
            if self._worker is None:
                raise RuntimeError(
                    "Executor has no worker; pass worker= or worker_manager= to run()"
                )
            return self._worker
        if hasattr(mgr, "acquire"):
            if current is None:
                return mgr.acquire()
            return current if current.is_alive() else mgr.replace(current)
        return mgr.get_worker()

    def _release_worker(self, worker: WorkerClient | None) -> None:
        mgr = self._worker_manager
        if worker is not None and mgr is not None and hasattr(mgr, "release"):
            mgr.release(worker)

    @staticmethod
    def _is_infra_error(exc: Exception, error_str: str) -> bool:
        """Infrastructure failure (install, timeout, dead channel): retry the same code."""
        if isinstance(exc, KathDBWorkerInstallError):
            return True
        if isinstance(exc, (EOFError, BrokenPipeError, OSError)):
            return True
        if isinstance(exc, KathDBWorkerExecuteError) and "worker timed out after" in (
            error_str or ""
        ):
            return True
        return False

    def _execute_with_regen(
        self,
        plan_node: FAOExecutableNode,
        exec_ctx: dict[str, Any],
        *,
        layer_idx: int,
    ) -> FAOExecutableNode:
        """Execute *plan_node*, recovering on worker errors via the error handler.

        Holds one worker lease for all attempts; a poisoned worker is swapped for
        a fresh one before the next attempt.
        """
        max_attempts = self._code_gen.max_retries + 1
        worker: WorkerClient | None = None
        try:
            for attempt in range(max_attempts):
                attempt_no = attempt + 1
                worker = self._lease_worker(worker)
                plan_node, done = self._attempt_or_regen(
                    plan_node, exec_ctx, worker, attempt_no, max_attempts, layer_idx
                )
                if done:
                    return plan_node
        finally:
            self._release_worker(worker)
        return plan_node  # pragma: no cover

    def _attempt_or_regen(
        self,
        plan_node: FAOExecutableNode,
        exec_ctx: dict[str, Any],
        worker: WorkerClient,
        attempt_no: int,
        max_attempts: int,
        layer_idx: int,
    ) -> tuple[FAOExecutableNode, bool]:
        """One execution attempt; on failure diagnose + regenerate (or retry as-is).

        Returns ``(node for the next attempt, done)``; ``done`` is True on success.
        """
        op_name = plan_node.op
        exec_t0 = time.time()
        try:
            plan_node.execute(exec_ctx, profile=True, worker=worker)
            exec_dt = time.time() - exec_t0

            for out_name in plan_node.outputs or []:
                df = exec_ctx.get(out_name)
                if isinstance(df, pd.DataFrame):
                    logger.info(
                        "[exec] op=%s layer=%d attempt=%d output=%s "
                        "shape=%dx%d cols=%s",
                        op_name,
                        layer_idx,
                        attempt_no,
                        out_name,
                        df.shape[0],
                        df.shape[1],
                        list(df.columns),
                    )
            logger.info(
                "[exec] op=%s layer=%d attempt=%d SUCCESS in %.3fs",
                op_name,
                layer_idx,
                attempt_no,
                exec_dt,
            )
            return plan_node, True
        except (
            KathDBWorkerInstallError,
            KathDBWorkerLoadError,
            KathDBWorkerExecuteError,
            KathDBWorkerError,
            FAOExecutionError,
            # Dead worker / pipe: retried on a fresh worker.
            EOFError,
            BrokenPipeError,
            OSError,
        ) as exc:
            exec_dt = time.time() - exec_t0
            error_str = getattr(exc, "underlying_error", str(exc))
            logger.warning(
                "[exec] op=%s layer=%d attempt=%d/%d FAILED in %.3fs: %s: %s",
                op_name,
                layer_idx,
                attempt_no,
                max_attempts,
                exec_dt,
                type(exc).__name__,
                error_str[:200],
            )
            if attempt_no >= max_attempts:
                raise

            for out in plan_node.outputs:
                exec_ctx.pop(out, None)

            # Infra failure: retry the same code, skip diagnosis.
            if self._is_infra_error(exc, error_str):
                logger.warning(
                    "[exec] op=%s layer=%d attempt=%d infra error (%s); "
                    "retrying same code on a fresh worker",
                    op_name,
                    layer_idx,
                    attempt_no,
                    type(exc).__name__,
                )
                return plan_node, False

            handler = self._get_error_handler()
            regen_t0 = time.time()
            try:
                fixed_node = handler.handle(
                    plan_node,
                    error_str,
                    exec_ctx,
                    worker=worker,
                    config=self._code_gen.config,
                )
            except Exception as regen_exc:
                logger.error(
                    "[codegen_retry] op=%s layer=%d attempt=%d "
                    "FAILED in %.3fs: %s",
                    op_name,
                    layer_idx,
                    attempt_no + 1,
                    time.time() - regen_t0,
                    str(regen_exc)[:500],
                )
                raise
            regen_dt = time.time() - regen_t0

            if fixed_node is not plan_node:
                fixed_node.replace_children(plan_node.children)
                op_changed = "regenerated"
            else:
                op_changed = "param_patched"
            logger.info(
                "[codegen_retry] op=%s layer=%d attempt=%d "
                "%s in %.3fs (next exec attempt incoming)",
                op_name,
                layer_idx,
                attempt_no + 1,
                op_changed,
                regen_dt,
            )
            return fixed_node, False

    # ------------------------------------------------------------------
    # Function save logic
    # ------------------------------------------------------------------

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
        t = threading.Thread(
            target=self._background_save_worker,
            args=(save_info,),
            daemon=True,
        )
        t.start()
        self._save_threads.append(t)

    def _background_save_worker(self, save_info: dict) -> None:
        """Finalize the code with the LLM (canonical name, typed params, CONTRACT),
        smoke-test it, and write it to ``generated_fn/``."""
        # The finalizer / smoke harness are only needed when a save is accepted.
        from ..common.fn_smoke import run_smoke
        from ..common.function_finalizer import finalize_with_llm

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
