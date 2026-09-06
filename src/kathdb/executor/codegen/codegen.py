"""Per-operator code generation and execution.

:class:`CodeGenerator` schedules a logical plan's operators by data dependency
(an operator runs once its inputs are materialized; up to ``max_workers`` at a
time), generates Python for each, executes it on a leased worker, and on failure
diagnoses + regenerates the code via :class:`ExecutionErrorHandler`.
"""

from __future__ import annotations

import ast
import copy
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Mapping, TYPE_CHECKING, TypeVar

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel

from ...config import DEFAULT_AI_OP_MODEL
from ...common.function_manager import FunctionManager
from ...common.logger import get_logger
from ...common.utils import invoke_structured_with_retry
from ...plan_gen.plan_node import FAONode
from ...worker import (
    KathDBWorkerError,
    KathDBWorkerExecuteError,
    KathDBWorkerInstallError,
    KathDBWorkerLoadError,
    WorkerClient,
    WorkerManager,
)
from .codegen_tree import FAOExecutionError, FAOExecutableNode, parse_llm_function
from .prompts import (
    format_codegen_prompt,
    format_physical_revision_prompt,
)
from .response_schemas import ConstrainedCodeGenerationResponse
from .state_schemas import CodegenInState, CodegenOutState
from .utils import map_input_relation_objects

if TYPE_CHECKING:
    from ..error_handler import ExecutionErrorHandler


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = ["CodeGenerator"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_consumer_demands_map(root: FAONode) -> dict[str, list[dict]]:
    """Map each output relation to the list of consumer demand dicts."""
    result: dict[str, list[dict]] = {}
    for node in root.iter_postorder():
        for cd in getattr(node, "consumer_demands", None) or []:
            for out in node.outputs:
                result.setdefault(out, []).append(cd)
    return result


def _demanded_columns_for_input(
    input_relation: str,
    consumer_op: str,
    member_atoms: list[str] | None,
    consumer_demands_map: dict[str, list[dict]],
) -> set[str] | None:
    """Return the columns *consumer_op* demands from *input_relation*.

    For fused nodes, *member_atoms* lists original atom names (demands
    were stamped pre-grouping under those names).  Returns ``None`` when
    no demand info is available so the caller can fall back to full schema.
    """
    demands = consumer_demands_map.get(input_relation)
    if not demands:
        return None
    candidates = {consumer_op}
    if member_atoms:
        candidates.update(member_atoms)
    cols: set[str] = set()
    matched = False
    for cd in demands:
        if cd.get("consumer") in candidates:
            matched = True
            for col in cd.get("required_columns", []):
                name = col.get("name", "")
                if name:
                    cols.add(name)
    if not matched:
        return None
    return cols if cols else None


def _wants_fused_optimization(
    cg_in: CodegenInState, member_atoms: list[str] | None
) -> bool:
    """Emit the min-token objective: for every fused node and for the optimizer's
    plan-time base-plan pass (so the ranker compares like with like)."""
    if cg_in.get("_grouping_base_plan"):
        return True
    return len(member_atoms or []) > 1


def _describe_value_shape(
    name: str,
    value: Any,
    *,
    indent: str = "",
    depth: int = 0,
    max_depth: int = 3,
    max_items: int = 10,
    max_chars: int = 4000,
) -> str:
    """Describe a DataFrame / dict / list / scalar structurally for the prompt,
    capped at ``max_chars`` characters and ``max_depth`` levels."""
    if isinstance(value, pd.DataFrame):
        cols = ", ".join(f"{c} ({value[c].dtype})" for c in value.columns)
        head = f"Relation `{name}` (DataFrame): shape=({value.shape[0]}, {value.shape[1]}), columns: {cols}"
        return head if len(head) <= max_chars else head[: max_chars - 1] + "…"

    if isinstance(value, dict):
        keys = list(value.keys())
        parts = [f"{indent}Relation `{name}` (dict, {len(keys)} entries):"]
        if depth >= max_depth:
            parts.append(f"{indent}  - (max depth reached)")
        else:
            child_indent = indent + "  "
            for k in keys[:max_items]:
                child = _describe_value_shape(
                    f"{name}[{k!r}]",
                    value[k],
                    indent=child_indent,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_chars=max_chars,
                )
                parts.append(f"{child_indent}- key={k!r}: {child.lstrip()}")
            if len(keys) > max_items:
                parts.append(f"{child_indent}- (+{len(keys) - max_items} more keys)")
        out = "\n".join(parts)
        return out if len(out) <= max_chars else out[: max_chars - 1] + "…"

    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        n = len(value)
        head = f"{indent}Relation `{name}` ({type(value).__name__}, length={n})"
        if n == 0 or depth >= max_depth:
            return head
        sample = _describe_value_shape(
            f"{name}[0]",
            value[0],
            indent=indent + "  ",
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
            max_chars=max_chars,
        )
        out = f"{head}; first element:\n{sample}"
        return out if len(out) <= max_chars else out[: max_chars - 1] + "…"

    type_name = type(value).__name__
    try:
        preview = repr(value)
    except Exception:
        preview = "<unrepr>"
    if len(preview) > 200:
        preview = preview[:200] + "…"
    return f"{indent}Relation `{name}` ({type_name}): {preview}"


def render_distinct_samples(
    name: str,
    df: Any,
    *,
    k: int = 5,
    cell_char_limit: int = 60,
    from_sample: bool = False,
) -> str:
    """Render per-column distinct values of *df* for the codegen prompt.

    ``from_sample`` labels the header and counts as sample-relative (plan-time pass).
    Non-DataFrame inputs are rendered via ``_describe_value_shape``.
    """
    if not isinstance(df, pd.DataFrame):
        return _describe_value_shape(name, df)
    nrows, ncols = df.shape

    def _fmt_value(v: Any) -> str:
        if isinstance(v, str):
            v = v if len(v) <= cell_char_limit else v[: cell_char_limit - 3] + "..."
            return repr(v)
        try:
            s = json.dumps(v, default=str, ensure_ascii=False)
        except Exception:
            s = str(v)
        return s if len(s) <= cell_char_limit else s[: cell_char_limit - 3] + "..."

    if from_sample:
        parts: list[str] = [
            f"Relation `{name}`: SAMPLE of {nrows} rows x {ncols} columns "
            "(the full input has more rows and may have values not shown)"
        ]
    else:
        parts = [f"Relation `{name}`: shape=({nrows}, {ncols})"]
    for col in df.columns:
        nn = df[col].dropna()
        dtype = str(df[col].dtype)
        try:
            distinct = nn.unique()
            n_distinct = len(distinct)
            sample = list(distinct[:k])
            rendered = ", ".join(_fmt_value(v) for v in sample)
            marker = "" if n_distinct <= k else f" (+{n_distinct - k} more)"
            if from_sample:
                count = f"{n_distinct} distinct in the sample; other values likely"
            else:
                count = f"{n_distinct} distinct"
            parts.append(f"  - `{col}` ({dtype}, {count}): [{rendered}]{marker}")
        except Exception:
            # Unhashable (struct / list / dict) column: ``.unique()`` raises, so
            # render real sample values rather than an empty list.
            try:
                sample = list(nn.head(k))
                rendered = ", ".join(_fmt_value(v) for v in sample)
                parts.append(
                    f"  - `{col}` ({dtype}, nested/struct, {len(nn)} non-null): "
                    f"[{rendered}]"
                )
            except Exception:
                parts.append(f"  - `{col}` ({dtype}, n/a): []")
    return "\n".join(parts)


def render_sibling_context(
    siblings: list[tuple[str, str | None]],
    parent_action: str | None,
) -> str:
    """Render pipeline context: parallel nodes and the downstream operation."""
    if not siblings and not parent_action:
        return ""
    lines: list[str] = [
        "This node's output is combined with outputs from other nodes by a "
        "downstream operation. Understanding this context helps you produce "
        "correctly shaped output.",
    ]
    if siblings:
        lines.append("\nParallel nodes (their outputs are joined/combined with yours):")
        for op, desc in siblings:
            line = f"  - `{op}`"
            if desc:
                line += f": {desc}"
            lines.append(line)
    if parent_action:
        lines.append(
            f"\nDownstream operation that combines the outputs: {parent_action}"
        )
    return "## Pipeline Context\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# CodeGenerator
# ---------------------------------------------------------------------------


class CodeGenerator:
    """Layered per-node code synthesis used by the executor."""

    def __init__(
        self,
        *,
        generation_llm: BaseChatModel,
        diagnosis_llm: BaseChatModel,
        revision_llm: BaseChatModel,
        # Retry budget for codegen + error recovery
        max_retries: int = 3,
        # Distinct values per column rendered into the prompt
        distinct_value_sample_k: int = 25,
        # Concurrent codegen LLM calls (1 = serial)
        max_concurrent_generations: int = 1,
        # Model + temperature the generated code must use for AI calls
        ai_op_model: str | None = DEFAULT_AI_OP_MODEL,
        ai_op_temperature: float = 0.0,
        # Generated code attaches images with detail="low"
        image_detail_low: bool = True,
        # Generated code may batch / parallelize / cascade model calls
        phy_opt: bool = True,
        # Shared function library
        fn_manager: FunctionManager | None = None,
    ) -> None:
        self.generation_llm = generation_llm
        self.diagnosis_llm = diagnosis_llm
        self.revision_llm = revision_llm
        self.max_retries = max_retries
        self._fn_manager = fn_manager or FunctionManager()
        self.distinct_value_sample_k = distinct_value_sample_k
        self.max_concurrent_generations = max_concurrent_generations
        self.ai_op_model = ai_op_model
        self.ai_op_temperature = ai_op_temperature
        self.image_detail_low = image_detail_low
        self.phy_opt = phy_opt

        # Set by run().
        self._worker: WorkerClient | None = None
        self._worker_manager: "WorkerManager | None" = None
        # RunnableConfig forwarded into every codegen LLM call (set by run()).
        self._config: RunnableConfig | None = None
        # Error handler, built lazily on first failure.
        self._layered_error_handler: "ExecutionErrorHandler | None" = None
        self._handler_lock = threading.Lock()
        # Bounds concurrent codegen LLM calls independently of execution parallelism.
        self._codegen_sem = threading.BoundedSemaphore(max(1, max_concurrent_generations))

    # ------------------------------------------------------------------
    # Compile (no-op)
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """No-op; codegen + execution happen in :meth:`run`."""
        logger.info("CodeGenerator compiled (layered mode; no state graph).")

    # ------------------------------------------------------------------
    # LLM invocation helpers
    # ------------------------------------------------------------------

    def _invoke_structured(
        self,
        prompt: str,
        *,
        llm: BaseChatModel,
        schema: type[T],
        max_retries: int = 3,
        config: RunnableConfig | None = None,
    ) -> T:
        """Structured LLM call with retries; ``config`` falls back to ``self._config``."""
        return invoke_structured_with_retry(
            prompt,
            llm=llm,
            schema=schema,
            max_retries=max_retries,
            config=config if config is not None else self._config,
        )

    # ------------------------------------------------------------------
    # Build a FAOExecutableNode from raw LLM-emitted code
    # ------------------------------------------------------------------

    def _build_function_signature(
        self,
        code: str,
        fn_name: str,
        df_names: list[str],
        parameters: list[Any] | None,
    ) -> tuple[str, dict[str, str]]:
        """Rebuild the ``def`` line so it matches the runtime ``fn(**kwargs)`` call:
        DataFrame arg names are kept from the LLM's code, extra *parameters* are
        appended without defaults. Returns ``(patched_code, df_bindings)`` where
        *df_bindings* maps parameter names to context table names."""
        fallback_bindings = dict(zip(df_names, df_names))

        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.warning(
                "Cannot rewrite signature for %r: source is not valid Python",
                fn_name,
            )
            return code, fallback_bindings

        target_fn: ast.FunctionDef | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                target_fn = node
                break
        if target_fn is None:
            logger.warning(
                "Cannot rewrite signature for %r: no matching function def",
                fn_name,
            )
            return code, fallback_bindings

        n_defaults = len(target_fn.args.defaults)
        n_positional = len(target_fn.args.args)
        n_required = n_positional - n_defaults
        df_param_names = [
            a.arg for a in target_fn.args.args[: min(n_required, len(df_names))]
        ]
        for i in range(len(df_param_names), len(df_names)):
            df_param_names.append(df_names[i])
        df_set = set(df_param_names)

        new_args = [ast.arg(arg=name) for name in df_param_names]
        if parameters:
            for p in parameters:
                name = (getattr(p, "name", None) or "").strip()
                if not name or name in df_set:
                    continue
                raw_val = getattr(p, "value", None)
                if raw_val in ("null", "None", None):
                    continue
                new_args.append(ast.arg(arg=name))
                df_set.add(name)

        target_fn.args.args = new_args
        target_fn.args.defaults = []
        target_fn.args.posonlyargs = []
        target_fn.args.kwonlyargs = []
        target_fn.args.kw_defaults = []

        try:
            patched = ast.unparse(tree)
        except Exception:
            logger.warning(
                "ast.unparse failed for %r; using original code",
                fn_name,
                exc_info=True,
            )
            return code, fallback_bindings

        df_bindings = dict(zip(df_param_names, df_names))
        logger.info(
            "Built signature for %r: df_args=%s, param_args=%s",
            fn_name,
            df_param_names,
            [a.arg for a in new_args[len(df_param_names) :]],
        )
        return patched, df_bindings

    def _build_codegen_node(
        self,
        *,
        code: str,
        input_relations: Mapping[str, Any],
        op_name: str,
        outputs: list[str],
        fix_code_parsing_llm: BaseChatModel,
        implementation_guidance: str = "",
        required_packages: list[str] | None = None,
        pip_install_commands: list[str] | None = None,
        df_bindings: dict[str, str] | None = None,
    ) -> FAOExecutableNode:
        """Compile raw LLM-emitted code into an executable :class:`FAOExecutableNode`."""
        try:
            physical_fn = parse_llm_function(
                code,
                max_attempts=3,
                llm=fix_code_parsing_llm,
                required_packages=required_packages,
                pip_install_commands=pip_install_commands,
            )
        except FAOExecutionError:
            logger.error("Failed to compile codegen implementation.")
            raise

        metadata: dict[str, Any] = {
            "required_packages": list(physical_fn.requirements),
            "pip_install_commands": list(physical_fn.pip_commands),
            "runtime_script_path": str(physical_fn.script_path),
        }

        df_names = list(input_relations.keys())

        if df_bindings is None:
            try:
                sig = FunctionManager.extract_function_signature(code, op_name)
                fn_df_params = [p["name"] for p in sig if p["default"] is None][
                    : len(df_names)
                ]
            except Exception:
                fn_df_params = df_names
            if not fn_df_params:
                fn_df_params = df_names
            df_bindings = dict(zip(fn_df_params, df_names))

        return FAOExecutableNode(
            op=op_name,
            function=physical_fn,
            implementation_guidance=implementation_guidance,
            inputs=df_names,
            outputs=outputs,
            metadata=metadata,
            arguments={"bindings": df_bindings},
        )

    # ------------------------------------------------------------------
    # Regeneration on execution failure
    # ------------------------------------------------------------------

    def _regenerate_node(
        self,
        *,
        node: FAOExecutableNode,
        guidance: str,
        input_relation_objects: Mapping[str, Any],
    ) -> FAOExecutableNode:
        """Regenerate a failing node's code from the diagnosis guidance (revision
        prompt: smallest patch to the prior implementation)."""
        meta = node.metadata or {}
        # ``node.inputs`` (not the retry-time context, which may lack an input
        # after a mid-plan failure) is the source of truth for the runtime kwargs.
        df_names = (
            list(node.inputs)
            or meta.get("input_rel_names")
            or list(input_relation_objects.keys())
        )
        df_descriptions = meta.get("input_descriptions") or [
            f"Table `{n}`" for n in df_names
        ]
        prior_implementation = (
            getattr(node.function, "str_impl", None)
            or node.metadata.get("code_str")
            or ""
        )

        prompt = format_physical_revision_prompt(
            implementation=prior_implementation,
            revision_instruction=guidance,
            output_descriptions=meta.get("output_descriptions") or None,
            input_descriptions=df_descriptions,
            nl_query=meta.get("nl_query"),
        )

        code_response = self._invoke_structured(
            prompt, llm=self.generation_llm, schema=ConstrainedCodeGenerationResponse
        )

        # Derive bindings from the regenerated code so renamed parameters still wire.
        code, df_bindings = self._build_function_signature(
            code_response.code,
            node.op,
            df_names,
            None,
        )

        regen_outputs = list(node.outputs) if node.outputs else [node.op]
        new_node = self._build_codegen_node(
            code=code,
            input_relations={n: input_relation_objects.get(n) for n in df_names},
            op_name=node.op,
            outputs=regen_outputs,
            fix_code_parsing_llm=self.revision_llm,
            implementation_guidance=guidance,
            required_packages=[],
            pip_install_commands=[],
            df_bindings=df_bindings,
        )
        # Keep the fresh node's script/package metadata; fill the rest from the old node.
        for key, value in meta.items():
            new_node.metadata.setdefault(key, value)
        new_node.metadata["codegen_prompt"] = prompt
        return new_node

    # ------------------------------------------------------------------
    # Topological layering
    # ------------------------------------------------------------------

    @staticmethod
    def _topo_layers(root: FAONode) -> list[list[FAONode]]:
        """Group the plan's operators into topological layers (``input_relation``
        and the ``logical_plan`` root excluded)."""
        # Dedupe by ``op``: diamond plans hold duplicate instances of a shared node.
        nodes: list[FAONode] = []
        seen_ops: set[str] = set()
        for n in root.iter_postorder():
            if n is root or n.op in {"input_relation", "logical_plan"}:
                continue
            if n.op in seen_ops:
                continue
            seen_ops.add(n.op)
            nodes.append(n)
        producers: dict[str, FAONode] = {}
        for n in nodes:
            for o in n.outputs:
                prev = producers.get(o)
                if prev is not None and prev is not n:
                    logger.warning(
                        "_topo_layers: distinct nodes %r and %r both declare "
                        "output %r; keeping %r as producer",
                        prev.op,
                        n.op,
                        o,
                        prev.op,
                    )
                producers.setdefault(o, n)

        depth: dict[int, int] = {}
        for n in nodes:
            d = 0
            for inp in n.inputs:
                prod = producers.get(inp)
                if prod is not None and prod is not n:
                    pd_ = depth.get(id(prod))
                    if pd_ is None:
                        pd_ = 0
                    d = max(d, pd_ + 1)
            depth[id(n)] = d

        max_d = max(depth.values(), default=-1)
        layers: list[list[FAONode]] = [[] for _ in range(max_d + 1)]
        for n in nodes:
            layers[depth[id(n)]].append(n)
        return layers

    # ------------------------------------------------------------------
    # Codegen for one node
    # ------------------------------------------------------------------

    def _codegen_layered_node(
        self,
        node: FAONode,
        materialized_outputs: dict[str, pd.DataFrame],
        sibling_meta: list[tuple[str, str | None]],
        parent_action: str | None,
        cg_in: CodegenInState,
        consumer_demands_map: dict[str, list[dict]],
        layer_idx: int = -1,
    ) -> tuple[FAOExecutableNode, str]:
        """Generate code for *node*; returns ``(code_node, code_str)``. A hit in
        ``cg_in["_grouping_cache"]`` is cloned instead of calling the LLM."""
        cache = cg_in.get("_grouping_cache")
        if cache is not None and node.op in cache.codegen:
            cached_node, cached_code = cache.codegen[node.op]
            # Cached code whose declared outputs differ from this plan node's
            # would return the wrong relations: discard and regenerate.
            _want = [o for o in (node.outputs or []) if o]
            _have = [o for o in (cached_node.outputs or []) if o]
            if _want and _have != _want:
                logger.warning(
                    "[codegen] op=%s cached outputs %s != plan outputs %s -- "
                    "discarding cached code, regenerating",
                    node.op,
                    _have,
                    _want,
                )
                cache.codegen.pop(node.op, None)
            else:
                clone = copy.copy(cached_node)
                clone.metadata = dict(cached_node.metadata or {})
                clone.arguments = dict(cached_node.arguments or {})
                clone.children = []
                cache.codegen_hits += 1
                logger.info(
                    "[codegen] op=%s layer=%d cache-hit (skipping LLM call)",
                    node.op,
                    layer_idx,
                )
                return clone, cached_code

        if cache is not None:
            cache.codegen_misses += 1

        rc = cg_in["relation_context"]

        df_names: list[str] = []
        df_objs: list[pd.DataFrame] = []
        for v in node.inputs:
            df = materialized_outputs.get(v)
            if df is None:
                if rc.has_table(v):
                    df = rc.load_table(v)
                else:
                    raise KeyError(
                        f"Layered codegen: missing input '{v}' for node '{node.op}'"
                    )
            df_names.append(v)
            df_objs.append(df)

        if not df_names:
            raise ValueError(
                f"Layered codegen: no DataFrame inputs resolved for node '{node.op}'"
            )

        input_relation_objects = map_input_relation_objects(df_names, df_objs)

        # Prompt rendering only shows demanded columns; runtime binding keeps full frames.
        member_atoms = getattr(node, "member_atoms", None) or []
        df_objs_for_prompt: list[Any] = []
        for name, df in zip(df_names, df_objs):
            if isinstance(df, pd.DataFrame):
                demanded = _demanded_columns_for_input(
                    name, node.op, member_atoms, consumer_demands_map
                )
                if demanded is not None:
                    keep = [c for c in df.columns if c in demanded]
                    df_objs_for_prompt.append(df[keep] if keep else df)
                else:
                    df_objs_for_prompt.append(df)
            else:
                df_objs_for_prompt.append(df)

        input_descriptions = [
            render_distinct_samples(
                name,
                df,
                k=self.distinct_value_sample_k,
                cell_char_limit=60,
                from_sample=bool(cg_in.get("_grouping_base_plan")),
            )
            for name, df in zip(df_names, df_objs_for_prompt)
        ]

        schema_descriptions: dict[str, str] = {}
        relation_attributes: dict[str, list[str]] = {}
        for name, df in zip(df_names, df_objs_for_prompt):
            if isinstance(df, pd.DataFrame):
                relation_attributes[name] = list(df.columns)
                cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
                schema_descriptions[name] = (
                    f"Relation `{name}` (materialized): "
                    f"shape=({df.shape[0]}, {df.shape[1]}), columns: {cols}"
                )
            else:
                relation_attributes[name] = []
                schema_descriptions[name] = _describe_value_shape(name, df)
        for tname in rc.list_tables():
            if tname in schema_descriptions:
                continue
            desc = rc.describe_table(tname)
            if desc is not None:
                schema_descriptions[tname] = desc
            relation_attributes.setdefault(tname, rc.get_columns(tname))

        output_names = list(node.outputs) if node.outputs else [""]

        function_docs = ""
        related_operator_names: list[str] | None = None
        selected = list(getattr(node, "selected_functions", []) or [])
        if selected:
            related_operator_names = selected
            # Full fn.md: selection is advisory, codegen makes the final use/skip call.
            function_docs = self._fn_manager.render_functions_full_docs(
                names=related_operator_names
            )

        consumer_demands = getattr(node, "consumer_demands", None) or None
        if consumer_demands is None:
            for out in node.outputs:
                if out in consumer_demands_map and consumer_demands_map[out]:
                    consumer_demands = consumer_demands_map[out]
                    break

        node_rationale = getattr(node, "merge_rationale", None)
        member_atoms = getattr(node, "member_atoms", None) or []
        member_descriptions = getattr(node, "member_descriptions", None) or []

        prompt = format_codegen_prompt(
            fn_name=node.op,
            fn_description=node.description,
            nl_query=cg_in.get("q_in"),
            input_rel_names=df_names,
            input_descriptions=input_descriptions,
            schema_descriptions=schema_descriptions,
            relation_attributes=relation_attributes,
            output_relation=output_names,
            output_descriptions=None,
            output_schema_description=None,
            function_docs=function_docs,
            related_operator_names=related_operator_names,
            implementation_guidance="",
            child_node_change_summaries=None,
            consumer_demands=consumer_demands,
            op_kind=getattr(node, "op_kind", None),
            op_kind_rewrite=getattr(node, "op_kind_rewrite", None),
            optimization_rationale=node_rationale,
            member_atoms=member_atoms,
            member_descriptions=member_descriptions,
            ai_op_model=self.ai_op_model,
            ai_op_temperature=self.ai_op_temperature,
            maximize_logical_optimization=_wants_fused_optimization(cg_in, member_atoms),
            image_detail_low=self.image_detail_low,
            phy_opt=self.phy_opt,
            inputs_are_sample=bool(cg_in.get("_grouping_base_plan")),
        )

        sibling_block = render_sibling_context(sibling_meta, parent_action)
        if sibling_block:
            prompt = f"{prompt}\n\n{sibling_block}"

        codegen_t0 = time.time()
        try:
            response = self._invoke_structured(
                prompt,
                llm=self.generation_llm,
                schema=ConstrainedCodeGenerationResponse,
            )
        except Exception as exc:
            codegen_dt = time.time() - codegen_t0
            logger.error(
                "[codegen] op=%s layer=%d FAILED in %.3fs: %s: %s",
                node.op,
                layer_idx,
                codegen_dt,
                type(exc).__name__,
                exc,
            )
            raise
        codegen_dt = time.time() - codegen_t0

        code, df_bindings = self._build_function_signature(
            response.code,
            node.op,
            df_names,
            None,
        )

        new_fn_worth_saving = bool(getattr(response, "new_fn_worth_saving", False))
        new_fn_worth_saving_reason = (
            getattr(response, "new_fn_worth_saving_reason", "") or ""
        ).strip()
        logger.info(
            "[codegen] op=%s layer=%d codegen_time=%.3fs lines=%d "
            "new_fn_worth_saving=%s reason=%s",
            node.op,
            layer_idx,
            codegen_dt,
            len(code.splitlines()),
            new_fn_worth_saving,
            new_fn_worth_saving_reason or "(none)",
        )

        plan_node = self._build_codegen_node(
            code=code,
            input_relations=input_relation_objects,
            op_name=node.op,
            outputs=output_names,
            fix_code_parsing_llm=self.revision_llm,
            implementation_guidance="",
            required_packages=getattr(response, "required_packages", []),
            pip_install_commands=getattr(response, "pip_install_commands", []),
            df_bindings=df_bindings,
        )

        plan_node.metadata["fn_description"] = node.description
        plan_node.metadata["nl_query"] = cg_in.get("q_in")
        plan_node.metadata["input_rel_names"] = df_names
        plan_node.metadata["input_descriptions"] = input_descriptions
        plan_node.metadata["output_descriptions"] = None
        plan_node.metadata["selected_functions"] = list(selected)
        plan_node.metadata["new_fn_worth_saving"] = new_fn_worth_saving
        plan_node.metadata["new_fn_worth_saving_reason"] = new_fn_worth_saving_reason
        plan_node.metadata["codegen_prompt"] = prompt
        plan_node.metadata["ai_op_model"] = self.ai_op_model
        plan_node.metadata["ai_op_temperature"] = self.ai_op_temperature
        plan_node.metadata["member_atoms"] = getattr(node, "member_atoms", []) or []

        if cache is not None:
            cache.codegen[node.op] = (plan_node, code)

        return plan_node, code

    # ------------------------------------------------------------------
    # Execute one node with diagnosis-driven recovery
    # ------------------------------------------------------------------

    def _get_error_handler(self) -> "ExecutionErrorHandler":
        """Lazily build (and cache) an execution error handler."""
        with self._handler_lock:
            cached = self._layered_error_handler
            if cached is not None:
                return cached
            # Lazy import: circular via ``kathdb.executor.__init__``.
            from ..error_handler import ExecutionErrorHandler

            handler = ExecutionErrorHandler(
                diagnosis_llm=self.diagnosis_llm,
                regenerate_fn=self._regenerate_node,
                fn_manager=self._fn_manager,
                max_retries=self.max_retries,
            )
            self._layered_error_handler = handler
            return handler

    def _lease_worker(self, current: WorkerClient | None) -> WorkerClient:
        """Worker for the next attempt: keep the leased one, or replace it if a prior
        attempt killed it. A manager without ``acquire`` is re-asked per attempt."""
        mgr = self._worker_manager
        if mgr is None:
            if self._worker is None:
                raise RuntimeError(
                    "CodeGenerator has no worker; pass worker= or worker_manager= "
                    "to run()"
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
        max_attempts = self.max_retries + 1
        handler: "ExecutionErrorHandler | None" = None
        op_name = plan_node.op
        worker: WorkerClient | None = None
        try:
            for attempt in range(max_attempts):
                attempt_no = attempt + 1
                worker = self._lease_worker(worker)
                plan_node = self._attempt_or_regen(
                    plan_node, exec_ctx, worker, attempt_no, max_attempts, layer_idx
                )
                if plan_node is not None and getattr(plan_node, "_kathdb_done", False):
                    del plan_node._kathdb_done
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
    ) -> FAOExecutableNode:
        """One execution attempt; on failure diagnose + regenerate (or retry as-is).

        Returns the node to use for the next attempt, flagged ``_kathdb_done`` when
        the attempt succeeded.
        """
        handler: "ExecutionErrorHandler | None" = None
        op_name = plan_node.op
        if True:
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
                plan_node._kathdb_done = True
                return plan_node
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
                    return plan_node

                if handler is None:
                    handler = self._get_error_handler()

                regen_t0 = time.time()
                try:
                    fixed_node = handler.handle(
                        plan_node,
                        error_str,
                        exec_ctx,
                        worker=worker,
                        config=self._config,
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
                return fixed_node
        return plan_node  # pragma: no cover

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
    ) -> CodegenOutState:
        """Code-generate and execute every operator of ``cg_in["logical_plan"]``.

        Operators run as soon as their inputs are materialized, up to
        ``worker_manager.max_workers`` at a time (a static *worker* runs sequentially).
        The materialized outputs are stashed on the sink node's metadata.
        """
        if worker is None and worker_manager is None:
            raise ValueError("CodeGenerator.run requires worker or worker_manager")
        self._worker = worker
        self._worker_manager = worker_manager
        self._config = config
        rc = cg_in["relation_context"]
        root = cg_in["logical_plan"]

        layers = self._topo_layers(root)
        nodes: list[FAONode] = [n for layer in layers for n in layer]
        if not nodes:
            logger.warning("Codegen: no processable nodes; nothing to do.")
            return CodegenOutState(  # type: ignore[typeddict-item]
                q_in=cg_in["q_in"],
                actions=cg_in["actions"],
                relation_context=rc,
                input_rel_names=cg_in["input_rel_names"],
                input_rel=cg_in["input_rel"],
                code_tree=root,  # type: ignore[arg-type]
            )
        n_slots = 1 if worker_manager is None else max(1, getattr(worker_manager, "max_workers", 1))
        logger.info(
            "[run] starting: %d node(s) in %d dependency level(s), up to %d in parallel",
            len(nodes),
            len(layers),
            n_slots,
        )

        materialized_outputs: dict[str, pd.DataFrame] = {
            name: df
            for name, df in zip(cg_in["input_rel_names"], cg_in["input_rel"])
            if isinstance(df, pd.DataFrame)
        }
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

        # Stash materialized outputs on the sink code-node for the executor.
        root_pp = last_sink
        if root_pp is not None:
            if root_pp.metadata is None:
                root_pp.metadata = {}
            root_pp.metadata["_layered_materialized_outputs"] = materialized_outputs

        logger.info("[run] DONE: %d node(s) in %.1fs", len(nodes), time.time() - t0)

        return CodegenOutState(  # type: ignore[typeddict-item]
            q_in=cg_in["q_in"],
            actions=cg_in["actions"],
            relation_context=rc,
            input_rel_names=cg_in["input_rel_names"],
            input_rel=cg_in["input_rel"],
            code_tree=root_pp,  # type: ignore[arg-type]
        )

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
        with self._codegen_sem:
            plan_node, _code = self._codegen_layered_node(
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
