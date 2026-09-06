"""Execution-time error handling: LLM diagnosis + code regeneration for a failed node."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig

from ..common.function_manager import FunctionManager
from ..common.logger import get_logger
from ..common.utils import (
    invoke_structured_with_retry,
    sample_dataframe,
)
from ..executor.codegen.codegen_tree import FAOExecutableNode
from ..executor.codegen.prompts import format_failure_diagnosis_prompt
from ..executor.codegen.response_schemas import FailureDiagnosisResponse
from ..worker import WorkerClient

logger = get_logger(__name__)


class RegenerateNodeFn(Protocol):
    """Callback signature for code regeneration."""

    def __call__(
        self,
        *,
        node: FAOExecutableNode,
        guidance: str,
        input_relation_objects: dict[str, Any],
    ) -> FAOExecutableNode: ...


class ExecutionErrorHandler:
    """Diagnoses an execution failure with ``diagnosis_llm`` and regenerates the node's
    code through ``regenerate_fn`` (normally ``CodeGenerator._regenerate_node``).
    A reused library function's source and fn.md are added to the diagnosis prompt."""

    def __init__(
        self,
        *,
        diagnosis_llm: BaseChatModel,
        regenerate_fn: RegenerateNodeFn,
        fn_manager: FunctionManager,
        max_retries: int = 3,
    ) -> None:
        self._diagnosis_llm = diagnosis_llm
        self._regenerate_fn = regenerate_fn
        self._fn_manager = fn_manager
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle(
        self,
        node: FAOExecutableNode,
        error: str,
        context: dict[str, Any],
        *,
        worker: WorkerClient,
        config: RunnableConfig | None = None,
    ) -> FAOExecutableNode:
        """Diagnose *error* and return the regenerated node to retry."""
        meta = node.metadata or {}

        input_samples = self._build_input_samples(node, context)
        prompt = self._build_diagnosis_prompt(node, error, input_samples, meta)

        diagnosis = self._invoke_diagnosis(prompt, config=config)
        logger.info("Execution diagnosis: %s", diagnosis.reasoning)

        input_relation_objects = {
            name: context[name] for name in node.inputs if name in context
        }
        new_node = self._regenerate_fn(
            node=node,
            guidance=diagnosis.new_function_guidance,
            input_relation_objects=input_relation_objects,
        )
        new_node.replace_children(node.children)
        return new_node

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_input_samples(
        node: FAOExecutableNode, context: dict[str, Any]
    ) -> dict[str, str]:
        samples: dict[str, str] = {}
        for name in node.inputs:
            if name in context and isinstance(context[name], pd.DataFrame):
                sample = sample_dataframe(context[name], 3)
                samples[name] = sample.to_string(index=False)
        return samples

    def _build_diagnosis_prompt(
        self,
        node: FAOExecutableNode,
        error: str,
        input_samples: dict[str, str],
        meta: dict[str, Any],
    ) -> str:
        selected = meta.get("selected_functions") or []
        is_reuse = bool(selected)
        fn_name = selected[0] if selected else None

        fn_source = None
        fn_docs = None
        if is_reuse and fn_name:
            fn_source = self._fn_manager.read_function_file(
                fn_name, "scripts/fn.py"
            )
            fn_docs = self._fn_manager.read_function_file(fn_name, "fn.md") or None

        return format_failure_diagnosis_prompt(
            fn_description=meta.get("fn_description"),
            nl_query=meta.get("nl_query"),
            implementation=node.function.str_impl or "",
            error_trace=error,
            input_samples=input_samples,
            output_descriptions=meta.get("output_descriptions"),
            is_reuse=is_reuse,
            fn_name=fn_name,
            fn_source=fn_source,
            fn_docs=fn_docs,
        )

    def _invoke_diagnosis(
        self,
        prompt: str,
        *,
        config: RunnableConfig | None = None,
    ) -> FailureDiagnosisResponse:
        return invoke_structured_with_retry(
            prompt,
            llm=self._diagnosis_llm,
            schema=FailureDiagnosisResponse,
            max_retries=self._max_retries,
            config=config,
        )

