"""Prompt templates for code generation, revision and failure diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping, Sequence

from ...config import DEFAULT_AI_OP_TEMPERATURE
from ...common.logger import get_logger

logger = get_logger(__name__)


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


__all__ = [
    "format_available_libs_block",
    "format_physical_revision_prompt",
    "format_failure_diagnosis_prompt",
    "format_codegen_prompt",
]


_GOOD_PRACTICE_MD_PATH = Path(__file__).parent / "good_practice.md"
_GOOD_PRACTICE_TEXT: str | None = None


def _load_good_practice_md() -> str:
    global _GOOD_PRACTICE_TEXT
    if _GOOD_PRACTICE_TEXT is None:
        _GOOD_PRACTICE_TEXT = _GOOD_PRACTICE_MD_PATH.read_text(encoding="utf-8").strip()
    return _GOOD_PRACTICE_TEXT


_REVISION_CONSTRAINED_PKG_INSTRUCTION = (
    "You MUST only use pre-installed packages listed in the "
    "`## Available Libraries` section (plus the Python standard library). "
    "Do NOT install packages at runtime."
)

# Logical-optimization hints; the physical-execution policy is in _phys_exec_rule().
_EFFICIENCY_HINTS = (
    "**Optimization:** Optimize freely — apply any rewrite that does less work or "
    "makes fewer AI/LLM calls (filter early, skip rows that cannot affect the "
    "result, dedup/distinct, short-circuit, reorder work so a downstream "
    "limit/top-k/existence or short-circuit condition is reached sooner, reuse "
    "intermediates) as long as the "
    "final output stays semantically identical to the naive result and matches "
    "the required output schema."
)


_IMAGE_DETAIL_LOW_RULE = 'Attach images through `call_model(..., image_detail="low")` (its default). '

# phy_opt=False: HOW each model call is made is pinned (one call per item, no
# batching / cascades / concurrency); only logical rewrites may cut cost.
_PHYS_PINNED_RULE = (
    "Issue model calls one at a time in a plain loop — no hand-written "
    "batching/combining multiple items into one call, model cascades, "
    "parallelism, or threading (this constrains how each call is made, not "
    "how many). "
)
# phy_opt=True: physical execution may be optimized too.
_PHYS_FREE_RULE = (
    "You may also optimize HOW calls are made: batch or combine several items into "
    "one model call, run calls concurrently, use a cheap model or a cheap check as a "
    "first pass (a cascade) before the expensive call, and cache repeated calls — as "
    "long as the answer for every row stays the same. "
)


def _phys_exec_rule(image_detail_low: bool = True, phy_opt: bool = True) -> str:
    """Physical-execution policy: minimize total model tokens; ``phy_opt`` pins or
    frees HOW calls are made; ``image_detail_low`` requests ``detail="low"`` images."""
    rule = (
        (_PHYS_FREE_RULE if phy_opt else _PHYS_PINNED_RULE)
        + (_IMAGE_DETAIL_LOW_RULE if image_detail_low else "")
        + "Execution cost is the TOTAL tokens the model processes — summed over every "
        "call, prompt plus completion — so minimize that total (not any single "
        "proxy). Two levers that interact: issue fewer calls, and keep each call's "
        "payload small. The dominant term is usually an expensive payload (an "
        "attached image is worth hundreds-to-thousands of tokens) sent on many "
        "calls, so the biggest lever is showing each expensive item to the model "
        "the FEWEST times — ideally read/classify it once and reuse the result. "
        "Before writing the body, reason explicitly: how many calls does your "
        "approach make and how large is each, as a function of the input row "
        "counts? Then write the construction that minimizes the total while staying "
        "correct. The model should look at each distinct item once; do everything "
        "else with ordinary non-model code — relational filtering/joining/dedup "
        "before the model; giving one call enough context to settle the question in "
        "a single judgement (cheap text in the prompt beats re-sending images); "
        "reusing an extracted attribute instead of re-deriving it; stopping once "
        "the answer is determined. Watch the growth rate above all: never let total "
        "tokens scale with a product of two inputs (e.g. judging every pair, which "
        "re-sends each image once per candidate) when a single pass over one input "
        "— extract once, then match in code — answers the query."
    )
    rule += (
            " You MAY call the provided `kathdb.fn` operators"
            + (
                " (pass `max_concurrency=1` so each issues one model call per item)"
                if not phy_opt
                else ""
            )
            + ". They are "
            "not magic and not free: an operator issues exactly one model call for "
            "every item you hand it and processes all of them — it cannot filter, "
            "dedup, combine, or stop early on its own. So its call count is just "
            "whatever you feed it; estimate that the same way and use an operator "
            "only when it is genuinely the fewest-call construction for this query."
    )
    return rule


def _efficiency_hints_block(image_detail_low: bool = True, phy_opt: bool = True) -> str:
    return _EFFICIENCY_HINTS + " " + _phys_exec_rule(image_detail_low, phy_opt)


def _format_temperature(value: float) -> str:
    """Render a Python float literal for prompt-level sampling instructions."""
    return repr(float(value))


PHYSICAL_PLAN_REVISION_PROMPT = dedent(
    """
## System
You are an expert Python engineer specialising in refining function implementations for KathDB.

## Instructions
You must revise the current implementation to faithfully satisfy the provided instruction.
You cannot change the input/output signature of the function (DataFrame inputs only; per-query literals stay hardcoded inline).
{package_instructions}
If an expected output schema is provided below, the returned DataFrame MUST contain exactly those columns with the specified dtypes. Do not add, remove, or rename any columns from the expected schema.

**Apply the SMALLEST POSSIBLE change** that resolves the issue described in the guidance — change a single literal, swap one call, fix one column reference, etc. Do NOT rewrite the function from scratch. Preserve the existing structure, naming, and unrelated logic verbatim. Only touch lines whose change is required to fix the reported error.

Return only the full, updated Python source enclosed in a single Markdown code fence and nothing else.

## Implementation Guidance
{revision_instruction}

## Current Implementation
```python
{implementation}
```
{output_schema_block}
"""
).strip()


FAILURE_DIAGNOSIS_PROMPT = dedent(
    """
## System
You are an expert Python debugger diagnosing a function execution issue in KathDB, a multi-modal database.

## Function Reference
{fn_reference_block}

## Instructions
A function needs revision after a failed execution. Read the function reference above and the error below carefully — understand what the function does, what it produces, and what specifically went wrong.

The function takes only DataFrame relations as inputs; any per-query
literals are hardcoded inline. Diagnose the root cause of the error
and produce ``new_function_guidance`` as a **minimum-change patch
description** — the smallest possible edit that fixes the error.
Examples of valid guidance:

  - "Change the prompt string on line N from `…` to `…`"
  - "Swap the model id `gpt-4o-mini` for `gpt-4o`"
  - "The column reference `Brand` should be `brand` (case mismatch)"
  - "Wrap the `re.search` call in `or ''` so None inputs don't crash"

Do NOT propose full rewrites or refactors. Do NOT propose changing the
function signature. Identify the single smallest edit (a literal, a
column name, a method call, a None guard) that resolves the error.

## Node Description
{fn_description}

## NL Query
{nl_query}

{issue_block}
## Implementation Code
```python
{implementation}
```

## Input Samples
{input_samples}

{output_schema_block}{available_tables_block}{reuse_info_block}{human_feedback_block}
"""
).strip()


_CALL_MODEL_SIG = (
    '`call_model(prompt: str, model: str, media=None, *, modality: str | None = None, '
    'image_detail: str = "low", reasoning_effort: str = "minimal", temperature: float = 0.0) -> str`'
)


def format_available_libs_block() -> str:
    """Return the pinned ``## Available Libraries`` prompt section."""
    return (
        "\n\n## Available Libraries\n"
        "IMPORTANT: Use ONLY these third-party packages (plus the Python "
        "standard library — `re`, `json`, `csv`, `os`, `pathlib`, "
        "`base64`, `dataclasses`, etc. — are always available). Do NOT "
        "use `pip install`, `subprocess`, or any other mechanism to "
        "install packages at runtime.\n"
        "- `from kathdb.common.model_call import call_model` — the ONLY way to call "
        "a model: " + _CALL_MODEL_SIG + "; `media` = image/audio path(s), URL(s) or data URI(s), text goes "
        "in `prompt`. Never call litellm or a provider SDK directly.\n"
        "- pandas\n"
        "- pydantic\n"
        "- scipy"
    )


# Op-kind steering: the planner tags every atom SEMANTIC (model inference) or
# RELATIONAL (pandas only); codegen must respect the tag.
_OP_KIND_STEERING_SEMANTIC = (
    "\n\nOp-kind steering: this node was tagged **SEMANTIC** by the planner "
    "— the description requires LLM/VLM inference. Do NOT substitute a "
    "regex / `str.split` / first-token heuristic for the semantic extraction, "
    "even when the input column looks regular. See the `## Good Practices` "
    "section for response-normalization guidance."
)

_OP_KIND_STEERING_RELATIONAL = (
    "\n\nOp-kind steering: this node was tagged **RELATIONAL** by the "
    "planner — implement it with pandas only on the input DataFrames. Do "
    "NOT introduce any LLM/VLM call for this node."
)


def _op_kind_steering(
    op_kind: str | None,
    op_kind_rewrite: Mapping[str, str] | None = None,
) -> str:
    kind = (op_kind or "").strip().upper()
    if kind.startswith("SEMANTIC"):
        return _OP_KIND_STEERING_SEMANTIC
    if kind.startswith("RELATIONAL"):
        base = _OP_KIND_STEERING_RELATIONAL
        if op_kind_rewrite:
            origin = (op_kind_rewrite.get("from") or "").strip().upper()
            if origin.startswith("SEMANTIC"):
                rationale = (op_kind_rewrite.get("rationale") or "").strip()
                evidence = (op_kind_rewrite.get("evidence") or "").strip()
                base += (
                    "\n\nThis node was downgraded from SEMANTIC to RELATIONAL by "
                    "demand propagation because its input/output value spaces are "
                    "constrained to a closed finite domain. "
                    f"Rationale: {rationale or '(none provided)'} "
                    f"Evidence: {evidence or '(none provided)'} "
                    "Implement using ONLY pandas/equality matching — no LLM/VLM "
                    "calls, no fuzzy matching, no regex normalization of free text. "
                    "The upstream producers are contractually emitting values from "
                    "the closed domain identified above; treat the node's input "
                    "columns as drawn from that domain even if the description "
                    "reads as open-ended."
                )
        return base
    return ""


def format_physical_revision_prompt(
    *,
    implementation: str,
    revision_instruction: str,
    output_descriptions: list[str] | None = None,
    input_descriptions: list[str] | None = None,
    nl_query: str | None = None,
) -> str:
    """Revision prompt: the failing implementation + diagnosis-derived guidance."""
    prompt = PHYSICAL_PLAN_REVISION_PROMPT.format(
        implementation=implementation.strip(),
        revision_instruction=revision_instruction.strip(),
        package_instructions=_REVISION_CONSTRAINED_PKG_INSTRUCTION,
        output_schema_block=_format_output_schema_block(output_descriptions),
    )
    prompt += format_available_libs_block()
    if input_descriptions:
        prompt += "\n\n## Input Relation Descriptions\n" + "\n".join(input_descriptions)
    if nl_query:
        prompt += f"\n\n## NL Query\n{nl_query.strip()}"
    return prompt


def format_failure_diagnosis_prompt(
    *,
    fn_description: str | None = None,
    nl_query: str | None = None,
    implementation: str,
    error_trace: str,
    input_samples: Mapping[str, Any] | None = None,
    output_descriptions: list[str] | None = None,
    is_reuse: bool = False,
    fn_name: str | None = None,
    fn_source: str | None = None,
    fn_docs: str | None = None,
    human_feedback: str | None = None,
    available_tables: list[str] | None = None,
) -> str:
    """Failure-diagnosis prompt. *fn_docs* (a reused library function's fn.md) is the
    function reference when given; otherwise the implementation itself is."""
    if fn_docs:
        fn_reference_block = (
            f"This node uses predefined function: {fn_name}\n\n" f"{fn_docs.strip()}\n"
        )
        if fn_source:
            fn_reference_block += (
                f"\nSource code:\n```python\n{fn_source.strip()}\n```\n"
            )
    else:
        fn_reference_block = (
            "This is a custom-generated function (no predefined documentation).\n"
            f"Implementation:\n```python\n{implementation.strip()}\n```\n"
        )

    error_text = error_trace.strip() if error_trace else ""
    has_real_error = bool(error_text)
    if has_real_error and human_feedback:
        issue_block = (
            f"Execution error:\n{error_text}\n\n"
            f"Human feedback:\n{human_feedback.strip()}\n"
        )
    elif has_real_error:
        issue_block = f"Execution error:\n{error_text}\n"
    elif human_feedback:
        issue_block = (
            "The function executed successfully but a reviewer identified "
            "issues with the output.\n\n"
            f"Reviewer feedback:\n{human_feedback.strip()}\n"
        )
    else:
        issue_block = f"Error:\n{error_text or 'Unknown error'}\n"

    if is_reuse:
        reuse_info_block = (
            f"This is a REUSE node using fn: {fn_name}\n"
            "If you choose 'new_function', a completely new function will be "
            "generated (the original fn is NOT modified).\n\n"
        )
    else:
        reuse_info_block = ""

    # human_feedback is folded into issue_block above.
    human_feedback_block = ""

    if available_tables:
        table_list = ", ".join(available_tables)
        available_tables_block = (
            "Available tables in DBContext (the function may read additional "
            f"data from these):\n{table_list}\n\n"
        )
    else:
        available_tables_block = ""

    input_samples_text = json.dumps(
        input_samples or {},
        indent=2,
        ensure_ascii=False,
        cls=_NumpyJSONEncoder,
    ).strip()

    return FAILURE_DIAGNOSIS_PROMPT.format(
        fn_description=(fn_description or "").strip(),
        fn_reference_block=fn_reference_block,
        nl_query=(nl_query or "").strip(),
        implementation=implementation.strip(),
        issue_block=issue_block,
        input_samples=input_samples_text,
        output_schema_block=_format_output_schema_block(output_descriptions),
        available_tables_block=available_tables_block,
        reuse_info_block=reuse_info_block,
        human_feedback_block=human_feedback_block,
    )


# ---------------------------------------------------------------------------
# Per-node code-generation prompt (used by the layered CodeGenerator).
# ---------------------------------------------------------------------------

CODEGEN_PROMPT = dedent(
    """
## System
You are an expert Python engineer for KathDB (multi-modal: tables, text, images, audio, video).
Given input/output schemas below, implement a Python function for this node.

{query_context_block}## Node
Name: {op_name}
Input relation names: {input_relation_names}
Description: {description}

## Inputs
{input_details}

{input_expectations_block}
{child_node_changes_block}
## Output
Output relation name: {output_relation}
{output_schema_block}

## Instructions
Produce one minimal, efficient function for this node. Here, "minimal, efficient" means the smallest correct implementation that satisfies the node requirements while avoiding unnecessary columns, scans, joins, reshaping, and model/API calls.

**Relevant functions:** See the `## Relevant Functions` section at the end of this prompt (if present). You can use them directly, add logic around them, or write your own code. DEFAULT TO WRITING YOUR OWN CODE — import a function ONLY if its `## Behavior` trace convinces you it produces exactly the logic this node needs; an almost-right function that differs in one behavioral detail (pairs-once vs every combination, counts vs excludes a catch-all label, capped vs all rows) silently gives a wrong answer. When unsure, write your own, or reuse only the part you are sure of. Avoid duckdb/SQL for relational work. Check each operator's Cost Warning before using.

**Signature** — DataFrame-only (one DataFrame arg per input relation, no other arguments):
```python
def {fn_name}({input_rel_names}):
    ...
```

- **Hardcode all per-query literals inline** in the function body — thresholds, keywords, prompts, model ids, category lists, filter labels, etc. Do NOT add non-DataFrame named arguments to the function signature.
{package_instructions}
{return_type_rule}
- If an output schema is given below, match it exactly (columns and dtypes).
- Follow the `## Good Practices` section below for all other code-shape, schema, and LLM/VLM-prompt rules.

{save_worthiness_block}

{related_operators_instructions}

{efficiency_hints}
{model_constraint}

## Good Practices
{good_practices_block}

{function_docs_block}

{implementation_guidance_block}
"""
).strip()


# Save-worthiness rubric: save only when BOTH non-trivial control flow AND a
# token-cutting optimization hold.
_SAVE_WORTHINESS = (
    "**Save-worthiness (`new_fn_worth_saving`):** a coding agent runs afterward and "
    "generalizes whatever you save (lifts hardcoded literals into parameters), so keep "
    "them inline here AND do not judge save-worthiness by them — query-specific "
    "constants (a threshold, a prompt string) are EXPECTED and are never a reason to "
    "decline. Set "
    "`new_fn_worth_saving=true` ONLY when BOTH hold (otherwise `false`; default "
    "`false` if unsure):\n"
    "1) non-trivial control flow — not a one-liner or a thin wrapper around a single "
    "library/operator call;\n"
    "2) a token-cutting execution optimization that issues FEWER model calls than the "
    "naive operator version (early-exit / short-circuit, pre-filter, model cascade).\n"
    "Judge ONLY these two structural conditions — 'tightly coupled to this query' / "
    "'not generally reusable' is NOT a valid reason for `false`; a coding agent "
    "generalizes it afterward. A fused early-exit loop over a hardcoded threshold is "
    "`true`."
)


def format_codegen_prompt(
    *,
    fn_name: str = "execute",
    fn_description: str | None = None,
    nl_query: str | None = None,
    input_rel_names: list[str],
    input_descriptions: list[str],
    schema_descriptions: dict[str, str],
    relation_attributes: dict[str, list[str]],
    output_relation: str | list[str],
    output_descriptions: list[str] | None = None,
    output_schema_description: str | None = None,
    function_docs: str = "",
    related_operator_names: list[str] | None = None,
    implementation_guidance: str | None = None,
    child_node_change_summaries: dict[str, str] | None = None,
    consumer_demands: list[dict] | None = None,
    op_kind: str | None = None,
    op_kind_rewrite: dict[str, str] | None = None,
    optimization_rationale: str | None = None,
    member_atoms: list[str] | None = None,
    member_descriptions: list[str] | None = None,
    ai_op_model: str | None = None,
    ai_op_temperature: float = DEFAULT_AI_OP_TEMPERATURE,
    maximize_logical_optimization: bool = False,
    image_detail_low: bool = True,
    phy_opt: bool = True,
    inputs_are_sample: bool = False,
) -> str:
    """Per-node code-generation prompt (one DataFrame argument per input relation;
    per-query literals hardcoded inline)."""
    # --- input details ---
    input_detail_parts: list[str] = []
    for i, input_name in enumerate(input_rel_names):
        sample_text = input_descriptions[i] if i < len(input_descriptions) else ""
        has_sample_data = sample_text and sample_text != "(no rows)"

        if has_sample_data:
            input_detail_parts.append(f"Relation `{input_name}`:\n{sample_text}")
        else:
            desc = schema_descriptions.get(input_name)
            if desc is not None:
                desc = desc.replace("(0 rows)", "(schema only — not yet materialized)")
                input_detail_parts.append(
                    f"Relation `{input_name}` (planned intermediate "
                    f"— use ONLY these column names):\n{desc}"
                )
            else:
                known_attrs = relation_attributes.get(input_name, [])
                attrs_text = ", ".join(known_attrs) if known_attrs else "unknown"
                input_detail_parts.append(
                    f"Relation `{input_name}` is an intermediate relation. "
                    f"Known attributes: {attrs_text}."
                )
    rendered_inputs = "\n".join(f"- {item}" for item in input_detail_parts) or "- None"
    # State whether the rows shown are a plan-time sample or the full input.
    if inputs_are_sample:
        data_shown = (
            "Data shown: a SAMPLE (the shapes above are sample sizes). The full data "
            "this function runs on is larger and contains values not shown here — any "
            "shortcut keyed on the values you see (a column test, a string match, a "
            "lookup) must keep the model call as the fallback for rows it does not cover."
        )
    else:
        data_shown = (
            "Data shown: the FULL input (the shapes above are the real row counts); the "
            "per-column value lists are truncated to the first distinct values, so a "
            "'(+N more)' marker means the column has values not listed here."
        )
    rendered_inputs = f"{data_shown}\n{rendered_inputs}"

    # --- package instructions ---
    combined_pkg_instructions = (
        "- Operate on input relations as data (you may open referenced media). "
        "Use ONLY the packages listed in `## Available Libraries` (plus the "
        "Python standard library). No runtime installs."
    )

    # --- output schema blocks ---
    output_block = _format_output_schema_block(output_descriptions)
    if output_block:
        combined_output_schema = output_block.strip()
    else:
        combined_output_schema = _format_output_schema_block(
            output_schema_description
        ).strip()

    # --- related kathdb.fn operators ---
    related_operators_instructions = _format_related_operators_instructions(
        related_operator_names or []
    )

    # --- function docs ---
    function_docs_block = function_docs if function_docs else ""

    # --- model constraint ---
    ai_op_temperature_literal = _format_temperature(ai_op_temperature)
    if ai_op_model:
        model_instruction = (
            f'- Inference: use exactly ``model="{ai_op_model}"`` for all '
            "AI/LLM operations (text+image+audio). Do not download model "
            "weights, use GPU/local transformers, or substitute another model id."
        )
    else:
        model_instruction = (
            "- Inference: use the LiteLLM model id specified in the query's "
            "instructions for all AI/LLM operations (text+image+audio). Do not "
            "download model weights or use GPU/local transformers, and do not "
            "substitute another model id."
        )
    model_constraint = (
        f"{model_instruction}\n"
        "- Sampling: every ``call_model(...)`` call MUST pass "
        f"``temperature={ai_op_temperature_literal}`` (yes/no determinism "
        "matters for downstream early-stopping logic; do not omit it, even "
        "if an example elsewhere only shows the model id)."
    )

    # --- implementation guidance ---
    guidance_text = (implementation_guidance or "").strip()
    implementation_guidance_block = ""
    if guidance_text:
        implementation_guidance_block = f"## Implementation Guidance\n{guidance_text}\n"
    implementation_guidance_block += _op_kind_steering(op_kind, op_kind_rewrite)

    if isinstance(output_relation, list):
        _output_names = output_relation if output_relation else ["unknown"]
    else:
        _output_names = [output_relation or "unknown"]

    if len(_output_names) == 1:
        output_relation_rendered = _output_names[0]
        return_type_rule = (
            "- Real data only — no placeholders. Return whichever Python "
            "value-shape (DataFrame, dict, list, scalar, etc.) best fits the operation."
        )
    else:
        output_relation_rendered = ", ".join(_output_names)
        return_type_rule = (
            f"- This node produces **{len(_output_names)} output relations**: "
            f"{output_relation_rendered}. "
            f"Return a Python **list** of exactly {len(_output_names)} DataFrames "
            f"in that order."
        )

    query_context_block = (
        (
            "## Original Query\n"
            "This node is one step in a pipeline answering the overall "
            "natural-language query below. Use it only as context for this "
            "node's intent — implement THIS node, not the whole query:\n"
            f"{nl_query.strip()}\n\n"
        )
        if nl_query and nl_query.strip()
        else ""
    )

    # Fused group (>1 member): fused objective, no generic efficiency hints.
    is_grouped = len(member_atoms or []) > 1
    save_worthiness_block = _SAVE_WORTHINESS

    prompt = CODEGEN_PROMPT.format(
        fn_name=fn_name,
        query_context_block=query_context_block,
        save_worthiness_block=save_worthiness_block,
        input_rel_names=", ".join(input_rel_names),
        input_details=rendered_inputs,
        op_name=fn_name,
        description=fn_description or "None provided.",
        input_relation_names=", ".join(input_rel_names) if input_rel_names else "None",
        output_relation=output_relation_rendered,
        output_schema_block=combined_output_schema,
        package_instructions=combined_pkg_instructions,
        related_operators_instructions=related_operators_instructions,
        function_docs_block=function_docs_block,
        model_constraint=model_constraint,
        implementation_guidance_block=implementation_guidance_block,
        input_expectations_block=_format_consumer_demands_block(
            consumer_demands, _output_names
        ),
        child_node_changes_block=_format_child_node_changes_block(
            child_node_change_summaries
        ),
        efficiency_hints=(
            "" if is_grouped else _efficiency_hints_block(image_detail_low, phy_opt)
        ),
        return_type_rule=return_type_rule,
        good_practices_block=_load_good_practice_md(),
    )

    member_steps_block = _format_member_steps_block(
        member_atoms or [],
        member_descriptions or [],
        maximize_logical_optimization,
    )
    rationale_block = _format_optimization_rationale_block(optimization_rationale)
    logical_opt_block = (
        _logical_optimization_objective_block(
            is_grouped=is_grouped, image_detail_low=image_detail_low, phy_opt=phy_opt
        )
        if maximize_logical_optimization
        else ""
    )
    # Order: sub-steps -> objective -> optimizer rationale.
    extra_blocks = (member_steps_block + logical_opt_block + rationale_block).rstrip()
    if extra_blocks:
        marker = "## Instructions"
        prompt = prompt.replace(marker, f"{extra_blocks}\n\n{marker}", 1)

    prompt += format_available_libs_block()
    return prompt


# ---------------------------------------------------------------------------
# Prompt-block helpers
# ---------------------------------------------------------------------------


def _format_output_schema_block(
    output_schema_description: str | Sequence[str] | None,
) -> str:
    if not output_schema_description:
        return ""
    if isinstance(output_schema_description, str):
        text = output_schema_description
    else:
        text = "\n".join(str(d) for d in output_schema_description)
    cleaned = text.replace("(0 rows)", "(schema only — not yet materialized)")
    return (
        "Predicted output schema (if casing conflicts with input data, "
        "input data is authoritative):\n" + cleaned
    )


def _format_optimization_rationale_block(rationale: str | None) -> str:
    """``## Optimization Rationale`` block (empty when no rationale)."""
    text = (rationale or "").strip()
    if not text:
        return ""
    return (
        "## Optimization Rationale\n"
        "The LP grouping stage picked this fusion because:\n\n"
        f"{text}\n"
    )


def _logical_optimization_objective_block(
    is_grouped: bool = False, image_detail_low: bool = True, phy_opt: bool = True
) -> str:
    """Min-token objective: ``## Fuse These Steps`` for a fused group, the same cost
    goal scoped to one operator otherwise."""
    if is_grouped:
        return _fused_objective_block(image_detail_low, phy_opt)
    return (
        "## Logical Optimization Objective\n"
        "Make this operator as cheap as possible: cost = total execution LLM tokens "
        "(model calls × prompt+completion length). Within THIS operator: if the "
        "information it infers can be read directly from an existing column, read the "
        "column and do not call the model over the multimodal data; call the model once "
        "per distinct input value and map the answer back, not once per row; send the "
        "shortest prompt that settles the question (only the fields the call needs); when "
        "a text column can decide a row, prefer the cheaper text call and fall back to the "
        "image call only for rows the text cannot resolve. Implement EXACTLY this "
        "operator's semantics — every input row it is asked about, its declared output "
        "columns — and do not absorb upstream or downstream steps (filters, limits, "
        "aggregates) into it: which operators are fused is decided by the plan "
        "optimizer, not here.\n\n"
        "The output must be SEMANTICALLY equivalent to the straightforward implementation "
        "(same content; row order and dropped intermediate columns do not matter) AND "
        "must satisfy the downstream schema exactly — emit precisely the required output "
        "columns with the dtypes and value constraints stated above. Open vocabulary: a "
        "fast path that decides rows from observed values (a column test, a string match, "
        "a lookup over the values you see) must keep the model call as the fallback for "
        "every row it does not cover.\n"
    )


def _fused_objective_block(image_detail_low: bool = True, phy_opt: bool = True) -> str:
    """Fused-group objective block."""
    return (
        "## Fuse These Steps\n"
        "Produce one function whose result provably equals running the sub-steps above "
        "in order, and make it as cheap as possible. Cost = total TOKENS and total "
        "model CALLS; ANY rewrite that reduces either is valid — the examples below are "
        "hints, not a fixed list, so use whatever you find. For instance: send the "
        "shortest prompt that still works (pass only the fields a call needs; drop "
        "unused context/columns); dedup identical or provably-equivalent inputs and "
        "call the model once per distinct value; pre-filter rows on cheap observable "
        "columns before an AI call; reorder work and short-circuit once a downstream "
        "limit / exists / all-none / threshold is already decided."
        " You are NOT bound to reproduce each sub-step's implementation — only its "
        "result. The strongest rewrite: if the information a sub-step asks the MODEL to "
        "infer can be read directly from an existing column, read the column and do not "
        "call the model over the multimodal data; keep model calls only for attributes "
        "that genuinely require inference from the image/text/audio. (This is the "
        "inverse of the trap below: replacing a model call with the column it duplicates "
        "is always correct and strictly better.)"
        " The only hard limits: the output must equal the unoptimized answer and match "
        "the output schema above. One correctness trap: do not collapse per-item model "
        "judgments via equality / group-by / equi-join on an AI-extracted label unless "
        "that equivalence holds regardless of the data — when correctness depends on "
        "AI-derived values you cannot see, keep the model in the loop. Open vocabulary: "
        "a fast path that decides rows from observed values (a column test, a string "
        "match, a lookup over the values you see) must keep the model call as the "
        "fallback for every row it does not cover. "
        f"{_phys_exec_rule(image_detail_low, phy_opt)}\n"
    )


def _format_member_steps_block(
    member_atoms: list[str],
    member_descriptions: list[str],
    maximize_logical_optimization: bool = False,
) -> str:
    """``## Original Sub-Step Semantics`` block for a fused group."""
    if not member_atoms:
        return ""
    lines: list[str] = []
    for i, atom in enumerate(member_atoms):
        desc = member_descriptions[i] if i < len(member_descriptions) else ""
        desc = (desc or "").strip()
        if desc:
            lines.append(f"- **{atom}**: {desc}")
        else:
            lines.append(f"- **{atom}**")
    block = (
        "\n## Original Sub-Step Semantics\n"
        "Before fusion, the pipeline consisted of these atomic actions "
        "(in execution order). Each description specifies the exact logic "
        "(thresholds, formulas, scoring rules) that your fused implementation "
        "must preserve:\n" + "\n".join(lines) + "\n"
    )
    if maximize_logical_optimization:
        block += (
            "These describe how each atom judged in isolation; you see the whole "
            "group, which it could not. A 'use only X / ignore Y' note constrains "
            "how THAT step forms its judgment, not which rows you route to it — "
            "pre-filter, skip, or collapse rows by any cheap signal, as long as "
            "the final output stays semantically equivalent and schema-correct "
            "(see above). Do not deliberate further on the modality notes.\n"
        )
    return block


def _format_related_operators_instructions(names: list[str]) -> str:
    """Reuse instructions + per-function semantic parameter hints."""
    from ...common.function_manager import FunctionManager

    if not names:
        return ""

    lines = [
        "**Relevant `kathdb.fn` functions:** See the `## Relevant Functions` "
        "section below. You can use them directly, add logic around them, or "
        "write your own code. DEFAULT TO WRITING YOUR OWN CODE — import one "
        "ONLY if its `## Behavior` trace convinces you it produces exactly "
        "the logic this node needs; when unsure, write your own or reuse "
        "only the part you are sure of.",
        "If you import a function, use its argument names for semantic parameters "
        "(e.g. `prompt`, `out_col`); no node-name prefix; omit DataFrame parameters "
        "(e.g. `df`, `left`, `right`)—the runtime wires those. If it has a `model` "
        "parameter, ALWAYS pass it explicitly with the same model string your own "
        "code would use — never rely on its default.",
    ]

    fm = FunctionManager()
    for fn_name in names:
        sem = fm.semantic_params(fn_name)
        if not sem:
            continue
        lines.append("")
        lines.append(f"If you use `{fn_name}`, expected semantic parameters:")
        for p in sem:
            default_info = f", default: {p.default}" if p.default is not None else ""
            lines.append(f"- `{p.name}` (type: {p.annotation or 'Any'}{default_info})")
        lines.append(
            "  (If a parameter defaults to `None` and the query does not need it, "
            "omit it from the `parameters` list so the default applies.)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Downstream-demand / child-change blocks
# ---------------------------------------------------------------------------


def _unique_required_cols(demands: list[dict]) -> list[str]:
    """First-seen-ordered unique ``required_columns`` names across *demands*."""
    cols: list[str] = []
    seen: set[str] = set()
    for cd in demands:
        for col in cd.get("required_columns", []):
            name = col.get("name", "")
            if name and name not in seen:
                seen.add(name)
                cols.append(name)
    return cols


def _render_demand_lines(demands: list[dict], *, indent: str) -> list[str]:
    """Render the per-consumer column / value-constraint bullets for *demands*."""
    lines: list[str] = []
    col_indent = indent + "  "
    for cd in demands:
        consumer = cd.get("consumer", "unknown")
        if cd.get("is_final_output", False):
            lines.append(
                f"{indent}This is the final user-facing output. It must contain:"
            )
        else:
            lines.append(
                f"{indent}The next node in the pipeline (`{consumer}`) reads this "
                f"frame and requires these columns:"
            )
        for col in cd.get("required_columns", []):
            lines.append(
                f"{col_indent}- `{col.get('name', '?')}` ({col.get('dtype', '?')}): "
                f"{col.get('reason', '')}"
            )
        for vc in cd.get("value_constraints", []):
            lines.append(
                f"{col_indent}- Constraint on `{vc.get('column', '?')}`: "
                f"{vc.get('constraint', '')}"
            )
    return lines


def _format_consumer_demands_block(
    consumer_demands: list[dict] | None,
    output_names: list[str] | None = None,
) -> str:
    """``## Downstream Column Requirements`` block. A multi-output (fused) node gets
    one section per output relation, demands bucketed by ``output_relation``."""
    if not consumer_demands:
        return ""

    out_order = [o for o in (output_names or []) if o]

    # ---- single output frame ----
    if len(out_order) <= 1:
        parts: list[str] = []
        for cd in consumer_demands:
            consumer = cd.get("consumer", "unknown")
            if cd.get("is_final_output", False):
                parts.append("This is the final user-facing output. It must contain:")
            else:
                parts.append(
                    f"The next node in the pipeline (`{consumer}`) reads this "
                    f"node's output and requires these columns:"
                )
            for col in cd.get("required_columns", []):
                parts.append(
                    f"  - `{col.get('name', '?')}` ({col.get('dtype', '?')}): "
                    f"{col.get('reason', '')}"
                )
            for vc in cd.get("value_constraints", []):
                parts.append(
                    f"  - Constraint on `{vc.get('column', '?')}`: "
                    f"{vc.get('constraint', '')}"
                )
        cols = _unique_required_cols(consumer_demands)
        if cols:
            col_list = ", ".join(f"`{c}`" for c in cols)
            parts.append(
                f"\n**Output column constraint:** Your output DataFrame MUST "
                f"contain exactly these columns: {col_list}. "
                f"Do not include extra columns."
            )
        return "## Downstream Column Requirements\n" + "\n".join(parts)

    # ---- multiple output frames: one section per output relation ----
    by_output: dict[str | None, list[dict]] = {}
    for cd in consumer_demands:
        by_output.setdefault(cd.get("output_relation") or None, []).append(cd)

    # Declared outputs first, then any tagged relation not declared.
    ordered_rels: list[str] = list(out_order)
    for rel in by_output:
        if rel is not None and rel not in ordered_rels:
            ordered_rels.append(rel)

    parts = [
        f"This node is a fused group: it returns a **list of {len(out_order)} "
        f"DataFrames** in this exact order: "
        + ", ".join(f"`{o}`" for o in out_order)
        + ". Each output frame has its OWN required columns below — populate "
        "each DataFrame with exactly its own columns and do not merge columns "
        "across frames."
    ]
    for idx, rel in enumerate(ordered_rels, start=1):
        demands = by_output.get(rel, [])
        parts.append("")
        parts.append(f"### Output `{rel}` (DataFrame #{idx} in the returned list)")
        if not demands:
            parts.append(
                "  - No recorded downstream column demands; include exactly the "
                "columns this sub-step is defined to produce."
            )
            continue
        parts.extend(_render_demand_lines(demands, indent="  "))
        cols = _unique_required_cols(demands)
        if cols:
            col_list = ", ".join(f"`{c}`" for c in cols)
            parts.append(
                f"  - **Constraint:** DataFrame `{rel}` MUST contain exactly "
                f"these columns: {col_list}. Do not include extra columns."
            )

    untagged = by_output.get(None)
    if untagged:
        parts.append("")
        parts.append("### Additional downstream demands (output frame not identified)")
        parts.extend(_render_demand_lines(untagged, indent="  "))

    return "## Downstream Column Requirements\n" + "\n".join(parts)


def _format_child_node_changes_block(
    summaries: dict[str, str] | None,
) -> str:
    """``## Child Node Changes`` block (empty when no summaries)."""
    if not summaries:
        return ""
    lines = [
        "## Child Node Changes",
        "The following upstream (child) nodes had their logic or parameters "
        "modified during generation. Their output DataFrames may differ from "
        "the original specification. Consider whether your function needs to adapt:",
        "",
    ]
    for op_name, impact in summaries.items():
        lines.append(f"- Node `{op_name}`: {impact}")
    return "\n".join(lines)
