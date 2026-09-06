"""An empty function library must parse with the original atomic prompt and schema,
regardless of ``fn_coarsening``."""

from __future__ import annotations

from kathdb.parser.parser import ActionNLParserWithFunctions
from kathdb.parser.response_schemas import (
    ActionItem,
    ActionItemWithFunctions,
    ActionSketchResponse,
    ActionSketchWithFunctionsResponse,
)

# Marker text that appears ONLY in the function-directed (coarsening) prompt.
_COARSEN_MARKER = "PREFER ONE coarse action"
_FN_BLOCK_MARKER = "## Available Functions"


class _FakeRC:
    def describe_all_tables(self):
        return ["styles_details(id, price)"]

    def list_tables(self):
        return ["styles_details"]

    def is_view(self, name):
        return False


class _FakeFM:
    """Minimal FunctionManager stub: empty or non-empty library."""

    def __init__(self, names):
        self._names = list(names)

    def discover_functions(self):
        return {n: object() for n in self._names}

    def render_functions_summary(self):
        body = "\n".join(self._names) if self._names else "(none registered)"
        return f"{_FN_BLOCK_MARKER}\n{body}"


def _atomic_response():
    return ActionSketchResponse(
        actions=[
            ActionItem(
                name="a", action="x", inputs=["styles_details"], output="o",
                output_type="dataframe", op_kind="SEMANTIC",
            )
        ],
    )


def _fn_response():
    return ActionSketchWithFunctionsResponse(
        actions=[
            ActionItemWithFunctions(
                name="a", action="x", inputs=["styles_details"], output="o",
                output_type="dataframe", op_kind="SEMANTIC",
                selected_functions=[],
            )
        ],
    )


def _run_draft(fn_names, *, fn_coarsening=True):
    """Drive _draft_sketch_node once; return (captured_prompt, captured_schema)."""
    parser = ActionNLParserWithFunctions(
        clarification_llm=None, sketch_llm=None, revision_llm=None,
        fn_manager=_FakeFM(fn_names), fn_coarsening=fn_coarsening,
    )
    captured = {}

    def _fake_invoke(prompt, llm=None, schema=None):
        captured["prompt"] = prompt
        captured["schema"] = schema
        return _atomic_response() if schema is ActionSketchResponse else _fn_response()

    parser._invoke_structured = _fake_invoke  # type: ignore[assignment]
    state = {"q_in": "find cheap white socks", "relation_context": _FakeRC()}
    parser._draft_sketch_node(state)
    return captured["prompt"], captured["schema"]


def test_empty_library_uses_original_atomic_prompt_even_with_coarsening_on():
    prompt, schema = _run_draft([], fn_coarsening=True)
    # Original atomic prompt: no coarsening steer, no functions block.
    assert _COARSEN_MARKER not in prompt
    assert _FN_BLOCK_MARKER not in prompt
    assert schema is ActionSketchResponse


def test_nonempty_library_uses_function_directed_coarsening_prompt():
    prompt, schema = _run_draft(["classify_white_socks"], fn_coarsening=True)
    assert _COARSEN_MARKER in prompt
    assert _FN_BLOCK_MARKER in prompt
    assert schema is ActionSketchWithFunctionsResponse


def test_empty_library_matches_original_regardless_of_coarsening_flag():
    # With no functions, the coarsening flag must not change the prompt at all.
    p_off, s_off = _run_draft([], fn_coarsening=False)
    p_on, s_on = _run_draft([], fn_coarsening=True)
    assert p_off == p_on
    assert s_off is s_on is ActionSketchResponse
