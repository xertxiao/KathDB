"""decide_persistence persists nothing when the LLM call fails."""

from __future__ import annotations

import pandas as pd

from kathdb.executor.codegen.codegen_tree import FAOExecutableNode, FAOFunction
from kathdb.executor.persistence import decide_persistence


class _ExplodingLLM:
    """LLM stand-in whose structured-output pipeline always raises."""

    def with_structured_output(self, schema):
        raise RuntimeError("simulated LLM outage")


def _make_plan() -> FAOExecutableNode:
    return FAOExecutableNode(
        op="answer_query",
        function=FAOFunction(name="answer_query", impl=lambda **kw: None),
        outputs=["answer"],
    )


def test_decide_persistence_fails_closed_on_llm_error():
    result_ctx = {
        "products": pd.DataFrame({"a": [1]}),  # input table, never a candidate
        "answer": pd.DataFrame({"b": [2]}),  # new table
        "intermediate": pd.DataFrame({"c": [3]}),  # new table
    }
    persisted = decide_persistence(
        _ExplodingLLM(),
        nl_query="which products have a red logo?",
        plan=_make_plan(),
        result_ctx=result_ctx,
        input_rel_names=["products"],
        skip_user_review=True,
    )
    assert persisted == []


def test_decide_persistence_no_new_tables_short_circuits():
    result_ctx = {"products": pd.DataFrame({"a": [1]})}
    persisted = decide_persistence(
        _ExplodingLLM(),
        nl_query="q",
        plan=_make_plan(),
        result_ctx=result_ctx,
        input_rel_names=["products"],
        skip_user_review=True,
    )
    assert persisted == []
