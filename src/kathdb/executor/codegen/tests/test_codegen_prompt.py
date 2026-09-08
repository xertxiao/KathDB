"""Smoke tests for the codegen prompt."""

from __future__ import annotations

import inspect

from kathdb.executor.codegen.prompts import (
    _format_consumer_demands_block,
    format_codegen_prompt,
)


_BASE_KWARGS = dict(
    fn_name="pick_top_brand",
    fn_description="Pick the top-scoring brand.",
    input_rel_names=["products"],
    input_descriptions=["sample products"],
    schema_descriptions={"products": "brand,name,price,image_path"},
    relation_attributes={"products": ["brand", "name", "price", "image_path"]},
    output_relation="answer",
)


def test_prompt_renders_without_rationale():
    prompt = format_codegen_prompt(**_BASE_KWARGS)
    assert "## Node" in prompt
    assert "pick_top_brand" in prompt
    assert "## Optimization Rationale" not in prompt
    assert "## Instructions" in prompt


def test_prompt_renders_rationale_block_when_present():
    prompt = format_codegen_prompt(
        **_BASE_KWARGS,
        optimization_rationale=(
            "Fuse SEM classify with the per-brand count + top-1 select; "
            "early-stop the inner classification once a brand locks in 5 "
            "positives."
        ),
    )
    assert "## Optimization Rationale" in prompt
    assert "early-stop the inner classification" in prompt
    # Block lands before ## Instructions.
    assert prompt.index("## Optimization Rationale") < prompt.index("## Instructions")


def test_prompt_empty_rationale_is_omitted():
    prompt = format_codegen_prompt(**_BASE_KWARGS, optimization_rationale="")
    assert "## Optimization Rationale" not in prompt
    prompt = format_codegen_prompt(**_BASE_KWARGS, optimization_rationale=None)
    assert "## Optimization Rationale" not in prompt


def test_prompt_signature_flags():
    """One rendering path; these are the only behaviour flags."""
    sig = inspect.signature(format_codegen_prompt)
    flags = {"maximize_logical_optimization", "image_detail_low", "phy_opt", "inputs_are_sample"}
    assert flags <= set(sig.parameters)


def test_logical_optimization_block_only_when_enabled():
    """The objective block appears only when ``maximize_logical_optimization`` is set."""
    off = format_codegen_prompt(**_BASE_KWARGS)
    assert "## Logical Optimization Objective" not in off

    on = format_codegen_prompt(**_BASE_KWARGS, maximize_logical_optimization=True)
    assert "## Logical Optimization Objective" in on
    # Lands before ## Instructions like the other pre-instruction blocks.
    assert on.index("Logical Optimization Objective") < on.index("## Instructions")
    # One-operator scope: cost goal, no-fusion boundary, semantic-equivalence guard.
    assert "semantically equal" in on
    assert "match the downstream schema exactly" in on
    assert "do not absorb upstream or downstream steps" in on
    assert "being measured" not in on and "fused sub-query" not in on


def test_fused_group_uses_lean_objective():
    """A fused group gets ``## Fuse These Steps`` after its sub-steps, no generic hints."""
    fused = format_codegen_prompt(
        **_BASE_KWARGS,
        maximize_logical_optimization=True,
        member_atoms=["classify_brand", "count_per_brand"],
        member_descriptions=["classify brand from image", "count rows per brand"],
    )
    assert "## Fuse These Steps" in fused
    assert "## Logical Optimization Objective" not in fused
    assert "branch-and-bound" not in fused
    assert "Any rewrite that reduces either is fine" in fused  # open-ended, hints not a closed allow-list
    assert "tokens and model calls" in fused  # cost = tokens AND calls
    assert "model-extracted label" in fused  # the one correctness trap is kept
    assert "## Original Sub-Step Semantics" in fused
    assert fused.index("## Original Sub-Step Semantics") < fused.index("## Fuse These Steps")
    assert "Save-worthiness (`new_fn_worth_saving`)" in fused
    assert "ONLY when BOTH hold" in fused
    assert "**Optimization:** Optimize freely" not in fused  # generic hints suppressed


def test_atomic_node_under_objective_keeps_generic_objective():
    """An atomic node under the objective gets the single-operator block, not the fused one."""
    atomic = format_codegen_prompt(**_BASE_KWARGS, maximize_logical_optimization=True)
    assert "Save-worthiness (`new_fn_worth_saving`)" in atomic
    assert "## Logical Optimization Objective" in atomic
    assert "## Fuse These Steps" not in atomic


def test_wants_fused_optimization_covers_final_run_fused_node():
    """The objective applies to fused nodes and to the plan-time base-plan pass."""
    from kathdb.executor.codegen.codegen import _wants_fused_optimization

    assert _wants_fused_optimization({"_grouping_base_plan": True}, [])
    assert _wants_fused_optimization({}, ["a", "b"])
    assert not _wants_fused_optimization({}, ["a"])
    assert not _wants_fused_optimization({}, None)


def test_prompt_has_no_query_context_section():
    prompt = format_codegen_prompt(**_BASE_KWARGS)
    assert "## Query Context" not in prompt
    assert "The full user query is" not in prompt



def test_codegen_prompt_enforces_ai_op_temperature():
    prompt = format_codegen_prompt(
        **_BASE_KWARGS,
        ai_op_model="openai/gpt-4o-mini",
    )

    assert 'model="openai/gpt-4o-mini"' in prompt
    assert "temperature=0.0" in prompt
    assert "do not omit it" in prompt


# ---------------------------------------------------------------------------
# Consumer-demands block: single vs. multi-output (fused group) rendering
# ---------------------------------------------------------------------------

_DEMAND_A = {
    "consumer": "filter_black_formal_pieces",
    "output_relation": "classified_pieces",
    "required_columns": [
        {"name": "brandName", "dtype": "VARCHAR", "reason": "grouping key"},
        {
            "name": "is_black_formal_piece",
            "dtype": "VARCHAR",
            "reason": "filter condition",
        },
    ],
    "value_constraints": [
        {"column": "is_black_formal_piece", "constraint": "one of ['yes', 'no']"}
    ],
}
_DEMAND_B = {
    "consumer": "rank_brands",
    "output_relation": "brand_counts",
    "required_columns": [
        {"name": "brandName", "dtype": "VARCHAR", "reason": "join key"},
        {"name": "n", "dtype": "BIGINT", "reason": "ordering"},
    ],
    "value_constraints": [],
}


def test_consumer_demands_single_output_is_flat():
    """One declared output → the original single-frame constraint wording."""
    block = _format_consumer_demands_block([_DEMAND_A], ["classified_pieces"])
    assert "## Downstream Column Requirements" in block
    assert "Your output DataFrame MUST contain exactly these columns:" in block
    assert "`brandName`, `is_black_formal_piece`" in block
    # No per-frame headers in single-output mode.
    assert "DataFrame #" not in block
    assert "list of" not in block


def test_consumer_demands_multi_output_renders_per_frame():
    """Two declared outputs → per-frame sections, no flattened constraint."""
    block = _format_consumer_demands_block(
        [_DEMAND_A, _DEMAND_B], ["classified_pieces", "brand_counts"]
    )
    # Announces the list-of-DataFrames contract in declared order.
    assert "list of 2" in block
    assert "`classified_pieces`, `brand_counts`" in block
    # One section per frame, numbered by returned-list position.
    assert "### Output `classified_pieces` (DataFrame #1 in the returned list)" in block
    assert "### Output `brand_counts` (DataFrame #2 in the returned list)" in block
    # Per-frame constraints — columns are NOT merged across frames.
    assert (
        "DataFrame `classified_pieces` MUST contain exactly these columns: "
        "`brandName`, `is_black_formal_piece`"
    ) in block
    assert (
        "DataFrame `brand_counts` MUST contain exactly these columns: "
        "`brandName`, `n`"
    ) in block
    # Single-frame wording must not appear for a fused group.
    assert "Your output DataFrame MUST contain exactly these columns" not in block
    # `n` belongs only to brand_counts, never leaks into classified_pieces' line.
    cp_section = block.split("### Output `brand_counts`")[0]
    assert "`n`" not in cp_section


def test_consumer_demands_multi_output_untagged_demand_isolated():
    """A demand lacking output_relation lands in the catch-all section."""
    untagged = dict(_DEMAND_B)
    untagged.pop("output_relation")
    block = _format_consumer_demands_block(
        [_DEMAND_A, untagged], ["classified_pieces", "brand_counts"]
    )
    assert "Additional downstream demands (output frame not identified)" in block
    # brand_counts has no tagged demand, so it surfaces with a no-demand note.
    assert "No recorded downstream column demands" in block


def test_consumer_demands_empty_returns_empty_string():
    assert _format_consumer_demands_block(None, ["a", "b"]) == ""
    assert _format_consumer_demands_block([], ["a", "b"]) == ""


def test_data_shown_line_states_sample_vs_full():
    full = format_codegen_prompt(**_BASE_KWARGS)
    assert "Data shown: the FULL input" in full and "SAMPLE" not in full
    sample = format_codegen_prompt(**_BASE_KWARGS, inputs_are_sample=True)
    assert "Data shown: a SAMPLE" in sample and "FULL input" not in sample
    assert sample.index("Data shown") < sample.index("## Instructions")


def test_open_vocabulary_fallback_rule_in_both_objectives():
    atomic = format_codegen_prompt(**_BASE_KWARGS, maximize_logical_optimization=True)
    fused = format_codegen_prompt(
        **_BASE_KWARGS,
        maximize_logical_optimization=True,
        member_atoms=["a", "b"],
        member_descriptions=["do a", "do b"],
    )
    for prompt in (atomic, fused):
        assert "keeps the model call as the fallback for the rows it does not cover" in prompt


def test_render_distinct_samples_marks_sample_relative_counts():
    import pandas as pd

    from kathdb.executor.codegen.codegen import render_distinct_samples

    df = pd.DataFrame({"category": ["books", "toys", "books"]})
    full = render_distinct_samples("products", df)
    assert "shape=(3, 1)" in full and "2 distinct)" in full and "sample" not in full
    sample = render_distinct_samples("products", df, from_sample=True)
    assert "SAMPLE of 3 rows" in sample
    assert "2 distinct in the sample; other values likely" in sample
