"""Demand propagation + op-kind rewrite, with ``ainvoke_structured_with_retry`` mocked."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from kathdb.plan_gen import plan_generator_base
from kathdb.plan_gen.plan_generator import PlanGenerator
from kathdb.plan_gen.plan_node import FAONode
from kathdb.plan_gen.response_schemas import (
    AllNodesDemandResponse,
    DemandedColumn,
    FinalOutputDemand,
    InputDemand,
    NodeDemand,
    OpKindDecision,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_classify_join_plan() -> tuple[FAONode, FAONode, FAONode, FAONode]:
    """Build a classify+classify -> join DAG.

    Returns (root, classify_text, classify_image, join).
    """
    root = FAONode(op="logical_plan")
    rel_text = FAONode(op="input_relation", outputs=["products"])
    rel_image = FAONode(op="input_relation", outputs=["photos"])
    classify_text = FAONode(
        op="classify_text",
        inputs=["products"],
        outputs=["text_labeled"],
        op_kind="SEMANTIC",
    )
    classify_image = FAONode(
        op="classify_image",
        inputs=["photos"],
        outputs=["image_labeled"],
        op_kind="SEMANTIC",
    )
    join = FAONode(
        op="join_formality",
        inputs=["text_labeled", "image_labeled"],
        outputs=["joined"],
        op_kind="SEMANTIC",
    )
    classify_text.children.append(rel_text)
    classify_image.children.append(rel_image)
    join.children.append(classify_text)
    join.children.append(classify_image)
    root.children.append(join)
    return root, classify_text, classify_image, join


def _grouped_node_plan() -> tuple[FAONode, FAONode]:
    """Build a tree with one GROUPED node that has no op_kind set."""
    root = FAONode(op="logical_plan")
    rel = FAONode(op="input_relation", outputs=["in_rel"])
    fused = FAONode(
        op="reviewed_fusion",
        inputs=["in_rel"],
        outputs=["fused_out"],
        type="GROUPED",
        op_kind=None,
        member_atoms=["filter_a", "filter_b"],
    )
    fused.children.append(rel)
    root.children.append(fused)
    return root, fused


def _make_generator() -> PlanGenerator:
    return PlanGenerator(lp_llm=object())


def _mock_response(decisions: list[OpKindDecision]) -> AllNodesDemandResponse:
    return AllNodesDemandResponse(
        final_output_demand=FinalOutputDemand(),
        node_demands=[],
        op_kind_decisions=decisions,
    )


def _patch_invoke(response: AllNodesDemandResponse):
    return patch.object(
        plan_generator_base,
        "ainvoke_structured_with_retry",
        new=AsyncMock(return_value=response),
    )


def _run_one_shot(gen: PlanGenerator, root: FAONode) -> None:
    root_children = [c for c in root.children if c.op != "input_relation"]
    asyncio.run(
        gen._apropagate_demands_one_shot(
            root,
            root_children,
            schema_descriptions={},
            nl_query="dummy",
            config=None,
        )
    )


# ---------------------------------------------------------------------------
# Tests: op_kind rewrite mutations
# ---------------------------------------------------------------------------


def test_downgrades_semantic_to_relational():
    root, classify_text, classify_image, join = _two_classify_join_plan()
    response = _mock_response(
        [
            OpKindDecision(
                node_id="text_labeled",
                parser_op_kind="SEMANTIC",
                chosen_op_kind="SEMANTIC",
                rationale="open-text input",
                evidence="products.description is free text",
            ),
            OpKindDecision(
                node_id="image_labeled",
                parser_op_kind="SEMANTIC",
                chosen_op_kind="SEMANTIC",
                rationale="image classification",
                evidence="photos.image is unstructured",
            ),
            OpKindDecision(
                node_id="joined",
                parser_op_kind="SEMANTIC",
                chosen_op_kind="RELATIONAL",
                rationale="both inputs collapse to {Formal, Non-Formal}",
                evidence="text_labeled.label and image_labeled.label share the same closed enum",
            ),
        ]
    )
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    assert join.op_kind == "RELATIONAL"
    assert join.op_kind_rewrite is not None
    assert join.op_kind_rewrite["from"] == "SEMANTIC"
    assert join.op_kind_rewrite["to"] == "RELATIONAL"
    assert "closed enum" in join.op_kind_rewrite["evidence"]

    # Unchanged nodes do not get a rewrite record.
    assert classify_text.op_kind == "SEMANTIC"
    assert classify_text.op_kind_rewrite is None
    assert classify_image.op_kind == "SEMANTIC"
    assert classify_image.op_kind_rewrite is None


def test_no_change_leaves_rewrite_none():
    root, classify_text, classify_image, join = _two_classify_join_plan()
    response = _mock_response(
        [
            OpKindDecision(
                node_id=node_id,
                parser_op_kind="SEMANTIC",
                chosen_op_kind="SEMANTIC",
                rationale="open domain",
                evidence="no closed value_constraint",
            )
            for node_id in ("text_labeled", "image_labeled", "joined")
        ]
    )
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    for n in (classify_text, classify_image, join):
        assert n.op_kind == "SEMANTIC"
        assert n.op_kind_rewrite is None


def test_grouped_node_assigned_op_kind():
    root, fused = _grouped_node_plan()
    response = _mock_response(
        [
            OpKindDecision(
                node_id="fused_out",
                parser_op_kind="UNSET",
                chosen_op_kind="RELATIONAL",
                rationale="fused filter chain runs on closed enum column",
                evidence="upstream demand pins filter column to {A, B}",
            ),
        ]
    )
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    assert fused.op_kind == "RELATIONAL"
    # UNSET -> RELATIONAL counts as a rewrite (audit trail).
    assert fused.op_kind_rewrite is not None
    assert fused.op_kind_rewrite["from"] == "UNSET"
    assert fused.op_kind_rewrite["to"] == "RELATIONAL"


def test_unknown_node_id_logged_and_skipped():
    root, _, _, join = _two_classify_join_plan()
    response = _mock_response(
        [
            OpKindDecision(
                node_id="does_not_exist",
                parser_op_kind="SEMANTIC",
                chosen_op_kind="RELATIONAL",
                rationale="hallucinated",
                evidence="n/a",
            ),
        ]
    )
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    # join must be untouched — the unknown decision is dropped.
    assert join.op_kind == "SEMANTIC"
    assert join.op_kind_rewrite is None


def test_invalid_chosen_value_skipped():
    root, _, _, join = _two_classify_join_plan()
    response = _mock_response(
        [
            OpKindDecision(
                node_id="joined",
                parser_op_kind="SEMANTIC",
                chosen_op_kind="MAYBE",
                rationale="invalid value",
                evidence="n/a",
            ),
        ]
    )
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    assert join.op_kind == "SEMANTIC"
    assert join.op_kind_rewrite is None


def test_empty_decisions_leaves_plan_unchanged():
    root, classify_text, classify_image, join = _two_classify_join_plan()
    response = _mock_response([])
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    for n in (classify_text, classify_image, join):
        assert n.op_kind == "SEMANTIC"
        assert n.op_kind_rewrite is None


def test_fine_grained_relational_preserved_when_base_kind_matches():
    """A confirmed RELATIONAL-* subtype is preserved; no spurious rewrite."""
    root = FAONode(op="logical_plan")
    rel = FAONode(op="input_relation", outputs=["products"])
    filter_node = FAONode(
        op="filter_cheap",
        inputs=["products"],
        outputs=["cheap"],
        op_kind="RELATIONAL-FILTER",
    )
    classify = FAONode(
        op="classify_sentiment",
        inputs=["cheap"],
        outputs=["labeled"],
        op_kind="SEMANTIC",
    )
    filter_node.children.append(rel)
    classify.children.append(filter_node)
    root.children.append(classify)

    response = _mock_response(
        [
            OpKindDecision(
                node_id="cheap",
                parser_op_kind="RELATIONAL-FILTER",
                chosen_op_kind="RELATIONAL",
                rationale="filter is relational",
                evidence="price column is numeric",
            ),
            OpKindDecision(
                node_id="labeled",
                parser_op_kind="SEMANTIC",
                chosen_op_kind="SEMANTIC",
                rationale="open-text sentiment",
                evidence="review column is free text",
            ),
        ]
    )
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    assert filter_node.op_kind == "RELATIONAL-FILTER"
    assert filter_node.op_kind_rewrite is None
    assert classify.op_kind == "SEMANTIC"
    assert classify.op_kind_rewrite is None


# ---------------------------------------------------------------------------
# Tests: consumer-demand stamping records the target output relation
# ---------------------------------------------------------------------------


def test_demand_stamping_tags_output_relation():
    """Each stamped consumer demand records the producer output it reads."""
    root, classify_text, classify_image, join = _two_classify_join_plan()
    response = AllNodesDemandResponse(
        final_output_demand=FinalOutputDemand(
            required_columns=[
                DemandedColumn(name="answer", dtype="VARCHAR", reason="result")
            ],
        ),
        node_demands=[
            NodeDemand(
                node_id="joined",
                input_demands=[
                    InputDemand(
                        input_relation="text_labeled",
                        required_columns=[
                            DemandedColumn(
                                name="text_label", dtype="VARCHAR", reason="join"
                            )
                        ],
                    ),
                    InputDemand(
                        input_relation="image_labeled",
                        required_columns=[
                            DemandedColumn(
                                name="image_label", dtype="VARCHAR", reason="join"
                            )
                        ],
                    ),
                ],
            ),
        ],
        op_kind_decisions=[],
    )
    with _patch_invoke(response):
        _run_one_shot(_make_generator(), root)

    # Producers carry a demand tagged with the exact output relation consumed.
    (text_demand,) = classify_text.consumer_demands
    assert text_demand["consumer"] == "join_formality"
    assert text_demand["output_relation"] == "text_labeled"

    (image_demand,) = classify_image.consumer_demands
    assert image_demand["output_relation"] == "image_labeled"

    # The final-output demand on the root child is tagged with its output too.
    final_demand = next(
        d for d in join.consumer_demands if d.get("is_final_output")
    )
    assert final_demand["output_relation"] == "joined"


# ---------------------------------------------------------------------------
# Dispatch: disabled / one-shot / top-down by plan size
# ---------------------------------------------------------------------------


def _run_dispatch(gen: PlanGenerator, root: FAONode) -> tuple[int, int]:
    one_shot = AsyncMock()
    top_down = AsyncMock()
    with patch.object(gen, "_apropagate_demands_one_shot", one_shot), patch.object(
        gen, "_apropagate_demands_top_down", top_down
    ), patch.object(PlanGenerator, "_build_schema_descriptions", return_value={}):
        asyncio.run(gen._apropagate_demands(root, rc=None, nl_query="q", config=None))
    return one_shot.await_count, top_down.await_count


def test_dispatch_one_shot_for_small_plans():
    root, *_ = _two_classify_join_plan()  # 3 actions
    gen = PlanGenerator(lp_llm=object(), demand_propagation_one_shot_max_actions=15)
    assert _run_dispatch(gen, root) == (1, 0)


def test_dispatch_top_down_when_plan_exceeds_one_shot_limit():
    root, *_ = _two_classify_join_plan()  # 3 actions
    gen = PlanGenerator(lp_llm=object(), demand_propagation_one_shot_max_actions=2)
    assert _run_dispatch(gen, root) == (0, 1)


def test_dispatch_disabled_skips_both():
    root, classify_text, classify_image, join = _two_classify_join_plan()
    gen = PlanGenerator(lp_llm=object(), demand_propagation=False)
    assert _run_dispatch(gen, root) == (0, 0)
    for n in (classify_text, classify_image, join):
        assert n.consumer_demands == [] and n.op_kind_rewrite is None
