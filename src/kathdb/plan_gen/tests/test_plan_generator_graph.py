"""Pipeline-wiring tests for the PlanGenerator LangGraph.

``annotate`` runs before ``group_actions`` so op-kind rewrites affect the
whole-plan partition search.
"""

from __future__ import annotations

from kathdb.plan_gen.optimizer import GroupingConfig
from kathdb.plan_gen.plan_generator import PlanGenerator


def _edge_set(pg: PlanGenerator) -> set[tuple[str, str]]:
    graph = pg.state_graph.get_graph()
    return {(e.source, e.target) for e in graph.edges}


def _make_pg(*, grouping_enabled: bool) -> PlanGenerator:
    pg = PlanGenerator(
        lp_llm=object(), grouping_cfg=GroupingConfig(enabled=grouping_enabled)
    )
    pg.compile()
    return pg


def test_grouping_runs_annotate_before_grouping():
    pg = _make_pg(grouping_enabled=True)
    edges = _edge_set(pg)

    assert ("build_fao_dag", "thread_fns") in edges
    assert ("thread_fns", "annotate") in edges
    assert ("annotate", "group_actions") in edges
    assert ("group_actions", "__end__") in edges
    assert ("group_actions", "annotate") not in edges


def test_graph_with_grouping_disabled_skips_group_actions():
    pg = _make_pg(grouping_enabled=False)
    nodes = {n for n in pg.state_graph.get_graph().nodes}
    edges = _edge_set(pg)

    assert "group_actions" not in nodes
    assert ("build_fao_dag", "thread_fns") in edges
    assert ("thread_fns", "annotate") in edges
    assert ("annotate", "__end__") in edges

