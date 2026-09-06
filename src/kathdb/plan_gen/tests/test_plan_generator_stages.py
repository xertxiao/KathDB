"""Stage order of ``PlanGenerator.arun``: ``annotate`` runs before ``group_actions``
(so op-kind rewrites affect the whole-plan partition search), and ``group_actions``
runs only when grouping is enabled."""

from __future__ import annotations

import asyncio

from kathdb.plan_gen.optimizer import GroupingConfig
from kathdb.plan_gen.plan_generator import PlanGenerator


def _run(*, grouping_enabled: bool):
    pg = PlanGenerator(
        lp_llm=object(), grouping_cfg=GroupingConfig(enabled=grouping_enabled)
    )
    calls: list[str] = []

    def build(state):
        calls.append("build_fao_dag")
        return {"logical_plan": "plan"}

    def thread(state, config=None):
        calls.append("thread_fns")
        return {}

    async def annotate(state, config=None):
        calls.append("annotate")
        return {"usage_by_model": {"annotate": {"m": 1}}}

    async def group(state, config=None):
        calls.append("group_actions")
        return {"logical_plan": "grouped", "grouping_trace": {"k": 1}}

    pg._build_plan_node = build  # type: ignore[assignment]
    pg._thread_fns_node = thread  # type: ignore[assignment]
    pg._annotate_node = annotate  # type: ignore[assignment]
    pg._group_actions_node = group  # type: ignore[assignment]
    out = asyncio.run(
        pg.arun(
            {
                "q_in": "q",
                "actions": [],
                "relation_context": None,
                "input_rel_names": [],
                "input_rel": [],
            }
        )
    )
    return calls, out


def test_grouping_runs_annotate_before_grouping():
    calls, out = _run(grouping_enabled=True)
    assert calls == ["build_fao_dag", "thread_fns", "annotate", "group_actions"]
    assert out["logical_plan"] == "grouped"
    assert out["grouping_trace"] == {"k": 1}
    assert set(out["usage_by_model"]) == {"annotate", "_total"}


def test_grouping_disabled_skips_group_actions():
    calls, out = _run(grouping_enabled=False)
    assert calls == ["build_fao_dag", "thread_fns", "annotate"]
    assert out["logical_plan"] == "plan"
    assert "grouping_trace" not in out
