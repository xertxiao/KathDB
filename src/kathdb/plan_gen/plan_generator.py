"""Plan generation: parser actions -> annotated operator DAG -> (optionally) grouped plan.

Graph: ``build_fao_dag -> thread_fns -> annotate -> [group_actions] -> END``.
``thread_fns`` copies the parser's ``selected_functions`` onto the nodes; ``annotate``
is demand propagation (may re-tag ``op_kind`` SEMANTIC <-> RELATIONAL); ``group_actions``
is the grouping optimizer (:mod:`kathdb.plan_gen.optimizer`), present only when enabled.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import END, START, StateGraph

from ..common.context import DBContext
from ..common.function_manager import FunctionManager
from ..common.logger import get_logger
from .optimizer import GroupingConfig, run_optimizer
from .plan_generator_base import PlanGeneratorBase
from .plan_node import FAONode
from .state_schemas import PlanGenInState, PlanGenOutState, PlanGenState

logger = get_logger(__name__)

__all__ = ["PlanGenerator"]


def _with_callback(base: RunnableConfig | None, handler: Any) -> RunnableConfig:
    """Return a RunnableConfig copy with ``handler`` appended to ``callbacks``."""
    base = dict(base or {})
    existing = list(base.get("callbacks") or [])
    base["callbacks"] = [*existing, handler]
    return base  # type: ignore[return-value]


class PlanGenerator(PlanGeneratorBase):
    """Base DAG build + function threading + demand propagation + grouping."""

    def __init__(
        self,
        *,
        lp_llm: BaseChatModel,
        max_retries: int = 3,
        demand_propagation: bool = True,
        demand_propagation_one_shot_max_actions: int = 15,
        grouping_cfg: GroupingConfig | None = None,
        fn_manager: FunctionManager | None = None,
    ) -> None:
        super().__init__(
            lp_llm=lp_llm,
            max_retries=max_retries,
            demand_propagation=demand_propagation,
            demand_propagation_one_shot_max_actions=demand_propagation_one_shot_max_actions,
        )
        self._grouping_cfg = grouping_cfg or GroupingConfig(enabled=False)
        self._fn_manager = fn_manager or FunctionManager()

    # ------------------------------------------------------------------
    # Compile / run
    # ------------------------------------------------------------------

    def compile(self) -> None:
        graph = StateGraph(
            PlanGenState,
            input_schema=PlanGenInState,
            output_schema=PlanGenOutState,
        )
        graph.add_node("build_fao_dag", self._build_plan_node)
        graph.add_node("thread_fns", self._thread_fns_node)
        graph.add_node("annotate", self._annotate_node)
        graph.add_edge(START, "build_fao_dag")
        graph.add_edge("build_fao_dag", "thread_fns")
        graph.add_edge("thread_fns", "annotate")
        if self._grouping_cfg.enabled:
            graph.add_node("group_actions", self._group_actions_node)
            graph.add_edge("annotate", "group_actions")
            graph.add_edge("group_actions", END)
        else:
            graph.add_edge("annotate", END)

        self.state_graph = graph.compile()
        logger.info(
            "PlanGenerator state graph compiled (grouping=%s).",
            "on" if self._grouping_cfg.enabled else "off",
        )

    async def arun(
        self, q_in: PlanGenInState, *, config: RunnableConfig | None = None
    ) -> PlanGenOutState:
        if self.state_graph is None:
            raise RuntimeError("PlanGenerator state graph not compiled.")
        top_cb = UsageMetadataCallbackHandler()
        base_cfg: RunnableConfig = config or {
            "configurable": {"thread_id": 1},
            "recursion_limit": 50,
        }
        cfg = _with_callback(base_cfg, top_cb)
        out = await self.state_graph.ainvoke(q_in, cfg)
        out_dict = dict(out)
        merged = dict(out_dict.get("usage_by_model") or {})
        merged["_total"] = dict(top_cb.usage_metadata)
        out_dict["usage_by_model"] = merged
        return PlanGenOutState(**out_dict)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _thread_fns_node(
        self, state: PlanGenState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Copy parser-supplied ``selected_functions`` onto the plan nodes (no LLM call)."""
        root: FAONode = state["logical_plan"]
        actions = state.get("actions") or []
        valid_names = set(self._fn_manager.discover_functions().keys())
        self._thread_selected_functions(root, actions, valid_names)
        return {"logical_plan": root}

    async def _annotate_node(
        self, state: PlanGenState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Demand propagation (+ op-kind rewriting)."""
        root: FAONode = state["logical_plan"]
        rc: DBContext = state["relation_context"]
        nl_query = state.get("q_in")

        stage_cb = UsageMetadataCallbackHandler()
        stage_cfg = _with_callback(config, stage_cb)
        await self._apropagate_demands(root, rc, nl_query=nl_query, config=stage_cfg)
        return {"usage_by_model": {"annotate": dict(stage_cb.usage_metadata)}}

    async def _group_actions_node(
        self, state: PlanGenState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Run the grouping optimizer. The selector factory is injected by
        :class:`~kathdb.KathDB`; without it the plan is returned unchanged."""
        root: FAONode = state["logical_plan"]
        selector_factory = state.get("grouping_selector_factory")
        if selector_factory is None:
            logger.warning("[optimizer] no selector factory injected; plan unchanged")
            return {"logical_plan": root}

        # Context-propagated callbacks attribute the optimizer's LLM usage to this handler.
        stage_cb = UsageMetadataCallbackHandler()
        _with_callback(config, stage_cb)
        new_root, trace = run_optimizer(
            root=root, cfg=self._grouping_cfg, selector=selector_factory(root)
        )
        return {
            "logical_plan": new_root,
            "grouping_trace": trace.to_dict(),
            "usage_by_model": {"group_actions": dict(stage_cb.usage_metadata)},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _thread_selected_functions(
        root: FAONode,
        actions: list[Any],
        valid_names: set[str],
    ) -> None:
        """Copy parser-supplied ``selected_functions`` onto matching plan nodes."""
        from ..parser.action import Action

        by_op: dict[str, list[str]] = {}
        for act in actions:
            if isinstance(act, Action):
                name = act.name
                sel = act.selected_functions or []
            elif isinstance(act, dict):
                name = act.get("name")
                sel = act.get("selected_functions") or []
            else:
                continue
            if not name:
                continue
            by_op[str(name)] = [s for s in sel if s in valid_names]

        annotated = 0
        for node in root.iter_preorder():
            if node is root or node.op in {"input_relation", "logical_plan"}:
                continue
            sel = by_op.get(node.op)
            if sel:
                node.selected_functions = list(sel)
                annotated += 1
        logger.info(
            "Threaded parser-supplied selected_functions onto %d node(s).",
            annotated,
        )
