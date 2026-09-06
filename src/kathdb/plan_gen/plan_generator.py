"""Plan generation: parser actions -> annotated operator DAG -> (optionally) grouped plan.

Stages, in order: ``build_fao_dag`` (actions -> DAG), ``thread_fns`` (copy the
parser's ``selected_functions`` onto the nodes), ``annotate`` (demand propagation,
may re-tag ``op_kind`` SEMANTIC <-> RELATIONAL) and ``group_actions`` (the grouping
optimizer, :mod:`kathdb.plan_gen.optimizer`, only when enabled).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig

from ..common.context import DBContext
from ..common.function_manager import FunctionManager
from ..common.logger import get_logger
from .demand_propagation import DemandPropagation
from .optimizer import GroupingConfig, run_optimizer
from .plan_node import FAONode, build_fao_dag
from .state_schemas import PlanGenInState, PlanGenOutState

logger = get_logger(__name__)

__all__ = ["PlanGenerator"]


def _with_callback(base: RunnableConfig | None, handler: Any) -> RunnableConfig:
    """Return a RunnableConfig copy with ``handler`` appended to ``callbacks``."""
    base = dict(base or {})
    existing = list(base.get("callbacks") or [])
    base["callbacks"] = [*existing, handler]
    return base  # type: ignore[return-value]


class PlanGenerator(DemandPropagation):
    """DAG build + function threading + demand propagation + grouping."""

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
        self.lp_llm = lp_llm
        self.max_retries = max_retries
        # Plans with at most ``demand_propagation_one_shot_max_actions`` actions are
        # annotated in one LLM call; larger plans go top-down, one BFS level per round.
        self.demand_propagation = demand_propagation
        self.demand_propagation_one_shot_max_actions = (
            demand_propagation_one_shot_max_actions
        )
        self._grouping_cfg = grouping_cfg or GroupingConfig(enabled=False)
        self._fn_manager = fn_manager or FunctionManager()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self, q_in: PlanGenInState, *, config: RunnableConfig | None = None
    ) -> PlanGenOutState:
        return asyncio.run(self.arun(q_in, config=config))

    async def arun(
        self, q_in: PlanGenInState, *, config: RunnableConfig | None = None
    ) -> PlanGenOutState:
        """Run the stages in order; ``usage_by_model`` holds per-stage token usage
        (``annotate``) plus ``_total`` for the whole run."""
        top_cb = UsageMetadataCallbackHandler()
        cfg = _with_callback(config, top_cb)
        state: dict[str, Any] = dict(q_in)
        usage: dict[str, dict[str, Any]] = {}

        state.update(self._build_plan_node(state))
        state.update(self._thread_fns_node(state))
        update = await self._annotate_node(state, cfg)
        usage.update(update.pop("usage_by_model", {}))
        state.update(update)
        if self._grouping_cfg.enabled:
            update = await self._group_actions_node(state, cfg)
            usage.update(update.pop("usage_by_model", {}))
            state.update(update)
        usage["_total"] = dict(top_cb.usage_metadata)

        out = PlanGenOutState(
            q_in=state["q_in"],
            actions=state["actions"],
            relation_context=state["relation_context"],
            input_rel_names=state["input_rel_names"],
            input_rel=state["input_rel"],
            logical_plan=state["logical_plan"],
            usage_by_model=usage,
        )
        if "grouping_trace" in state:
            out["grouping_trace"] = state["grouping_trace"]
        return out

    # ------------------------------------------------------------------
    # Stages (each returns an update to the working state)
    # ------------------------------------------------------------------

    def _build_plan_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build the operator DAG from the parser actions; no annotation here."""
        actions = state["actions"]
        rc: DBContext = state["relation_context"]
        schema_names = rc.list_tables()
        plan_entries = [self._make_entry(action) for action in actions]
        return {"logical_plan": build_fao_dag(plan_entries, schema_names)}

    def _thread_fns_node(
        self, state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Copy parser-supplied ``selected_functions`` onto the plan nodes (no LLM call)."""
        root: FAONode = state["logical_plan"]
        actions = state.get("actions") or []
        valid_names = set(self._fn_manager.discover_functions().keys())
        self._thread_selected_functions(root, actions, valid_names)
        return {"logical_plan": root}

    async def _annotate_node(
        self, state: dict[str, Any], config: RunnableConfig | None = None
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
        self, state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Run the grouping optimizer. The selector factory is injected by
        :class:`~kathdb.KathDB`; without it the plan is returned unchanged."""
        root: FAONode = state["logical_plan"]
        selector_factory = state.get("grouping_selector_factory")
        if selector_factory is None:
            logger.warning("[optimizer] no selector factory injected; plan unchanged")
            return {"logical_plan": root}

        new_root, trace = run_optimizer(
            root=root, cfg=self._grouping_cfg, selector=selector_factory(root)
        )
        return {"logical_plan": new_root, "grouping_trace": trace.to_dict()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_entry(action: Any) -> dict[str, Any]:
        """Convert one parser :class:`~kathdb.parser.action.Action` into a plan entry."""
        from ..parser.action import Action

        if isinstance(action, Action):
            output_name = action.output.strip() if action.output else ""
            description = action.action.strip() if action.action else ""
            payload: dict[str, Any] = {
                # Order-preserving dedup keeps child-attachment order deterministic.
                "input": list(dict.fromkeys(action.inputs)),
                "output": [output_name] if output_name else [],
            }
            if description:
                payload["description"] = description
            if action.name and action.name.strip():
                payload["name"] = action.name.strip()
            if action.output_type:
                payload["output_type"] = str(action.output_type)
            if action.op_kind:
                payload["op_kind"] = str(action.op_kind)
            return payload
        # Plain-dict actions (tests / programmatic plans).
        output = action.get("out")
        output_name = output.strip() if isinstance(output, str) else ""
        text = action.get("action")
        description = text.strip() if isinstance(text, str) else ""
        payload = {
            "input": list(dict.fromkeys(action.get("in", []))),
            "output": [output_name] if output_name else [],
        }
        if description:
            payload["description"] = description
        name = action.get("name")
        if isinstance(name, str) and name.strip():
            payload["name"] = name.strip()
        if "output_type" in action:
            payload["output_type"] = str(action["output_type"])
        if "op_kind" in action and action["op_kind"]:
            payload["op_kind"] = str(action["op_kind"])
        return payload

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

    @staticmethod
    def save_lp(lp_node: FAONode, path: str | None = None) -> Path:
        """Save the logical plan to a JSON file."""
        out_path = Path(path or "logical_plan.json")
        out_path.write_text(json.dumps(lp_node.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved logical plan to %s", out_path)
        return out_path
