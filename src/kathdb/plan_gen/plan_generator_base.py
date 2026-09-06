"""Base plan generator: parser actions -> operator DAG, plus the demand-propagation pass.

The base graph is ``build_fao_dag -> END``; subclasses extend :meth:`compile`
(see :class:`kathdb.plan_gen.plan_generator.PlanGenerator`).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import END, START, StateGraph

from ..common.context import DBContext
from ..common.logger import get_logger
from ..common.utils import (
    abatch_structured_with_retry,
    ainvoke_structured_with_retry,
)
from .plan_node import FAONode, build_fao_dag
from .prompts import (
    format_lp_all_node_demand_prompt,
    format_lp_demand_propagation_prompt,
    format_lp_query_demand_prompt,
)
from .response_schemas import (
    AllNodesDemandResponse,
    DemandPropagationResponse,
    QueryDemandResponse,
)
from .state_schemas import PlanGenInState, PlanGenOutState, PlanGenState

logger = get_logger(__name__)

__all__ = ["PlanGeneratorBase"]


class PlanGeneratorBase:
    """Structural base: builds the operator DAG; provides demand propagation."""

    def __init__(
        self,
        *,
        lp_llm: BaseChatModel,
        max_retries: int = 3,
        demand_propagation: bool = True,
        demand_propagation_one_shot_max_actions: int = 15,
    ) -> None:
        self.lp_llm = lp_llm
        self.max_retries = max_retries
        # Plans with at most ``demand_propagation_one_shot_max_actions`` actions are
        # annotated in one LLM call; larger plans go top-down, one BFS level per round.
        self.demand_propagation = demand_propagation
        self.demand_propagation_one_shot_max_actions = (
            demand_propagation_one_shot_max_actions
        )
        self.state_graph: Any | None = None

    # ------------------------------------------------------------------
    # Compile / run
    # ------------------------------------------------------------------

    def compile(self) -> None:
        graph = StateGraph(
            PlanGenState, input_schema=PlanGenInState, output_schema=PlanGenOutState
        )
        graph.add_node("build_fao_dag", self._build_plan_node)
        graph.add_edge(START, "build_fao_dag")
        graph.add_edge("build_fao_dag", END)
        self.state_graph = graph.compile()
        logger.info("Plan-generation state graph compiled.")

    def run(
        self, q_in: PlanGenInState, *, config: RunnableConfig | None = None
    ) -> PlanGenOutState:
        return asyncio.run(self.arun(q_in, config=config))

    async def arun(
        self, q_in: PlanGenInState, *, config: RunnableConfig | None = None
    ) -> PlanGenOutState:
        if self.state_graph is None:
            raise RuntimeError("Plan-generation state graph not compiled.")
        cfg: RunnableConfig = config or {
            "configurable": {"thread_id": 1},
            "recursion_limit": 50,
        }
        out = await self.state_graph.ainvoke(q_in, cfg)
        return PlanGenOutState(**out)

    def visualize(self) -> None:
        if self.state_graph is None:
            raise RuntimeError("Plan-generation state graph not compiled.")
        try:
            from IPython.display import Image, display

            display(Image(self.state_graph.get_graph().draw_mermaid_png()))
        except Exception as exc:  # pragma: no cover - debug helper
            logger.warning(
                "Failed to render mermaid graph (%s); printing ASCII fallback.",
                exc,
            )
            print(self.state_graph.get_graph().draw_ascii())

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _build_plan_node(self, state: PlanGenState, config=None) -> dict[str, Any]:
        """Build the operator DAG from the parser actions; no annotation here."""
        actions = state["actions"]
        rc: DBContext = state["relation_context"]
        schema_names = rc.list_tables()
        plan_entries = [self._make_entry(action) for action in actions]
        return {"logical_plan": build_fao_dag(plan_entries, schema_names)}

    # ------------------------------------------------------------------
    # Demand propagation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schema_descriptions(rc: DBContext) -> dict[str, str]:
        """Schema descriptions for all tables; multimodal views as one-line summaries."""
        schema_descriptions: dict[str, str] = {}
        for tname in rc.list_tables():
            if rc.is_view(tname):
                vs = rc.get_view_source(tname)
                cols = rc.get_columns(tname)
                schema_descriptions[tname] = (
                    f"View `{tname}`: from {vs.source_table}.{vs.source_column} "
                    f"({vs.modality.value}), columns: {', '.join(cols)}"
                )
            else:
                desc = rc.describe_table(tname)
                if desc is not None:
                    schema_descriptions[tname] = desc
        return schema_descriptions

    async def _apropagate_demands(
        self,
        root: FAONode,
        rc: DBContext,
        *,
        nl_query: str | None,
        config: RunnableConfig | None = None,
    ) -> None:
        """Annotate the DAG with consumer demands (and, one-shot, op-kind rewrites).

        Dispatch: disabled -> no-op; up to
        ``demand_propagation_one_shot_max_actions`` actions -> one LLM call over the
        whole DAG; more -> top-down, one BFS level per LLM round.
        """
        if not self.demand_propagation:
            logger.info("Demand propagation disabled; skipping the annotate pass.")
            return
        if not nl_query:
            logger.info("No NL query provided; skipping demand propagation.")
            return
        root_children = [c for c in root.children if c.op != "input_relation"]
        if not root_children:
            return
        schema_descriptions = self._build_schema_descriptions(rc)
        n_actions = sum(
            1
            for n in root.iter_preorder()
            if n is not root and n.op not in {"input_relation", "logical_plan"}
        )
        if n_actions <= self.demand_propagation_one_shot_max_actions:
            await self._apropagate_demands_one_shot(
                root, root_children, schema_descriptions, nl_query, config
            )
        else:
            await self._apropagate_demands_top_down(
                root, root_children, schema_descriptions, nl_query, config
            )

    @staticmethod
    def _node_needs_propagation(node: FAONode) -> bool:
        """Filter for ``_apropagate_demands``: node has demand to propagate."""
        if node.op == "input_relation":
            return False
        if not node.consumer_demands or not node.inputs:
            return False
        return any(c.op != "input_relation" for c in node.children)

    async def _apropagate_demands_top_down(
        self,
        root: FAONode,
        root_children: list[FAONode],
        schema_descriptions: dict[str, str],
        nl_query: str,
        config: RunnableConfig | None,
    ) -> None:
        """Top-down demand propagation: a root call for the query's demands, then one
        batched LLM call per BFS frontier. Sees only local context, so it never
        rewrites ``op_kind``; only the one-shot path does.
        """
        logger.info("Demand propagation: top-down (plan too large for one shot).")
        output_to_node: dict[str, FAONode] = {}
        for node in root.iter_preorder():
            for out in node.outputs:
                output_to_node[out] = node

        final_output_relation = (
            root_children[0].outputs[0] if root_children[0].outputs else "unknown"
        )

        prompt = format_lp_query_demand_prompt(
            nl_query=nl_query,
            output_relation=final_output_relation,
            schema_descriptions=schema_descriptions,
        )
        query_demand = await ainvoke_structured_with_retry(
            prompt,
            llm=self.lp_llm,
            schema=QueryDemandResponse,
            max_retries=self.max_retries,
            config=config,
        )

        demand_entry = {
            "consumer": "user_query",
            "required_columns": [
                {"name": col.name, "dtype": col.dtype, "reason": col.reason}
                for col in query_demand.required_columns
            ],
            "value_constraints": [
                {"column": vc.column, "constraint": vc.constraint}
                for vc in query_demand.value_constraints
            ],
            "is_final_output": True,
        }
        for child in root_children:
            # Tagged with the output relation it constrains (see _combine_consumer_demands).
            entry = dict(demand_entry)
            if child.outputs:
                entry["output_relation"] = child.outputs[0]
            child.consumer_demands.append(entry)

        logger.info(
            "Root demand propagation: %d required columns, %d value constraints → %s",
            len(query_demand.required_columns),
            len(query_demand.value_constraints),
            [c.op for c in root_children],
        )

        frontier: list[FAONode] = [
            c for c in root_children if self._node_needs_propagation(c)
        ]
        seen: set[int] = {id(n) for n in frontier}
        depth = 0
        while frontier:
            depth += 1
            prompts = [
                format_lp_demand_propagation_prompt(
                    op_name=node.op,
                    description=node.description or "",
                    input_relation_names=node.inputs,
                    consumer_demands=node.consumer_demands,
                    schema_descriptions=schema_descriptions,
                )
                for node in frontier
            ]
            logger.info(
                "Demand propagation: dispatching frontier d=%d size=%d",
                depth,
                len(frontier),
            )
            responses = await abatch_structured_with_retry(
                prompts,
                llm=self.lp_llm,
                schema=DemandPropagationResponse,
                max_retries=self.max_retries,
                config=config,
            )

            next_frontier: list[FAONode] = []
            for node, response in zip(frontier, responses):
                touched: list[FAONode] = []
                for input_demand in response.input_demands:
                    child_node = output_to_node.get(input_demand.input_relation)
                    if child_node is None or child_node.op == "input_relation":
                        continue
                    child_demand = {
                        "consumer": node.op,
                        "output_relation": input_demand.input_relation,
                        "required_columns": [
                            {"name": col.name, "dtype": col.dtype, "reason": col.reason}
                            for col in input_demand.required_columns
                        ],
                        "value_constraints": [
                            {"column": vc.column, "constraint": vc.constraint}
                            for vc in input_demand.value_constraints
                        ],
                    }
                    child_node.consumer_demands.append(child_demand)
                    touched.append(child_node)
                logger.info(
                    "Demand propagation for '%s': %d input demands",
                    node.op,
                    len(response.input_demands),
                )
                for child in touched:
                    if id(child) in seen:
                        continue
                    if self._node_needs_propagation(child):
                        next_frontier.append(child)
                        seen.add(id(child))
            frontier = next_frontier


    async def _apropagate_demands_one_shot(
        self,
        root: FAONode,
        root_children: list[FAONode],
        schema_descriptions: dict[str, str],
        nl_query: str,
        config: RunnableConfig | None,
    ) -> None:
        """One LLM call over the full DAG: stamps ``consumer_demands`` on producers and
        applies SEMANTIC/RELATIONAL decisions (audit trail in ``op_kind_rewrite``)."""
        output_to_node: dict[str, FAONode] = {}
        for node in root.iter_preorder():
            for out in node.outputs:
                output_to_node[out] = node

        # ``node_id`` = primary output relation (unique: build_fao_dag rejects duplicates).
        node_id_to_node: dict[str, FAONode] = {}
        nodes_payload: list[dict] = []
        for node in root.iter_preorder():
            if node is root or node.op == "input_relation":
                continue
            if not node.outputs:
                continue
            node_id = node.outputs[0]
            node_id_to_node[node_id] = node
            payload = {
                "node_id": node_id,
                "op": node.op,
                "parser_op_kind": (node.op_kind or "UNSET").upper(),
                "description": node.description or "",
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
            }
            if node.type == "GROUPED":
                payload["type"] = "GROUPED"
                if node.member_atoms:
                    payload["member_atoms"] = list(node.member_atoms)
            nodes_payload.append(payload)

        final_output_relation = (
            root_children[0].outputs[0] if root_children[0].outputs else "unknown"
        )

        prompt = format_lp_all_node_demand_prompt(
            nl_query=nl_query,
            final_output_relation=final_output_relation,
            schema_descriptions=schema_descriptions,
            nodes=nodes_payload,
        )
        response = await ainvoke_structured_with_retry(
            prompt,
            llm=self.lp_llm,
            schema=AllNodesDemandResponse,
            max_retries=self.max_retries,
            config=config,
        )

        final_demand = response.final_output_demand
        demand_entry = {
            "consumer": "user_query",
            "required_columns": [
                {"name": col.name, "dtype": col.dtype, "reason": col.reason}
                for col in final_demand.required_columns
            ],
            "value_constraints": [
                {"column": vc.column, "constraint": vc.constraint}
                for vc in final_demand.value_constraints
            ],
            "is_final_output": True,
        }
        for child in root_children:
            entry = dict(demand_entry)
            if child.outputs:
                entry["output_relation"] = child.outputs[0]
            child.consumer_demands.append(entry)

        logger.info(
            "Demand propagation: final-output demands "
            "(%d columns, %d constraints) stamped on %s; %d node-demand entries returned.",
            len(final_demand.required_columns),
            len(final_demand.value_constraints),
            [c.op for c in root_children],
            len(response.node_demands),
        )

        for node_demand in response.node_demands:
            consumer_node = node_id_to_node.get(node_demand.node_id)
            if consumer_node is None:
                logger.warning(
                    "Demand propagation: unknown node_id %r in response; skipping.",
                    node_demand.node_id,
                )
                continue
            for input_demand in node_demand.input_demands:
                producer = output_to_node.get(input_demand.input_relation)
                if producer is None:
                    logger.warning(
                        "Demand propagation: node %r refers to unknown input "
                        "relation %r; skipping.",
                        node_demand.node_id,
                        input_demand.input_relation,
                    )
                    continue
                if producer.op == "input_relation":
                    continue
                producer.consumer_demands.append(
                    {
                        "consumer": consumer_node.op,
                        "output_relation": input_demand.input_relation,
                        "required_columns": [
                            {"name": col.name, "dtype": col.dtype, "reason": col.reason}
                            for col in input_demand.required_columns
                        ],
                        "value_constraints": [
                            {"column": vc.column, "constraint": vc.constraint}
                            for vc in input_demand.value_constraints
                        ],
                    }
                )

        rewrites_recorded = 0
        for dec in response.op_kind_decisions:
            target = node_id_to_node.get(dec.node_id)
            if target is None:
                logger.warning(
                    "op_kind decision: unknown node_id %r; skipping.", dec.node_id
                )
                continue
            chosen = (dec.chosen_op_kind or "").strip().upper()
            if chosen not in {"SEMANTIC", "RELATIONAL"}:
                logger.warning(
                    "op_kind decision for %r: invalid chosen_op_kind=%r; skipping.",
                    dec.node_id,
                    dec.chosen_op_kind,
                )
                continue
            parser_kind = (target.op_kind or "UNSET").upper()
            if parser_kind.split("-")[0] == chosen:
                continue
            target.op_kind_rewrite = {
                "from": parser_kind,
                "to": chosen,
                "rationale": dec.rationale,
                "evidence": dec.evidence,
            }
            rewrites_recorded += 1
            logger.info(
                "op_kind rewrite on %s (%s): %s -> %s (%s)",
                target.op,
                dec.node_id,
                parser_kind,
                chosen,
                (dec.rationale or "")[:80],
            )
            target.op_kind = chosen

        if response.op_kind_decisions:
            logger.info(
                "Demand propagation: %d op_kind decisions applied, %d rewrote a "
                "node's op_kind.",
                len(response.op_kind_decisions),
                rewrites_recorded,
            )

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
    def save_lp(lp_node: FAONode, path: str | None = None) -> Path:
        """Save the logical plan to a JSON file."""
        out_path = Path(path or "logical_plan.json")
        out_path.write_text(json.dumps(lp_node.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved logical plan to %s", out_path)
        return out_path
