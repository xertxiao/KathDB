"""Demand propagation: annotate the operator DAG with the columns each consumer needs
(and, in the one-shot variant, decide SEMANTIC vs RELATIONAL per node).

Mixed into :class:`~kathdb.plan_gen.plan_generator.PlanGenerator`, which sets
``lp_llm``, ``max_retries``, ``demand_propagation`` and
``demand_propagation_one_shot_max_actions``.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig

from ..common.context import DBContext
from ..common.logger import get_logger
from ..common.utils import (
    abatch_structured_with_retry,
    ainvoke_structured_with_retry,
)
from .plan_node import FAONode
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

logger = get_logger(__name__)

__all__ = ["DemandPropagation"]


class DemandPropagation:
    """Demand-propagation pass over an :class:`FAONode` DAG (mixin)."""

    lp_llm: BaseChatModel
    max_retries: int
    demand_propagation: bool
    demand_propagation_one_shot_max_actions: int

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
