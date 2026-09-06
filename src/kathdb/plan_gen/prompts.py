"""Prompts for the plan-annotation (demand propagation) LLM calls.

One-shot (default, small plans): ``format_lp_all_node_demand_prompt`` — one call over
the whole DAG. Top-down (large plans): ``format_lp_query_demand_prompt`` for the
final output, then ``format_lp_demand_propagation_prompt`` per node, one BFS level
at a time.
"""

from __future__ import annotations

from textwrap import dedent


__all__ = [
    "format_lp_all_node_demand_prompt",
    "format_lp_demand_propagation_prompt",
    "format_lp_query_demand_prompt",
]


def _format_template(template: str, **variables: str) -> str:
    try:
        return template.format(**variables)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing template variable: {missing}") from None


LP_QUERY_DEMAND_PROMPT = dedent(
    """
## System
You are an expert query analyzer for KathDB, a multi-modal database.

## Instructions
Given a user's natural language query and the output relation that will produce the final result,
determine what columns and value constraints are needed in the final output.

1. Identify all columns the user expects in the result (explicit or implicit).
   For each column, specify:
   - name: a valid column identifier
   - dtype: a DuckDB type ('BIGINT', 'DOUBLE', 'VARCHAR', 'BOOLEAN', 'TIMESTAMP')
   - reason: why this column is needed to answer the query

2. Identify any value-level constraints the query implies (e.g., specific categories,
   numeric ranges, fixed enums) that downstream nodes should produce.

## Query
{nl_query}

## Available Schemas
{schema_descriptions}

## Output Relation
{output_relation}
"""
).strip()


LP_DEMAND_PROPAGATION_PROMPT = dedent(
    """
## System
You are an expert demand propagator for KathDB's Logical Planner.

## Instructions
You are given a node in the query plan and the demands placed on it by its consumers.
Determine what this node needs from each of its input relations to satisfy those demands.

Some input relations may be intermediate results produced by earlier plan nodes whose
exact schemas are not yet known. Declare your demands from each input relation so that
the downstream schema prediction step can ensure those columns and constraints are satisfied.

For each input relation, specify:
1. Required columns: columns that must exist in the input, with expected dtype and reason.
2. Value constraints: any value-level constraints on input columns (e.g., fixed enum values,
   numeric ranges) that would help this node produce output satisfying consumer demands.

{self_optimization_hint}

## Node
Name: {op_name}
Description: {description}
Input relation names: {input_relation_names}

## Consumer Demands
{consumer_demands}

{schema_descriptions_block}
"""
).strip()


def format_lp_query_demand_prompt(
    *,
    nl_query: str,
    output_relation: str,
    schema_descriptions: dict[str, str],
) -> str:
    """Top-down pass 1: demands the user query places on the final output."""
    schema_parts = []
    for name, desc in schema_descriptions.items():
        schema_parts.append(f"- {name}: {desc}")
    schema_text = "\n".join(schema_parts) if schema_parts else "None available."
    return _format_template(
        LP_QUERY_DEMAND_PROMPT,
        nl_query=nl_query,
        output_relation=output_relation or "unknown",
        schema_descriptions=schema_text,
    )


LP_ALL_NODE_DEMAND_PROMPT = dedent(
    """
## System
You are an expert demand analyzer for KathDB's Logical Planner.

## Instructions
You are given the FULL operator DAG of a query plan, the user's natural-language
query, and the schema descriptions of available relations. In a SINGLE response,
produce three things:

1. ``final_output_demand`` — the columns and value constraints that must appear
   in the LP's final output relation to fully answer the user's NL query.
   For each column specify:
   - name: a valid column identifier
   - dtype: one of 'BIGINT', 'DOUBLE', 'VARCHAR', 'BOOLEAN', 'TIMESTAMP'
   - reason: why this column is needed to answer the query

2. ``node_demands`` — for EVERY node listed in `## Nodes`, declare what that node
   needs from EACH of its input relations to satisfy demands from its
   downstream consumers (and ultimately the final output demand above).
   Identify the consumer node by its ``node_id`` (the node's primary output
   relation name, as shown in `## Nodes`). For each input relation give:
   - required_columns: name + dtype + reason
   - value_constraints: per-column value-level constraints (fixed enums,
     numeric ranges) that downstream code should rely on

Reason about the entire DAG simultaneously: each node should only demand from
an input what it truly needs to produce its outputs, given what its consumers
require. Some input relations are intermediate results produced by other plan
nodes whose exact schemas are not yet known; declaring demands clearly lets
the downstream code-generation step ensure those columns and constraints are
materialized.

3. ``op_kind_decisions`` — for EVERY node listed in `## Nodes`, classify its
   final ``op_kind`` as either "SEMANTIC" (requires ML/LLM/VLM inference at
   execution time) or "RELATIONAL" (pure pandas / SQL / DuckDB; no model call).
   Each node arrives tagged with a ``parser_op_kind`` (its current label, or
   "UNSET" for GROUPED nodes that have not yet been tagged). You may downgrade
   a SEMANTIC node to RELATIONAL, AND ONLY WHEN, the demands you just derived
   in step 2 imply the node's effective input/output values are constrained to
   a closed, finite domain on every relevant column.

   ### Theory
   If a node's input value spaces from every producer are constrained to a
   closed finite domain (a fixed enum, a small set of literal strings, an
   integer bin set), AND the node's computation reduces to deterministic
   comparison on those domains (equality join, equality filter, group-by on
   an enum key, IN-list test, sort on a bounded numeric range), then the node
   is realizable as RELATIONAL — no LLM call needed.

   ### Demonstration (closed-enum classify → join)
   Two upstream classifiers tag products from text descriptions and from
   product images respectively. Demand propagation pushes the closed enum
   {{"Formal", "Non-Formal"}} up to both classifiers, so both contractually
   emit values in that set. The downstream join that compares the two tags
   would naïvely require O(L*R) LLM calls for fuzzy reconciliation; because
   both inputs share an identical finite enum, it collapses to a deterministic
   equality merge — RELATIONAL. Note what made this legal: the enum values are
   COMPUTED by nodes that stay SEMANTIC (the two classifiers).

   ### Counter-demonstration (INVALID downgrade — orphaned model work)
   A pairing node needs a `sentiment` column constrained to
   {{'positive','negative'}}, and its only producer is a RELATIONAL row
   filter over a table that has no sentiment column. Writing the closed-enum
   demand onto that filter does NOT make the values exist — a relational
   node cannot classify text, so its generated code can only fabricate an
   empty placeholder column, and every downstream comparison silently
   returns zero rows. Downgrading the pairing node here orphans the
   classification: no node owns the model call anymore. The pairing node
   must stay SEMANTIC (it, or a function it calls, computes sentiment from
   the raw text itself).

   ### Per-node response shape
   - node_id (use the value shown in `## Nodes`)
   - parser_op_kind: "SEMANTIC", "RELATIONAL", or "UNSET"
   - chosen_op_kind: "SEMANTIC" or "RELATIONAL"
   - rationale: 1-2 sentences explaining the decision in this DAG context
   - evidence: which producer outputs / value_constraints / demands you relied
     on (cite by node_id or column name)

   ### Guardrails
   - Objective: minimize SEMANTIC operators without sacrificing correctness.
   - If even ONE producer of an input emits open-ended text (no closed
     value_constraint on that column), the consuming node remains SEMANTIC.
   - Anchor every downgrade claim on closed-domain evidence drawn from the
     value_constraints you derived in step 2. Do NOT reason from row counts,
     selectivity, or other cardinality language.
   - A value_constraint is only evidence if its producer can actually COMPUTE
     those values: the column exists in the producer's input, or the producer
     is (or remains) SEMANTIC. Every model-derived column must be owned by a
     SEMANTIC node somewhere upstream of its consumers — never justify a
     downgrade with a demand you yourself materialize onto a RELATIONAL
     producer (see the counter-demonstration).
   - Cascades are allowed: when you downgrade a node in this response, treat
     its outputs as RELATIONAL when evaluating downstream nodes within the
     SAME response.
   - Never upgrade RELATIONAL to SEMANTIC.

## Query
{nl_query}

## Final Output Relation
{final_output_relation}

## Available Schemas
{schema_descriptions}

## Nodes
{nodes_block}
"""
).strip()


def format_lp_demand_propagation_prompt(
    *,
    op_name: str,
    description: str,
    input_relation_names: list[str],
    consumer_demands: list[dict],
    schema_descriptions: dict[str, str],
) -> str:
    """Top-down pass 2: what one node needs from its inputs, given its consumers' demands."""
    demand_parts: list[str] = []
    for cd in consumer_demands:
        consumer = cd.get("consumer", "unknown")
        is_final = cd.get("is_final_output", False)
        header = f"From '{consumer}'"
        if is_final:
            header += " (FINAL OUTPUT)"
        demand_parts.append(header + ":")
        for col in cd.get("required_columns", []):
            demand_parts.append(
                f"  - Column '{col.get('name', '?')}' ({col.get('dtype', '?')}): "
                f"{col.get('reason', '')}"
            )
        for vc in cd.get("value_constraints", []):
            demand_parts.append(
                f"  - Constraint on '{vc.get('column', '?')}': {vc.get('constraint', '')}"
            )
    demands_text = "\n".join(demand_parts) if demand_parts else "None."

    schema_parts: list[str] = []
    for name in input_relation_names:
        desc = schema_descriptions.get(name)
        if desc:
            schema_parts.append(f"- {name}: {desc}")
    schema_block = (
        "## Available Schemas\n" + "\n".join(schema_parts) if schema_parts else ""
    )

    return _format_template(
        LP_DEMAND_PROPAGATION_PROMPT,
        op_name=op_name,
        description=description or "None provided.",
        input_relation_names=(
            ", ".join(input_relation_names) if input_relation_names else "None"
        ),
        consumer_demands=demands_text,
        schema_descriptions_block=schema_block,
        self_optimization_hint="",
    )


def format_lp_all_node_demand_prompt(
    *,
    nl_query: str,
    final_output_relation: str,
    schema_descriptions: dict[str, str],
    nodes: list[dict],
) -> str:
    """Format the one-shot all-node demand prompt.

    ``nodes`` is a list of dicts with keys ``node_id``, ``op``,
    ``description``, ``inputs``, ``outputs``, ``parser_op_kind`` (and
    optionally ``type`` + ``member_atoms`` for GROUPED nodes) covering every
    non-input-relation, non-root node in the LP DAG.
    """
    schema_parts: list[str] = []
    for name, desc in schema_descriptions.items():
        schema_parts.append(f"- {name}: {desc}")
    schema_text = "\n".join(schema_parts) if schema_parts else "None available."

    node_parts: list[str] = []
    for node in nodes:
        inputs = node.get("inputs") or []
        outputs = node.get("outputs") or []
        node_parts.append(f"- node_id: {node.get('node_id', '?')}")
        node_parts.append(f"  op: {node.get('op', '?')}")
        node_parts.append(f"  parser_op_kind: {node.get('parser_op_kind', 'UNSET')}")
        if node.get("type") == "GROUPED":
            node_parts.append("  type: GROUPED")
            members = node.get("member_atoms") or []
            if members:
                node_parts.append(
                    f"  member_atoms: {', '.join(str(m) for m in members)}"
                )
        node_parts.append(
            f"  description: {node.get('description') or 'None provided.'}"
        )
        node_parts.append(f"  inputs: {', '.join(inputs) if inputs else 'None'}")
        node_parts.append(f"  outputs: {', '.join(outputs) if outputs else 'None'}")
    nodes_block = "\n".join(node_parts) if node_parts else "None."

    return _format_template(
        LP_ALL_NODE_DEMAND_PROMPT,
        nl_query=nl_query,
        final_output_relation=final_output_relation or "unknown",
        schema_descriptions=schema_text,
        nodes_block=nodes_block,
    )
