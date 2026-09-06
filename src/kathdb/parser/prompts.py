"""Prompt templates dedicated to the natural-language parser workflow."""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence


__all__ = [
    "format_clarification_prompt",
    "format_revision_prompt",
    "format_action_query_sketch_prompt",
    "format_action_query_sketch_with_functions_prompt",
    "format_pick_functions_prompt",
    "format_refine_query_prompt",
]

_CANONICAL_REL_OPS = (
    "filter, join, group_by_aggregate, project, sort, limit, "
    "rename, compute, distinct, union, intersect, difference, window"
)

_CANONICAL_REL_OP_KINDS = (
    "RELATIONAL-FILTER, RELATIONAL-JOIN, RELATIONAL-GROUP_BY_AGGREGATE, "
    "RELATIONAL-PROJECT, RELATIONAL-SORT, RELATIONAL-LIMIT, "
    "RELATIONAL-RENAME, RELATIONAL-COMPUTE, RELATIONAL-DISTINCT, "
    "RELATIONAL-UNION, RELATIONAL-INTERSECT, RELATIONAL-DIFFERENCE, "
    "RELATIONAL-WINDOW"
)

_NAME_INSTRUCTION = (
    '- "name": A query-specific snake_case label describing what this '
    "action does in the context of this query (e.g. filter_cheap_products, "
    "join_with_images, classify_sentiment). "
    "If the query needs two actions of the same kind, use distinct "
    "descriptive names — the system auto-suffixes duplicates (_2, _3, ...) "
    "as a fallback."
)

_REVISION_NAME_INSTRUCTION = (
    '- "name": A query-specific snake_case label describing what this '
    "action does (e.g. filter_cheap_products, classify_sentiment). "
    'Do not use generic names like "step_1".'
)

_ACTION_FIELD_INSTRUCTION = (
    '- "action": Query-specific Verb + Subject; include entities/filters when known. '
    "No implementation details (e.g. which model, which library, which query language). "
    "Only when the query asks for a BOUNDED subset (e.g. 'ten pairs', 'top 5'), so any "
    "record you set aside is replaceable by another that also qualifies, you may name an "
    "extra label for the genuinely undecidable ones (e.g. '(positive/negative/unclear)'). "
    "Use it ONLY for records that truly support neither class -- never as the default and "
    "never for failures -- because over-applying it swallows records that were clearly "
    "labelable and starves the answer. If the query counts, groups, ranks or otherwise "
    "assigns EVERY record, state the label set exactly as the query does."
)

_REVISION_ACTION_FIELD_INSTRUCTION = (
    '- "action": a query-specific Verb + Subject phrase with concrete details for this query. '
    "The necessary logic and parameter inputs will be filled in later."
)

CLARIFICATION_PROMPT = dedent(
    """
    ## System
    You are an expert natural language query parser for KathDB, a multi-modal database (including table, text, images, audio, and videos).
    KathDB can perform multimodal data understanding and reasoning in additional to traditional relational database operations.

    ## Instructions
    Analyze the following user query for a multi-modal database system: "{question}"

    Look for ambiguous terms or unresolvable sub-queries that could have multiple interpretations
    in a multimodal database context.

    Your previous clarification questions history (in order of original query, your question, refined query):
    {previous_questions}
    Do not repeat any previous questions


    Your goal is that, after clarification,
    the query becomes unambiguous in terms of *what information is needed to answer the question*,
    while remaining succinct and containing no implementation details.

    Rules:
    1. If you find one ambiguous term that needs clarification, set status to "clarify",
       provide a single clarification question in the "question" field, and provide at least
       2 options (labeled sequentially A, B, ...) in the "options" field.
       Each option must represent a substantively different interpretation of the ambiguous term.
       Generate only as many options as there are genuinely distinct interpretations.
       Each option must describe a concrete meaning or interpretation, not a method.
       Ground each option in the provided relation schemas whenever possible (i.e., reference specific columns and tables that support the interpretation. e.g., "Measure how popular a book is by its sales rank (table: books; column: sales_rank)").
    2. If the query has clear, unambiguous terms, set status to "clear" and leave question and options null.
    3. If a term does not fall into the specified ambiguity category above, treat it as clear.
    4. Do NOT expose or ask about any implementation details (e.g., models, algorithms, metrics, thresholds).
    5. You can assume that all and only query-specific data needed for each action is available in the input relations, for example, if one action needs receipt images, the image relation will be provided as input.
    """
).strip()


REFINE_QUERY_PROMPT = dedent(
    """
    ## System
    You are an expert natural language query parser for KathDB, a multi-modal database (including table, text, images, audio, and videos).
    KathDB can perform multimodal data understanding and reasoning in additional to traditional relational database operations.

    ## Instructions
    You previously asked a clarification question:
    {clarification_question}
    based on the original query:
    "{original_question}"

    The user provided the following clarification:
    {user_clarification}

    Incorporate this clarification into the original query to resolve the specific ambiguity.

    Only modify the original query to clarify *what information is requested*.
    Preserve all other parts of the query verbatim.
    You can assume that all and only query-specific data needed for each action is available in the input relations, for example, if one action needs receipt images, the image relation will be provided as input.

    The refined query must:
    - Remove the identified ambiguity
    - Remain natural-language and high-level
    - Contain no implementation details
    """
).strip()


REVISION_PROMPT = dedent(
    """
    ## System
    You are an expert natural language query parser for KathDB, a multi-modal database (including table, text, images, audio, and videos).
    KathDB can perform multimodal data understanding and reasoning in additional to traditional relational database operations.

    ## Instructions
    You previously have drafted a query sketch for answering a user's question with a multi-modal database system.
    Your previous query sketch was:
    {sketch}

    The user's feedback on your draft was:
    {human_feedback}

    Revise the original draft so it addresses, and ONLY addresses the feedback while preserving accurate, useful details from the initial plan.
    If user reply with unclear, do not make any changes to the original draft and leave it for the execution engine to resolve at runtime.
    Each action must include:
    {name_instruction}
    {action_field_instruction}
    {op_kind_instruction}
    """
).strip()


_ATOMICITY_RULE = (
    "- Each action is exactly ONE op: either one SEMANTIC op "
    "(a single ML/LLM/VLM inference problem) or one RELATIONAL op "
    f"whose op_kind is one of: {_CANONICAL_REL_OP_KINDS}. "
    "A RELATIONAL action maps 1:1 to an extended relational algebra "
    "operator — not to an ad-hoc English verb-phrase.\n"
)

# Atomicity rule with ``fn_coarsening``: one library function may license one coarse action.
_ATOMICITY_RULE_FN_AWARE = (
    "- FIRST, scan `## Available Functions`. Whenever a SINGLE function's "
    "documented purpose (its `use_when`) covers what would otherwise be "
    "several adjacent steps of this query, you should PREFER to emit ONE "
    "coarse action spanning exactly that function's scope rather than "
    "splitting those steps into separate actions. Set that action's op_kind "
    'to "SEMANTIC" and list that covering function in selected_functions; you '
    "MAY also add any other plausibly-relevant functions — the executor's "
    "code-generation step decides exactly which to use and how to compose "
    "them. Write its `action` to describe the whole bundled task. Coarsen only "
    "a chunk a listed function covers (which therefore includes inference); "
    "do NOT "
    "coarsen a step no listed function covers. The cost-ordering rule below "
    "still governs the ORDER of actions — keep cheap relational filters before "
    "a coarse action — but it does NOT require splitting the steps a covering "
    "function already performs internally.\n"
    "- For every remaining part of the query (anything no single function "
    "covers), fall back to the default: each action is exactly ONE op — one "
    "SEMANTIC op (a single ML/LLM/VLM inference problem) or one RELATIONAL op "
    f"whose op_kind is one of: {_CANONICAL_REL_OP_KINDS} (mapping 1:1 to an "
    "extended relational algebra operator, not an ad-hoc verb-phrase).\n"
)

_OP_KIND_INSTRUCTION = (
    '- "op_kind": operator type from a fixed vocabulary. '
    '"SEMANTIC" if this step needs an ML/LLM/VLM inference call to '
    "compute its output (one model call per input row or per group). "
    "For a pure data transformation computable by DuckDB or pandas "
    "without any model inference, use the specific relational algebra "
    f"operator: {_CANONICAL_REL_OP_KINDS}."
)


# Worked demonstration for ACTION_SKETCH_PROMPT.
_ATOMIC_DEMO = (
    "## Demonstration\n"
    'Example query: "For all products priced under 130, determine solely from '
    "the product image whether it depicts white socks. Return a single "
    'column: id."\n'
    "Catalog tables: products(id, price, ...), product_images(id, image_path).\n"
    "An atomic decomposition (5 actions, one SEMANTIC op isolated between "
    "atomic RELATIONAL ops):\n"
    "[\n"
    '  {"name": "filter_cheap", "action": "Filter products where '
    'price < 130", "inputs": ["products"], "output": "cheap_products", '
    '"output_type": "dataframe", "op_kind": "RELATIONAL-FILTER"},\n'
    '  {"name": "attach_images", "action": "Join cheap_products with '
    'product_images on id to attach image_path", "inputs": ["cheap_products", '
    '"product_images"], "output": "cheap_products_with_images", "output_type": '
    '"dataframe", "op_kind": "RELATIONAL-JOIN"},\n'
    '  {"name": "classify_white_socks", "action": "Determine from product '
    'image whether it depicts white socks (yes/no)", "inputs": '
    '["cheap_products_with_images"], "output": "products_with_classification", '
    '"output_type": "dataframe", "op_kind": "SEMANTIC"},\n'
    '  {"name": "keep_white_socks", "action": "Keep rows where '
    'classification is yes", "inputs": ["products_with_classification"], '
    '"output": "white_socks_products", "output_type": "dataframe", '
    '"op_kind": "RELATIONAL-FILTER"},\n'
    '  {"name": "select_id", "action": "Project id", "inputs": '
    '["white_socks_products"], "output": "result", "output_type": "dataframe", '
    '"op_kind": "RELATIONAL-PROJECT"}\n'
    "]\n"
    "Note: names are query-specific descriptors. op_kind carries the "
    "fixed relational algebra type (RELATIONAL-FILTER, RELATIONAL-JOIN, etc.) "
    "or SEMANTIC for inference ops.\n"
    "Contrast — the query above keeps EVERY qualifying row, so a two-value label is right. "
    "When the query instead asks for a bounded subset (top-k / 'ten pairs' / LIMIT), "
    "an extra label is usually the better choice, because a set-aside row is simply "
    "replaced by another that qualifies:\n"
    '  {"name": "classify_sentiment", "action": "Classify each review\'s sentiment '
    '(positive/negative/unclear) from review_text", "inputs": ["reviews"], '
    '"output": "reviews_with_sentiment", "output_type": "dataframe", "op_kind": "SEMANTIC"},\n'
    '  {"name": "take_ten_pairs", "action": "Pair positive with negative reviews on id, '
    'limit 10", "inputs": ["reviews_with_sentiment"], "output": "result", '
    '"output_type": "dataframe", "op_kind": "RELATIONAL-LIMIT"}'
)

# Same demonstration with the ``selected_functions`` field of ActionItemWithFunctions.
_ATOMIC_DEMO_WITH_FUNCTIONS = (
    "## Demonstration\n"
    'Example query: "For all products priced under 130, determine solely from '
    "the product image whether it depicts white socks. Return a single "
    'column: id."\n'
    "Catalog tables: products(id, price, ...), product_images(id, image_path).\n"
    "An atomic decomposition (5 actions, one SEMANTIC op isolated between "
    "atomic RELATIONAL ops):\n"
    "[\n"
    '  {"name": "filter_cheap", "action": "Filter products where '
    'price < 130", "inputs": ["products"], "output": "cheap_products", '
    '"output_type": "dataframe", "op_kind": "RELATIONAL-FILTER", '
    '"selected_functions": []},\n'
    '  {"name": "attach_images", "action": "Join cheap_products with '
    'product_images on id to attach image_path", "inputs": ["cheap_products", '
    '"product_images"], "output": "cheap_products_with_images", "output_type": '
    '"dataframe", "op_kind": "RELATIONAL-JOIN", '
    '"selected_functions": []},\n'
    '  {"name": "classify_white_socks", "action": "Determine from product '
    'image whether it depicts white socks (yes/no)", "inputs": '
    '["cheap_products_with_images"], "output": "products_with_classification", '
    '"output_type": "dataframe", "op_kind": "SEMANTIC", '
    '"selected_functions": []},\n'
    '  {"name": "keep_white_socks", "action": "Keep rows where '
    'classification is yes", "inputs": ["products_with_classification"], '
    '"output": "white_socks_products", "output_type": "dataframe", '
    '"op_kind": "RELATIONAL-FILTER", "selected_functions": []},\n'
    '  {"name": "select_id", "action": "Project id", "inputs": '
    '["white_socks_products"], "output": "result", "output_type": "dataframe", '
    '"op_kind": "RELATIONAL-PROJECT", "selected_functions": []}\n'
    "]\n"
    "Note: names are query-specific descriptors. op_kind carries the "
    "fixed relational algebra type (RELATIONAL-FILTER, RELATIONAL-JOIN, etc.) "
    "or SEMANTIC for inference ops. "
    "``selected_functions`` is left empty because no pre-built function "
    "in `## Available Functions` matched any step.\n"
    "Contrast — the query above keeps EVERY qualifying row, so a two-value label is right. "
    "When the query instead asks for a bounded subset (top-k / 'ten pairs' / LIMIT), "
    "an extra label is usually the better choice, because a set-aside row is simply "
    "replaced by another that qualifies, e.g. an action reading "
    "\"Classify each review's sentiment (positive/negative/unclear) from review_text\" "
    "feeding a \"limit 10\" step."
)

# Sole demonstration when ``fn_coarsening`` is on.
_COARSE_FN_DEMO = (
    "## Demonstration — function-directed grouping (prefer a covering function)\n"
    "Suppose `## Available Functions` lists "
    "`flag_products_with_majority_negative_reviews` (use_when: \"classify each "
    "review's sentiment with an LLM, then keep products whose reviews are majority "
    'negative; returns product ids"). For the query "Among products priced under '
    '50, list the ones that are mostly panned by reviewers" the price filter is '
    "plain SQL that no function covers (stays atomic), while the entire "
    "panned-detection chunk (classify each review -> count negatives per product "
    "-> keep the majority-negative ones) IS covered by that one function — so "
    "emit it as a SINGLE coarse action instead of three:\n"
    "[\n"
    '  {"name": "cheap_products", "action": "Filter products where price < 50", '
    '"inputs": ["products"], "output": "cheap_products", "output_type": '
    '"dataframe", "op_kind": "RELATIONAL-FILTER", "selected_functions": []},\n'
    '  {"name": "mostly_panned_products", "action": "Keep cheap products whose '
    "reviews are majority negative (classify each review's sentiment, then keep "
    'products where most reviews are negative)", "inputs": ["cheap_products", '
    '"reviews"], "output": "result", "output_type": "dataframe", "op_kind": '
    '"SEMANTIC", "selected_functions": '
    '["flag_products_with_majority_negative_reviews"]}\n'
    "]\n"
    "The sentiment chunk collapses to ONE coarse action because a single function "
    "spans its whole scope, so code generation imports and calls it directly "
    "(from kathdb.fn import flag_products_with_majority_negative_reviews) instead "
    "of rebuilding the logic. The price filter stays its own atomic action because "
    "no function covers it. If several functions are relevant to the same chunk "
    "(e.g. the whole-chunk function plus a finer review-sentiment classifier), list "
    "them ALL in that action's selected_functions — the code-generation step picks "
    "which to use and how to compose them. Apply this same test to every part of "
    "the query: prefer a covering function, otherwise stay atomic.\n"
    "Contrast — bounded subset: if the query instead asked for a small fixed number "
    "of results (e.g. 'list 5 cheap products that reviewers clearly praise'), the "
    "same function-preference test applies, and the sentiment vocabulary may "
    "additionally include an escape label such as 'unclear' so ambiguous reviews "
    "are set aside rather than forced into positive/negative — the bounded output "
    "is filled from the confident ones. Never add such a label when the query "
    "counts, groups, or ranks EVERY record, as in the majority-negative query "
    "above, where every review must receive a real label."
)

ACTION_SKETCH_PROMPT = dedent(
    """
    ## System
    You are an expert NL query parser for KathDB (multi-modal DB). Given an NL query and a relational catalog, {sketch_directive}

    ## Instructions
    Each action must have:
    {name_instruction}
    {action_field_instruction}
    {io_field_instructions}
    - "output_type": Python value-shape of this action's output. One of "dataframe" (a table), "int", "float", "string", "bool", "list", or "dict" (a single non-tabular Python value).
    {op_kind_instruction}

    Rules:
    {constraint_rules}
    Assumptions: Inputs contain all data needed per action. Multimodal view tables are pre-registered and auto-populated — reference them; do NOT generate populate actions.

    {demonstration}

    ## User Query
    {question}
"""
).strip()


def format_clarification_prompt(
    *,
    question: str,
    previous_questions: list[str],
    schemas: Sequence[str] | None = None,
) -> str:
    """Return the clarification prompt filled with the supplied question and optional schemas."""
    prompt = _format_template(
        CLARIFICATION_PROMPT, question=question, previous_questions=previous_questions  # type: ignore
    )
    if schemas:
        prompt += "\n\n## Relevant Relation Schemas\n" + "\n".join(schemas) + "\n"
    return prompt


def format_revision_prompt(
    *,
    sketch: str,
    human_feedback: str,
    schemas: Sequence[str] | None = None,
) -> str:
    """Return the revision prompt populated with the previous draft and feedback."""
    prompt = _format_template(
        REVISION_PROMPT,
        sketch=sketch,
        human_feedback=human_feedback,
        name_instruction=_REVISION_NAME_INSTRUCTION,
        action_field_instruction=_REVISION_ACTION_FIELD_INSTRUCTION,
        op_kind_instruction=_OP_KIND_INSTRUCTION,
    )
    if schemas:
        prompt += "\n\n## Relevant Relation Schemas\n" + "\n".join(schemas) + "\n"
    return prompt


def format_action_query_sketch_prompt(
    *,
    question: str,
    schemas: Sequence[str] | None = None,
) -> str:
    """The atomic action-sketch prompt."""
    rules = (
        "- Succinct Verb + Subject; no implementation detail (model, library, query language).\n"
        + _ATOMICITY_RULE
        + "- Order for cost and reproducibility: run cheap, deterministic work "
        "(relational filters, and extracting/classifying a needed attribute with one "
        "model call per row) BEFORE expensive operations, and compute a derived "
        "attribute (e.g. genre, brand, color) BEFORE any action that groups, filters, "
        "or joins on it. NEVER cross-join two tables and verify each pair with a model "
        "call (O(n*m) calls) — instead extract the join key per row (one call each) "
        "and then equi-join on the extracted value. Pushing relational filters and "
        "key-extraction ahead of per-row inference shrinks how many rows reach the "
        "expensive op, cutting cost and making the plan deterministic.\n"
        + "- If the query expects a tabular result with specific column names, add a final step to match those names.\n"
    )

    sketch_directive = (
        "output semantically-atomic actions that form a query execution plan."
    )

    io_field_instructions = (
        '- "inputs": names of inputs this action consumes — catalog tables '
        "or previously-defined values.\n"
        '    - "output": Unique name for this action\'s output.'
    )

    prompt = _format_template(
        ACTION_SKETCH_PROMPT,
        question=question,
        name_instruction=_NAME_INSTRUCTION,
        action_field_instruction=_ACTION_FIELD_INSTRUCTION,
        constraint_rules=rules,
        sketch_directive=sketch_directive,
        io_field_instructions=io_field_instructions,
        op_kind_instruction=_OP_KIND_INSTRUCTION,
        demonstration=_ATOMIC_DEMO,
    )
    if schemas:
        prompt += "\n\n## Relevant Relation Schemas\n" + "\n".join(schemas) + "\n"
    return prompt


ACTION_SKETCH_WITH_FUNCTIONS_PROMPT = dedent(
    """
    ## System
    You are an expert NL query parser for KathDB (multi-modal DB). Given an NL query, a relational catalog, and a list of pre-built functions, {sketch_directive}

    ## Instructions
    Each action must have:
    {name_instruction}
    {action_field_instruction}
    {io_field_instructions}
    - "output_type": Python value-shape of this action's output. One of "dataframe" (a table), "int", "float", "string", "bool", "list", or "dict" (a single non-tabular Python value).
    {op_kind_instruction}
    - "selected_functions": names of pre-built functions from `## Available Functions` whose docs match this action. Include EVERY plausible match — multiple functions can apply and may be composed downstream. Leave empty when none fits and codegen will implement from scratch.

    Rules:
    {constraint_rules}
    Assumptions: Inputs contain all data needed per action. Multimodal view tables are pre-registered and auto-populated — reference them; do NOT generate populate actions.

    {functions_block}

    {demonstration}

    ## User Query
    {question}
"""
).strip()


PICK_FUNCTIONS_PROMPT = dedent(
    """
    ## System
    You are an expert function selector for KathDB's NL parser. You read one action from a query plan and pick every pre-built function whose documentation matches.

    ## Instructions
    From `## Available Functions`, select ALL pre-built functions whose docs fit this action. Multiple functions can apply and may be composed downstream. For each selection give the function name and one short reasoning line. Leave the list empty when no listed function fits — code generation will implement from scratch.

    Prefer non-LLM functions when they suffice. Do not invent function names; choose only from the list.

    ## Query Context
    Full user query: "{nl_query}"
    Note: the action below is ONE step of the overall plan. Pick functions for THIS step only.

    ## Action
    Name: {action_name}
    Description: {action}
    Inputs: {inputs}
    Output: {output} (output_type: {output_type})

    {functions_block}
    """
).strip()


def format_action_query_sketch_with_functions_prompt(
    *,
    question: str,
    functions_block: str,
    schemas: Sequence[str] | None = None,
    fn_coarsening: bool = False,
) -> str:
    """Sketch prompt fused with library-function picking (one LLM call). With
    ``fn_coarsening`` the atomicity rule and demonstration switch to their
    function-directed variants."""
    rules = (
        "- Succinct Verb + Subject; no implementation detail (model, library, query language).\n"
        + (_ATOMICITY_RULE_FN_AWARE if fn_coarsening else _ATOMICITY_RULE)
        + "- Order for cost and reproducibility: run cheap, deterministic work "
        "(relational filters, and extracting/classifying a needed attribute with one "
        "model call per row) BEFORE expensive operations, and compute a derived "
        "attribute (e.g. genre, brand, color) BEFORE any action that groups, filters, "
        "or joins on it. NEVER cross-join two tables and verify each pair with a model "
        "call (O(n*m) calls) — instead extract the join key per row (one call each) "
        "and then equi-join on the extracted value. Pushing relational filters and "
        "key-extraction ahead of per-row inference shrinks how many rows reach the "
        "expensive op, cutting cost and making the plan deterministic.\n"
        + "- If the query expects a tabular result with specific column names, add a final step to match those names.\n"
    )

    if fn_coarsening:
        sketch_directive = (
            "output actions that form a query execution plan, each annotated with "
            "matching pre-built functions. PREFER ONE coarse action wherever a "
            "single available function covers several adjacent steps; keep actions "
            "semantically atomic everywhere no function applies."
        )
    else:
        sketch_directive = (
            "output semantically-atomic actions that form a query execution plan, "
            "each annotated with all matching pre-built functions."
        )

    io_field_instructions = (
        '- "inputs": names of inputs this action consumes — catalog tables '
        "or previously-defined values.\n"
        '    - "output": Unique name for this action\'s output.'
    )

    # With coarsening on the coarse demo is the sole demonstration (it includes an
    # atomic action, so the field shape is still shown).
    demonstration = (
        _COARSE_FN_DEMO if fn_coarsening else _ATOMIC_DEMO_WITH_FUNCTIONS
    )

    prompt = _format_template(
        ACTION_SKETCH_WITH_FUNCTIONS_PROMPT,
        question=question,
        name_instruction=_NAME_INSTRUCTION,
        action_field_instruction=_ACTION_FIELD_INSTRUCTION,
        constraint_rules=rules,
        sketch_directive=sketch_directive,
        io_field_instructions=io_field_instructions,
        op_kind_instruction=_OP_KIND_INSTRUCTION,
        functions_block=functions_block or "## Available Functions\n(none registered)",
        demonstration=demonstration,
    )
    if schemas:
        prompt += "\n\n## Relevant Relation Schemas\n" + "\n".join(schemas) + "\n"
    return prompt


def format_pick_functions_prompt(
    *,
    action_name: str,
    action: str,
    inputs: Sequence[str],
    output: str,
    output_type: str,
    functions_block: str,
    nl_query: str,
) -> str:
    """Per-action pre-built function picking prompt (one call per action)."""
    return _format_template(
        PICK_FUNCTIONS_PROMPT,
        action_name=action_name,
        action=action or "(no description)",
        inputs=", ".join(inputs) if inputs else "None",
        output=output or "(unnamed)",
        output_type=output_type or "dataframe",
        functions_block=functions_block or "## Available Functions\n(none registered)",
        nl_query=nl_query or "(unavailable)",
    )


def format_refine_query_prompt(
    *,
    original_question: str,
    clarification_question: str,
    user_clarification: str,
    schemas: Sequence[str] | None = None,
) -> str:
    """Return the refine query prompt populated with the question details and optional schemas."""
    prompt = _format_template(
        REFINE_QUERY_PROMPT,
        original_question=original_question,
        clarification_question=clarification_question,
        user_clarification=user_clarification,
    )
    if schemas:
        prompt += "\n\n## Relevant Relation Schemas\n" + "\n".join(schemas) + "\n"
    return prompt


def _format_template(template: str, **variables: str) -> str:
    try:
        return template.format(**variables)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing template variable: {missing}") from None
