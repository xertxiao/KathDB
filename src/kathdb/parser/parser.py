"""Natural-language query -> ordered list of atomic actions (the query sketch).

``ActionNLParser.run`` is two bounded loops: an optional clarification loop
(model asks, user answers, query is refined) and a sketch/review loop (model
drafts, user accepts or corrects). Both human steps are skipped in ``auto_mode``.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


from .action import Action
from .prompts import (
    format_action_query_sketch_prompt,
    format_action_query_sketch_with_functions_prompt,
    format_clarification_prompt,
    format_pick_functions_prompt,
    format_refine_query_prompt,
    format_revision_prompt,
)
from .response_schemas import (
    ActionSketchResponse,
    ActionSketchWithFunctionsResponse,
    ClarificationResponse,
    PickFunctionsResponse,
    RefinedQueryResponse,
)
from ..common.public_state_schemas import QueryInState, QueryOutState
from ..common.context import DBContext
from ..common.function_manager import FunctionManager
from ..common.logger import get_logger
from ..common.utils import invoke_structured_with_retry

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)

__all__ = ["ActionNLParser"]

# Exact-match (trim + lowercase) acceptance replies during human review.
_ACCEPT_RESPONSES = frozenset({"accept", "accepted", "ok", "okay", "lgtm", "yes", "y"})

# Working-state keys that accumulate across steps; every other key is overwritten.
_APPEND_KEYS = frozenset({"reviews_messages", "clarification_messages", "runtime_deferred"})


def _is_acceptance(message: str | None) -> bool:
    """True iff *message* is an exact acceptance reply."""
    return bool(message) and message.strip().lower() in _ACCEPT_RESPONSES


def _merge(state: dict[str, Any], update: dict[str, Any]) -> None:
    """Apply a step's update to the working state (lists in ``_APPEND_KEYS`` append)."""
    for key, value in update.items():
        if key in _APPEND_KEYS:
            state.setdefault(key, []).extend(value)
        else:
            state[key] = value


class ActionNLParser:
    """Optional human-in-the-loop clarification + review, then an atomic action
    sketch (one SEMANTIC or RELATIONAL op per action).

    With ``function_reuse`` each action is annotated with matching library
    functions: either by a separate per-action pick after the sketch (default) or,
    with ``sketch_with_functions``, in the same LLM call that drafts the sketch (the
    library is shown to the sketch model; with ``fn_coarsening`` a function covering
    several adjacent steps licenses one coarse action). An empty library always
    falls back to the plain atomic prompt.
    """

    def __init__(
        self,
        *,
        max_clarifications: int = 5,
        max_revisions: int = 3,
        clarification_llm: BaseChatModel,
        sketch_llm: BaseChatModel,
        revision_llm: BaseChatModel,
        auto_mode: bool = False,
        function_reuse: bool = True,
        fn_manager: FunctionManager | None = None,
        sketch_with_functions: bool = False,
        fn_coarsening: bool = False,
        max_pick_concurrency: int = 10,
    ) -> None:
        self.max_clarifications = max_clarifications
        self.max_revisions = max_revisions
        self.clarification_llm: BaseChatModel = clarification_llm
        self.sketch_llm: BaseChatModel = sketch_llm
        self.revision_llm: BaseChatModel = revision_llm
        # True skips both human review points (clarification, sketch).
        self.auto_mode = auto_mode
        # True annotates actions with library functions (pick step or fused sketch).
        self.function_reuse = function_reuse
        self._fn_manager = fn_manager or FunctionManager()
        # True fuses sketch + function pick into one call (no separate pick step).
        self.sketch_with_functions = sketch_with_functions
        # Honoured by the fused sketch only.
        self.fn_coarsening = fn_coarsening
        self.max_pick_concurrency = max_pick_concurrency

    def run(self, q_in: QueryInState) -> QueryOutState:
        """Process the NL question through the clarification and review loops."""
        logger.info("Starting NL parsing process.")
        state: dict[str, Any] = {
            "q_in": q_in["q_in"],
            "relation_context": q_in["relation_context"],
            "input_rel_names": list(q_in.get("input_rel_names") or []),
            "actions": [],
            "reviews_messages": [],
            "clarification_messages": [],
            "runtime_deferred": [],
            "reviews_count": 0,
            "clarifications_count": 0,
        }

        # Clarification loop: bounded by ``max_clarifications`` in the decision.
        while True:
            _merge(state, self._clarification_check_node(state))
            if self._decide_clarification_needed_edge(state) == "next":
                break
            _merge(state, self._get_clarification_node(state))

        # Sketch + review loop: bounded by ``max_revisions`` in the decision.
        while True:
            _merge(state, self._draft_sketch_node(state))
            if self.function_reuse and not self.sketch_with_functions:
                _merge(state, self._pick_functions_node(state))
            _merge(state, self._get_human_feedback_node(state))
            if self._decide_revision_feedback_edge(state) == "next":
                break

        return QueryOutState(
            q_in=state["q_in"],
            actions=state["actions"],
            relation_context=state["relation_context"],
            input_rel_names=state["input_rel_names"],
        )

    # ------------------------------------------------------------------
    # Steps and decisions (each step returns an update to the working state)

    def _clarification_check_node(self, state: dict[str, Any]) -> dict:
        """entry-node, check for clarification need"""
        q_in = state["q_in"]
        assert (
            isinstance(q_in, str) and q_in.strip()
        ), "Input question must be a non-empty string."

        if self.auto_mode:
            logger.info("auto_mode: skipping clarification check.")
            return {
                "clarification_messages": ["CLEAR"],
                "clarification_status": "clear",
                "clarification_options": [],
                "reviews_count": state.get("reviews_count", 0),
                "clarifications_count": state.get("clarifications_count", 0),
            }

        rc = state["relation_context"]
        df_descriptions = rc.describe_all_tables()
        p = format_clarification_prompt(
            question=q_in,
            previous_questions=state.get("clarification_messages", []),
            schemas=df_descriptions,
        )
        response = self._invoke_structured(
            p, llm=self.clarification_llm, schema=ClarificationResponse
        )
        logger.info(
            f"🤔 Clarification check: status={response.status}, question={response.question}"
        )

        reviews_count = state.get("reviews_count", 0)
        clarifications_count = state.get("clarifications_count", 0)

        logger.warning(
            f"clarification_messages: {state.get('clarification_messages', [])}, "
            f"reviews_count: {reviews_count}, "
            f"clarifications_count: {clarifications_count}"
        )

        clarification_message = (
            response.question if response.status == "clarify" else "CLEAR"
        )

        clarification_options: list[dict[str, str]] = []
        if response.status == "clarify" and response.options:
            clarification_options = [
                {"label": opt.label, "description": opt.description}
                for opt in response.options
            ]

        return {
            "clarification_messages": [clarification_message],
            "clarification_status": response.status,
            "clarification_options": clarification_options,
            "reviews_count": reviews_count,
            "clarifications_count": clarifications_count,
        }

    def _get_clarification_node(self, state: dict[str, Any]) -> dict:
        """Get user's response to clarification question."""
        question_text = state["clarification_messages"][-1]
        options: list[dict[str, str]] = state.get("clarification_options", [])

        option_labels: set[str] = set()
        if options:
            option_labels = {opt["label"].upper() for opt in options}
        # "Decide at runtime" gets the next sequential letter after all options
        defer_label = chr(ord("A") + len(options)) if options else "D"

        logger.interact("---" * 20)
        logger.interact("❓ Clarification needed")
        logger.interact(f"  Original question: {state['q_in']}")
        logger.interact(f"  Question: {question_text}")
        if options:
            for opt in options:
                logger.interact(f"    {opt['label']}: {opt['description']}")
        logger.interact(f"    {defer_label}: Decide at runtime (no preference now)")
        logger.interact("---" * 20)

        valid_choices = (
            "/".join([opt["label"].upper() for opt in options] + [defer_label])
            if options
            else defer_label
        )
        h_response = input(
            f"\nPlease select an option ({valid_choices}) or type a custom response: "
        ).strip()

        clarifications_count = (
            state["clarifications_count"] + 1 if state["clarifications_count"] else 1
        )

        if h_response.upper() == defer_label:
            logger.info(
                f"📋 User chose {defer_label} (runtime-deferred) for: {question_text}"
            )
            return {
                "q_in": state["q_in"],
                "clarifications_count": clarifications_count,
                "clarification_messages": [
                    f"{defer_label}: Decide at runtime",
                    "Ambiguity deferred to runtime",
                    state["q_in"],
                ],
                "runtime_deferred": [question_text],
            }

        resolved_response = h_response
        if options and h_response.upper() in option_labels:
            for opt in options:
                if opt["label"].upper() == h_response.upper():
                    resolved_response = opt["description"]
                    break

        rc = state["relation_context"]
        df_descriptions = rc.describe_all_tables()

        p = format_refine_query_prompt(
            original_question=state["q_in"],
            clarification_question=state["clarification_messages"],
            user_clarification=resolved_response,
            schemas=df_descriptions,
        )
        response = self._invoke_structured(
            p, llm=self.clarification_llm, schema=RefinedQueryResponse
        )
        logger.info(f"📋 Refined question: {response.refined_query}\n")
        return {
            "q_in": response.refined_query,
            "clarifications_count": clarifications_count,
            "clarification_messages": [
                resolved_response,
                "Revising original query",
                response.refined_query,
            ],
        }

    def _decide_clarification_needed_edge(
        self, state: dict[str, Any]
    ) -> Literal["clarify", "next"]:
        """Decide whether clarification is needed based on model response"""
        clarification_count = (
            state["clarifications_count"] if state["clarifications_count"] else 0
        )
        if clarification_count >= self.max_clarifications:
            logger.warning("Maximum clarification attempts reached.")
            return "next"
        status = state.get("clarification_status")
        if status == "clear":
            return "next"
        if status == "clarify":
            return "clarify"
        # Fallback: string check on the last message.
        last_message = state["clarification_messages"][-1].lower()
        if "clear" in last_message[:6]:
            return "next"
        return "clarify"

    def _draft_sketch_node(self, state: dict[str, Any]) -> dict:
        """Draft (or revise) a query sketch with chain-of-thought reasoning.

        Revisions always use the plain revision prompt; function picks carry over
        on actions whose name is unchanged. A fresh draft uses the function-directed
        prompt only with ``sketch_with_functions`` and a non-empty library.
        """
        q_in = state["q_in"]
        assert (
            isinstance(q_in, str) and q_in.strip()
        ), "Input question must be a non-empty string."
        rc = state["relation_context"]
        df_descriptions = rc.describe_all_tables()

        feedback_messages = state.get("reviews_messages") or []
        last_feedback = feedback_messages[-1] if feedback_messages else None
        has_revision_feedback = bool(last_feedback) and not _is_acceptance(
            last_feedback
        )
        previous_sketch = state.get("actions")
        previous_by_name: dict[str, Action] = {
            a.name: a for a in (previous_sketch or [])
        }

        revision_ct = state.get("reviews_count", 0)
        if has_revision_feedback and previous_sketch:
            p = format_revision_prompt(
                sketch=str(previous_sketch),
                human_feedback=str(last_feedback),
                schemas=df_descriptions,
            )
            response = self._invoke_structured(
                p, llm=self.revision_llm, schema=ActionSketchResponse
            )
            parsed_actions = [
                self._to_action(
                    item,
                    selected_functions=(
                        previous_by_name[item.name].selected_functions
                        if item.name in previous_by_name
                        else []
                    ),
                )
                for item in response.actions
            ]
            revision_ct += 1
        else:
            valid_names: set[str] = set()
            if self.sketch_with_functions:
                valid_names = set(self._fn_manager.discover_functions().keys())
            if valid_names:
                p = format_action_query_sketch_with_functions_prompt(
                    question=q_in,
                    functions_block=self._fn_manager.render_functions_summary(),
                    schemas=df_descriptions,
                    fn_coarsening=self.fn_coarsening,
                )
                response = self._invoke_structured(
                    p, llm=self.sketch_llm, schema=ActionSketchWithFunctionsResponse
                )
                parsed_actions = [
                    self._to_action(
                        item,
                        selected_functions=[
                            n
                            for n in (item.selected_functions or [])
                            if n in valid_names
                        ],
                    )
                    for item in response.actions
                ]
            else:
                # Plain atomic prompt (also the fallback for an empty library).
                p = format_action_query_sketch_prompt(
                    question=q_in, schemas=df_descriptions
                )
                response = self._invoke_structured(
                    p, llm=self.sketch_llm, schema=ActionSketchResponse
                )
                parsed_actions = [self._to_action(item) for item in response.actions]

        parsed_actions = self._inject_populate_actions(parsed_actions, rc)
        parsed_actions = self._dedupe_action_names(parsed_actions)
        return {
            "actions": parsed_actions,
            "reviews_count": revision_ct,
        }

    @staticmethod
    def _to_action(item: Any, *, selected_functions: list[str] | None = None) -> Action:
        """One response item -> :class:`Action`."""
        return Action(
            name=item.name,
            action=item.action,
            inputs=item.inputs,
            output=item.output,
            output_type=item.output_type,
            op_kind=item.op_kind,
            selected_functions=list(selected_functions or []),
        )

    @staticmethod
    def _dedupe_action_names(
        actions: list[Action],
    ) -> list[Action]:
        """Suffix repeated ``name``s (``_2``, ``_3``, ...) so every action, and hence
        every ``FAONode.op``, is unique. Output-name uniqueness is enforced downstream
        in ``build_fao_dag``."""
        from dataclasses import replace

        assigned: set[str] = set()
        out: list[Action] = []
        for a in actions:
            name = a.name or ""
            if not name:
                out.append(a)
                continue
            candidate = name
            suffix = 2
            while candidate in assigned:
                candidate = f"{name}_{suffix}"
                suffix += 1
            out.append(replace(a, name=candidate))
            assigned.add(candidate)
        return out

    @staticmethod
    def _format_sketch(sketch: list[Action]) -> str:
        """Pretty-print a query sketch for human review."""
        lines: list[str] = []
        for i, act in enumerate(sketch, 1):
            lines.append(f"  Step {i}: {act.name}")
            lines.append(f"    action : {act.action}")
            lines.append(
                f"    in     : {', '.join(act.inputs) if act.inputs else '(none)'}"
            )
            lines.append(f"    out    : {act.output}")
        return "\n".join(lines)

    def _get_human_feedback_node(self, state: dict[str, Any]) -> dict:
        """Get human feedback on the drafted query sketch"""
        if self.auto_mode:
            logger.info("auto_mode: skipping human review of query sketch.")
            return {"reviews_messages": ["accept"]}

        logger.interact("👋 Human review needed (reply exactly 'accept' to finalize)")
        logger.interact("─" * 60)
        logger.interact("Drafted query sketch:")
        logger.interact(self._format_sketch(state["actions"]))
        logger.interact("─" * 60)
        rc = state["relation_context"]
        df_descriptions = rc.describe_all_tables()
        if df_descriptions:
            logger.interact("Relevant relation schemas:")
            for desc in df_descriptions:
                logger.interact(desc)
            logger.interact("─" * 60)

        response = input(
            "\nPlease provide your feedback (reply exactly 'accept' to finalize, "
            "or type your corrections): "
        )
        logger.info(f"📋 Human feedback: {response}\n")
        return {
            "reviews_messages": [response],
        }

    def _decide_revision_feedback_edge(
        self, state: dict[str, Any]
    ) -> Literal["revise", "next"]:
        """Decide whether revision is needed based on human feedback"""
        last_message = state["reviews_messages"][-1]
        revision_count = state["reviews_count"]
        if revision_count >= self.max_revisions:
            logger.warning("Maximum revision attempts reached.")
            return "next"
        if _is_acceptance(last_message):
            return "next"
        return "revise"

    def _inject_populate_actions(
        self,
        actions: list[Action],
        rc: DBContext,
    ) -> list[Action]:
        """Prepend a populate action for every unpopulated multimodal view used as input."""
        referenced_views: dict[str, None] = {}  # ordered set
        for act in actions:
            for rel in act.inputs:
                if rc.is_view(rel) and rel not in referenced_views:
                    referenced_views[rel] = None

        if not referenced_views:
            return actions

        populate_actions: list[Action] = []
        for view_name in referenced_views:
            if rc.execute(f'SELECT COUNT(*) FROM "{view_name}"').fetchone()[0] > 0:
                continue
            vs = rc.get_view_source(view_name)
            assert vs is not None  # guaranteed by is_view check
            populate_actions.append(
                Action(
                    name=f"populate_{view_name}",
                    action=(
                        f"Populate {view_name} from "
                        f"{vs.source_table}.{vs.source_column} "
                        f"({vs.modality.value} data)"
                    ),
                    inputs=[vs.source_table],
                    output=view_name,
                    op_kind="RELATIONAL",
                )
            )

        return populate_actions + actions

    def _pick_functions_node(self, state: dict[str, Any]) -> dict:
        """Pick library functions per action (parallel LLM calls); mutates
        ``selected_functions`` in place. ``contextvars.copy_context().run`` keeps the
        child LLM runs under the parent trace."""
        actions: list[Action] = state.get("actions") or []
        if not actions:
            return {"actions": actions}

        functions_block = self._fn_manager.render_functions_summary()
        valid_names = set(self._fn_manager.discover_functions().keys())
        nl_query = state.get("q_in") or ""

        targets = [a for a in actions if not a.name.startswith("populate_")]
        if not targets:
            return {"actions": actions}

        def _pick_one(act: Action) -> tuple[Action, list[str]]:
            prompt = format_pick_functions_prompt(
                action_name=act.name,
                action=act.action,
                inputs=act.inputs,
                output=act.output,
                output_type=act.output_type,
                functions_block=functions_block,
                nl_query=nl_query,
            )
            try:
                resp = invoke_structured_with_retry(
                    prompt,
                    llm=self.sketch_llm,
                    schema=PickFunctionsResponse,
                    max_retries=3,
                )
                names = [s.function_name for s in resp.selected_functions]
            except Exception:
                logger.warning(
                    "Function picking failed for action '%s'; treating as no functions.",
                    act.name,
                    exc_info=True,
                )
                names = []
            return act, [n for n in names if n in valid_names]

        n_workers = min(self.max_pick_concurrency, len(targets))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for a in targets:
                ctx = contextvars.copy_context()
                futures.append(pool.submit(ctx.run, _pick_one, a))
            for fut in as_completed(futures):
                act, sel = fut.result()
                act.selected_functions = sel

        logger.info(
            "Parser pick_functions: annotated %d action(s); functions per action: %s",
            len(targets),
            {a.name: a.selected_functions for a in targets},
        )
        return {"actions": actions}

    def _invoke_structured(
        self, prompt: str, *, llm: BaseChatModel, schema: type[T], max_retries: int = 3
    ) -> T:
        """Invoke the LLM with structured output, retrying on validation errors."""
        return invoke_structured_with_retry(
            prompt, llm=llm, schema=schema, max_retries=max_retries
        )
