"""Natural-language query -> ordered list of atomic actions (the query sketch)."""

from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel


from .action import Action
from .parser_state_schemas import ParserState
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

__all__ = ["BaseParser", "ActionNLParser", "ActionNLParserWithFunctions"]

# Exact-match (trim + lowercase) acceptance replies during human review.
_ACCEPT_RESPONSES = frozenset({"accept", "accepted", "ok", "okay", "lgtm", "yes", "y"})


def _is_acceptance(message: str | None) -> bool:
    """True iff *message* is an exact acceptance reply."""
    return bool(message) and message.strip().lower() in _ACCEPT_RESPONSES


class BaseParser(ABC):
    """Minimal base class for NL parsers: compile a state graph, then run it."""

    def __init__(self) -> None:
        self.state_graph: Any | None = None

    @abstractmethod
    def compile(self) -> None:
        """Compile the LangGraph state machine into ``self.state_graph``."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self, q_in: QueryInState, *, config: RunnableConfig | None = None
    ) -> QueryOutState:
        """Process a question and return the resulting parser state."""
        raise NotImplementedError

    @abstractmethod
    def visualize(self) -> None:
        """Generate a visualization of the state graph."""
        raise NotImplementedError


class ActionNLParser(BaseParser):
    """Optional human-in-the-loop clarification + review, then an atomic action
    sketch (one SEMANTIC or RELATIONAL op per action) and, with function reuse on,
    a separate per-action pick of matching library functions."""

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
        fn_coarsening: bool = False,
        max_pick_concurrency: int = 10,
    ) -> None:
        super().__init__()
        self.max_clarifications = max_clarifications
        self.max_revisions = max_revisions
        self.clarification_llm: BaseChatModel = clarification_llm
        self.sketch_llm: BaseChatModel = sketch_llm
        self.revision_llm: BaseChatModel = revision_llm
        # True skips both human review points (clarification, sketch).
        self.auto_mode = auto_mode
        # True adds the ``pick_functions`` node.
        self.function_reuse = function_reuse
        self._fn_manager = fn_manager or FunctionManager()
        # Honoured by ActionNLParserWithFunctions only.
        self.fn_coarsening = fn_coarsening
        self.max_pick_concurrency = max_pick_concurrency

    def compile(self) -> None:
        graph = StateGraph(
            ParserState, input_schema=QueryInState, output_schema=QueryOutState
        )

        # nodes
        graph.add_node("clarification_check", self._clarification_check_node)
        graph.add_node("get_clarification", self._get_clarification_node)
        graph.add_node("draft_sketch", self._draft_sketch_node)
        if self.function_reuse:
            graph.add_node("pick_functions", self._pick_functions_node)
        graph.add_node("get_human_review", self._get_human_feedback_node)

        # edges
        graph.add_edge(START, "clarification_check")
        graph.add_conditional_edges(
            "clarification_check",
            self._decide_clarification_needed_edge,
            {"clarify": "get_clarification", "next": "draft_sketch"},
        )
        graph.add_edge("get_clarification", "clarification_check")
        if self.function_reuse:
            graph.add_edge("draft_sketch", "pick_functions")
            graph.add_edge("pick_functions", "get_human_review")
        else:
            graph.add_edge("draft_sketch", "get_human_review")
        graph.add_conditional_edges(
            "get_human_review",
            self._decide_revision_feedback_edge,
            {"revise": "draft_sketch", "next": END},
        )

        self.state_graph = graph.compile()
        logger.info("State graph compiled successfully.")

    def visualize(self) -> None:
        """Visualize the compiled state graph if available."""
        if self.state_graph is None:
            raise RuntimeError(
                "State graph has not been compiled. Run `compile()` first."
            )
        else:
            try:
                from IPython.display import Image, display

                display(Image(self.state_graph.get_graph().draw_mermaid_png()))
            except Exception as e:
                logger.warning(
                    f"Failed to display graph image: {e}, falling back to ASCII."
                )
                print(self.state_graph.get_graph().draw_ascii())

    def run(
        self, q_in: QueryInState, *, config: RunnableConfig | None = None
    ) -> QueryOutState:
        """Process the NL question through clarification and review loops."""
        if self.state_graph is None:
            raise RuntimeError(
                "State graph has not been compiled. Run `compile()` first."
            )
        else:
            logger.info("Starting NL parsing process.")
            cfg: RunnableConfig = config or {
                "configurable": {"thread_id": 1},
                "recursion_limit": 20,
            }
            out = self.state_graph.invoke(q_in, cfg)
            return QueryOutState(**out)

    # ------------------------------------------------------------------
    # Internal nodes and edges

    def _clarification_check_node(self, state: ParserState, config=None) -> dict:
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

    def _get_clarification_node(self, state: ParserState, config=None) -> dict:
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
        self, state: ParserState, config=None
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

    def _draft_sketch_node(self, state: ParserState, config=None) -> dict:
        """Draft (or revise) a query sketch with chain-of-thought reasoning"""
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

        revision_ct = state.get("reviews_count", 0)
        if has_revision_feedback and previous_sketch:
            p = format_revision_prompt(
                sketch=str(previous_sketch),
                human_feedback=str(last_feedback),
                schemas=df_descriptions,
            )
            llm_for_sketch = self.revision_llm
            revision_ct += 1
        else:
            p = format_action_query_sketch_prompt(
                question=q_in,
                schemas=df_descriptions,
            )
            llm_for_sketch = self.sketch_llm

        response = self._invoke_structured(
            p, llm=llm_for_sketch, schema=ActionSketchResponse
        )
        parsed_actions = [
            Action(
                name=item.name,
                action=item.action,
                inputs=item.inputs,
                output=item.output,
                output_type=item.output_type,
                op_kind=item.op_kind,
            )
            for item in response.actions
        ]

        parsed_actions = self._inject_populate_actions(parsed_actions, rc)
        parsed_actions = self._dedupe_action_names(parsed_actions)
        return {
            "actions": parsed_actions,
            "reviews_count": revision_ct,
        }

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

    def _get_human_feedback_node(self, state: ParserState, config=None) -> dict:
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
        self, state: ParserState, config=None
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

    def _pick_functions_node(self, state: ParserState, config=None) -> dict:
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


class ActionNLParserWithFunctions(ActionNLParser):
    """Sketch generation + function picking in ONE LLM call: the library is shown to
    the sketch LLM and each action carries its ``selected_functions``. With
    ``fn_coarsening`` a function covering several adjacent steps licenses one coarse
    action. With an empty library the atomic prompt of :class:`ActionNLParser` is used.
    """

    def compile(self) -> None:
        graph = StateGraph(
            ParserState, input_schema=QueryInState, output_schema=QueryOutState
        )

        graph.add_node("clarification_check", self._clarification_check_node)
        graph.add_node("get_clarification", self._get_clarification_node)
        graph.add_node("draft_sketch", self._draft_sketch_node)
        graph.add_node("get_human_review", self._get_human_feedback_node)

        graph.add_edge(START, "clarification_check")
        graph.add_conditional_edges(
            "clarification_check",
            self._decide_clarification_needed_edge,
            {"clarify": "get_clarification", "next": "draft_sketch"},
        )
        graph.add_edge("get_clarification", "clarification_check")
        graph.add_edge("draft_sketch", "get_human_review")
        graph.add_conditional_edges(
            "get_human_review",
            self._decide_revision_feedback_edge,
            {"revise": "draft_sketch", "next": END},
        )

        self.state_graph = graph.compile()
        logger.info("State graph (with-functions) compiled successfully.")

    def _draft_sketch_node(self, state: ParserState, config=None) -> dict:
        """Fused sketch + function-picking sketch generation (one LLM call)."""
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
            # Revisions use the no-functions prompt; picks carry over on unchanged actions.
            p = format_revision_prompt(
                sketch=str(previous_sketch),
                human_feedback=str(last_feedback),
                schemas=df_descriptions,
            )
            response = self._invoke_structured(
                p, llm=self.revision_llm, schema=ActionSketchResponse
            )
            parsed_actions = []
            for item in response.actions:
                prev = previous_by_name.get(item.name)
                parsed_actions.append(
                    Action(
                        name=item.name,
                        action=item.action,
                        inputs=item.inputs,
                        output=item.output,
                        output_type=item.output_type,
                        op_kind=item.op_kind,
                        selected_functions=(prev.selected_functions if prev else []),
                    )
                )
            revision_ct += 1
        else:
            functions_block = self._fn_manager.render_functions_summary()
            valid_names = set(self._fn_manager.discover_functions().keys())
            if not valid_names:
                # Empty library: nothing can steer or coarsen, so use the atomic prompt.
                p = format_action_query_sketch_prompt(
                    question=q_in, schemas=df_descriptions
                )
                response = self._invoke_structured(
                    p, llm=self.sketch_llm, schema=ActionSketchResponse
                )
                parsed_actions = [
                    Action(
                        name=item.name,
                        action=item.action,
                        inputs=item.inputs,
                        output=item.output,
                        output_type=item.output_type,
                        op_kind=item.op_kind,
                        selected_functions=[],
                    )
                    for item in response.actions
                ]
            else:
                p = format_action_query_sketch_with_functions_prompt(
                    question=q_in,
                    functions_block=functions_block,
                    schemas=df_descriptions,
                    fn_coarsening=self.fn_coarsening,
                )
                response = self._invoke_structured(
                    p, llm=self.sketch_llm, schema=ActionSketchWithFunctionsResponse
                )
                parsed_actions = [
                    Action(
                        name=item.name,
                        action=item.action,
                        inputs=item.inputs,
                        output=item.output,
                        output_type=item.output_type,
                        op_kind=item.op_kind,
                        selected_functions=[
                            n
                            for n in (item.selected_functions or [])
                            if n in valid_names
                        ],
                    )
                    for item in response.actions
                ]

        parsed_actions = self._inject_populate_actions(parsed_actions, rc)
        parsed_actions = self._dedupe_action_names(parsed_actions)
        return {
            "actions": parsed_actions,
            "reviews_count": revision_ct,
        }
