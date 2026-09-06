"""Centralized configuration for the KathDB pipeline.

:class:`KathDBConfig` has two groups of settings:

* **Basic settings** — also accepted as keyword arguments by :class:`kathdb.KathDB`;
  what most users need: which models, human in the loop, optimizer on/off, the
  function library.
* **Advanced settings** — edit the defaults in this file (or pass a
  ``KathDBConfig``); they tune the internals: parser variant, demand propagation,
  ranking width, group-size cap, profiling sample, prompt details, retries, the worker.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from langchain_core.language_models import BaseChatModel

DEFAULT_PLANNER_MODEL: str = "anthropic/claude-opus-5"
DEFAULT_AI_OP_MODEL: str = "openai/gpt-4o-mini"
DEFAULT_LLM_TEMPERATURE: float = 0.0
DEFAULT_AI_OP_TEMPERATURE: float = 0.0

# Providers understood by :func:`make_llm` (the part before the slash in a model id).
LLM_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "google", "azure_anthropic")

PARSER_TYPES: tuple[str, ...] = (
    "action",
    "action_with_functions",
    "action_with_functions_with_coarsening",
)

__all__ = [
    "KathDBConfig",
    "DEFAULT_PLANNER_MODEL",
    "DEFAULT_AI_OP_MODEL",
    "DEFAULT_LLM_TEMPERATURE",
    "DEFAULT_AI_OP_TEMPERATURE",
    "LLM_PROVIDERS",
    "PARSER_TYPES",
    "make_llm",
    "split_model_id",
]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def split_model_id(model_id: str) -> tuple[str, str]:
    """``"provider/model"`` -> ``(provider, model)``; a bare model name means OpenAI."""
    provider, sep, model = model_id.partition("/")
    if not sep:
        return "openai", model_id
    provider = provider.lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider {provider!r} in model id {model_id!r}. "
            f"Choose from {list(LLM_PROVIDERS)}."
        )
    return provider, model


def make_llm(
    model_id: str, *, temperature: float = DEFAULT_LLM_TEMPERATURE
) -> BaseChatModel:
    """Instantiate a LangChain chat model from a ``provider/model`` id.

    Credentials come from the provider's usual environment variable
    (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``);
    ``azure_anthropic/<deployment>`` reads ``AZURE_ANTHROPIC_ENDPOINT`` and
    ``AZURE_ANTHROPIC_API_KEY`` and does not forward ``temperature``.
    """
    provider, model = split_model_id(model_id)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    # azure_anthropic
    import os

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        base_url=os.environ["AZURE_ANTHROPIC_ENDPOINT"].rstrip("/"),
        api_key=os.environ["AZURE_ANTHROPIC_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class KathDBConfig:
    """All KathDB settings: basic (user-facing) and advanced (edit here)."""

    # ======================================================================
    # Basic settings (also keyword arguments of ``KathDB(...)``)
    # ======================================================================

    # Model KathDB reasons with (parsing, planning, optimizing, code generation), as
    # ``provider/model``; needs that provider's API key in the environment.
    planner_model: str = DEFAULT_PLANNER_MODEL
    # LiteLLM model id the GENERATED code calls for each record's semantic operation
    # (classify an image, judge a review, ...).
    ai_op_model: str = DEFAULT_AI_OP_MODEL
    # True: ask clarification questions, show the drafted plan for review, and ask
    # before saving a function or persisting a result table. False: fully automatic.
    human_in_the_loop: bool = False
    # Run the grouping optimizer (fuse operators so generated code can push filters
    # ahead of model calls and stop early). False = execute the atomic plan as-is.
    logical_rewrite: bool = True
    # Let the generated code batch, parallelize, cascade and cache model calls.
    # False pins one model call per item.
    phy_opt: bool = False
    # Let the planner and code generator use hand-written functions from
    # ``pre_built_fn/`` (ships empty — add your own).
    prebuilt_functions: bool = True
    # Reuse functions saved from prior queries (``generated_fn/``) and save new ones;
    # least-used functions are evicted beyond ``max_generated_functions``.
    generated_functions: bool = True
    max_generated_functions: int = 10
    # Existing conda environment to run the generated code in. None = provision one
    # from ``requirements.txt``.
    worker_env: str | None = None
    # Worker processes: how many independent plan operators execute at the same time
    # (an operator starts as soon as its inputs are ready).
    num_executor_workers: int = 1

    # ======================================================================
    # Advanced settings (edit the defaults here)
    # ======================================================================

    # -- Parser --
    # "action": function-blind atomic sketch, library functions picked per action after;
    # "action_with_functions": the sketch LLM sees the library and picks while sketching;
    # "action_with_functions_with_coarsening": as above, and a function covering several
    # adjacent steps licenses one coarse action. With an empty library all behave alike.
    parser_type: str = "action_with_functions_with_coarsening"
    max_parser_clarifications: int = 5
    max_parser_revisions: int = 3

    # -- Demand propagation (plan annotation) --
    # Stamp every operator with the columns / value constraints its consumers need
    # (column pruning, SEMANTIC -> RELATIONAL rewrites). False skips the pass.
    demand_propagation: bool = True
    # Plans with at most this many actions are annotated in ONE LLM call over the whole
    # DAG; larger plans go top-down, one BFS level per LLM round.
    demand_propagation_one_shot_max_actions: int = 15

    # -- Grouping optimizer --
    # Candidate groupings ranked per LLM call; more candidates run a knockout tournament.
    grouping_rank_k: int = 10
    # Max atomic operators fused into one group. None = no cap.
    grouping_max_group_size: int | None = 5
    # Run the atomic plan's code on a sample at plan time so the ranker sees measured
    # cardinalities / selectivities (model calls on ``grouping_sample_rows`` rows).
    grouping_base_plan_profiling: bool = True
    grouping_sample_rows: int = 50

    # -- Code generation --
    # Generated code attaches images with ``detail="low"`` (cheap resolution).
    image_quality_low_ai_op: bool = True
    # Distinct values per column shown in code-gen prompts.
    distinct_value_sample_k: int = 25
    # Parallel code-generation LLM calls (not execution; see num_executor_workers).
    codegen_concurrency: int = 1
    # Per-stage model overrides (``provider/model``; None = ``planner_model``).
    parser_llm_model: str | None = None
    plan_gen_llm_model: str | None = None
    executor_generation_llm_model: str | None = None
    executor_diagnosis_llm_model: str | None = None
    executor_revision_llm_model: str | None = None
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    ai_op_temperature: float = DEFAULT_AI_OP_TEMPERATURE

    # -- Retry budgets --
    max_plan_gen_retries: int = 3
    max_codegen_retries: int = 3

    # -- Function library --
    # Where saved functions live. None = ``<package>/generated_fn``.
    generated_fn_dir: str | None = None
    save_function_timeout_sec: float = 180.0

    # -- Worker (sandboxed subprocess that runs the generated code) --
    # Requirements installed into a newly provisioned worker env. None = the
    # package's ``worker/requirements.txt``. Ignored when ``worker_env`` is set.
    requirements_path: str | None = None
    remove_worker_env_on_close: bool = False
    worker_connect_timeout_s: float = 180.0
    # Wall-clock budget for one execution of generated code; on expiry the worker is
    # killed and respawned.
    worker_exec_timeout_s: float = 1800.0
    # Directory the generated scripts are staged in. None = a temp dir.
    runtime_dir: str | None = None

    # -- Logging --
    log_level: str | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on invalid values."""
        split_model_id(self.planner_model)
        for name in (
            "parser_llm_model",
            "plan_gen_llm_model",
            "executor_generation_llm_model",
            "executor_diagnosis_llm_model",
            "executor_revision_llm_model",
        ):
            if getattr(self, name):
                split_model_id(getattr(self, name))
        if self.parser_type not in PARSER_TYPES:
            raise ValueError(
                f"parser_type={self.parser_type!r} must be one of {list(PARSER_TYPES)}"
            )
        for name in (
            "human_in_the_loop",
            "logical_rewrite",
            "phy_opt",
            "prebuilt_functions",
            "generated_functions",
            "grouping_base_plan_profiling",
            "image_quality_low_ai_op",
            "demand_propagation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if (
            not isinstance(self.grouping_rank_k, int)
            or isinstance(self.grouping_rank_k, bool)
            or self.grouping_rank_k < 2
        ):
            raise ValueError(
                f"grouping_rank_k must be an int >= 2, got {self.grouping_rank_k!r}"
            )
        mgs = self.grouping_max_group_size
        if mgs is not None and (
            not isinstance(mgs, int) or isinstance(mgs, bool) or mgs < 1
        ):
            raise ValueError(
                f"grouping_max_group_size must be None or an int >= 1, got {mgs!r}"
            )
        n1 = self.demand_propagation_one_shot_max_actions
        if not isinstance(n1, int) or isinstance(n1, bool) or n1 < 1:
            raise ValueError(
                f"demand_propagation_one_shot_max_actions must be an int >= 1, got {n1!r}"
            )
        if self.grouping_sample_rows < 1:
            raise ValueError(
                f"grouping_sample_rows must be >= 1, got {self.grouping_sample_rows}"
            )
        if (
            not isinstance(self.max_generated_functions, int)
            or isinstance(self.max_generated_functions, bool)
            or self.max_generated_functions < 1
        ):
            raise ValueError(
                "max_generated_functions must be an int >= 1, got "
                f"{self.max_generated_functions!r}"
            )
        if (
            not isinstance(self.num_executor_workers, int)
            or isinstance(self.num_executor_workers, bool)
            or self.num_executor_workers < 1
        ):
            raise ValueError(f"num_executor_workers must be an int >= 1, got {self.num_executor_workers!r}")
        if self.codegen_concurrency < 1:
            raise ValueError(
                f"codegen_concurrency must be >= 1, got {self.codegen_concurrency}"
            )
        for name in ("llm_temperature", "ai_op_temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.worker_exec_timeout_s <= 0:
            raise ValueError(
                f"worker_exec_timeout_s must be > 0, got {self.worker_exec_timeout_s}"
            )

    def get_llm(self, stage: str) -> BaseChatModel:
        """Return the chat model for *stage*, resolving per-stage overrides.

        *stage* is one of ``"parser"``, ``"plan_gen"``, ``"executor_generation"``,
        ``"executor_diagnosis"``, ``"executor_revision"``.
        """
        model_id = getattr(self, f"{stage}_llm_model", None) or self.planner_model
        return make_llm(model_id, temperature=self.llm_temperature)

    def update(self, **overrides: Any) -> set[str]:
        """Apply *overrides* and re-validate; returns the changed field names.

        Validation failure rolls back every applied override and re-raises.
        """
        valid = {f.name for f in fields(self)}
        changed: set[str] = set()
        previous: dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in valid:
                raise TypeError(f"Unknown config key: {key!r}")
            if getattr(self, key) != value:
                previous[key] = getattr(self, key)
                setattr(self, key, value)
                changed.add(key)
        if changed:
            try:
                self.validate()
            except Exception:
                for key, value in previous.items():
                    setattr(self, key, value)
                raise
        return changed
