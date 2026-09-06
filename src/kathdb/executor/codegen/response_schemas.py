"""Pydantic response schemas for the codegen module."""

from __future__ import annotations

from pydantic import BaseModel, Field


__all__ = ["FailureDiagnosisResponse", "ConstrainedCodeGenerationResponse"]


class FailureDiagnosisResponse(BaseModel):
    """Diagnosis of an execution failure: reasoning + guidance for regenerating the function."""

    reasoning: str = Field(
        description="Brief explanation of what went wrong.",
    )
    new_function_guidance: str = Field(
        description=("What the regenerated implementation should do differently."),
    )


class ConstrainedCodeGenerationResponse(BaseModel):
    """Code-generation response: the function source + the save-worthiness verdict."""

    code: str = Field(
        description=(
            "The complete Python function implementation. Must be valid, "
            "executable Python code. Do NOT WRAP this field's value in a "
            "markdown code fence (no leading ```python and no trailing ```). "
            "Use ONLY packages from the provided available libraries list. "
            "The function signature MUST take only DataFrame relations as "
            "inputs; hardcode any per-query literals inline in the body. "
            "Do NOT add non-DataFrame named arguments to the function "
            "signature."
        )
    )
    new_fn_worth_saving_reason: str = Field(
        default="",
        description=(
            "One sentence justifying the next field, for THIS code: "
            "(1) does it have non-trivial control flow — not a one-liner or a "
            "thin wrapper around a single library/operator call? and (2) does it "
            "encode a token-cutting execution optimization that issues FEWER "
            "model calls than the naive operator version (early-exit / "
            "short-circuit / pre-filter / cascade)? Name which holds and which "
            "fails; the boolean follows from this. Judge ONLY these two — do NOT "
            "cite hardcoded query literals (a threshold, a prompt string) or "
            "'too query-specific' as a failure; a coding agent runs afterward and "
            "generalizes the function (lifts those literals into parameters)."
        ),
    )
    new_fn_worth_saving: bool = Field(
        default=False,
        description=(
            "Save this function as a reusable KathDB function? Set TRUE only "
            "when BOTH conditions named in the reason above hold: non-trivial "
            "control flow AND a token-cutting execution optimization. A "
            "multi-operator fusion built around an early-exit / short-circuit "
            "loop is the canonical TRUE case. Thin glue around one "
            "already-imported function fails the control-flow condition -> "
            "FALSE. Judge ONLY those two structural conditions: do NOT downgrade "
            "to FALSE because the code hardcodes query-specific literals (a "
            "threshold like 2 or 3, a prompt string) — a coding agent runs "
            "afterward and generalizes the function (lifts those literals into "
            "parameters), so query-specific constants are EXPECTED and are NEVER "
            "a reason for FALSE. Saving is skipped when FALSE."
        ),
    )
