"""Semantic map operator."""

from __future__ import annotations

import pandas as pd

from kathdb.common.model_call import call_model, run_batch
from kathdb.common.logger import get_logger
from kathdb.pre_built_fn._common import (
    build_row_prompt_and_media,
    parse_prompt_columns,
)

logger = get_logger(__name__)

# Single source of truth; fn.md + spec.py are GENERATED from this + the typed
# signature below (kathdb.common.fn_contract). Do not hand-edit them.
CONTRACT = {
    "purpose": "Add one new column by applying an LLM prompt to each row (transform / extract / classify).",
    "params": {
        "df": "Input rows.",
        "prompt": (
            "Template using {col} placeholders (single braces; each col must exist in df); ask for a "
            "single-line plain-text answer. Do NOT call .format() yourself — substitution is per-row."
        ),
        "out_col": "Name of the new column (use a descriptive name, e.g. 'sentiment', not the default).",
        "modality_map": (
            "Map media columns to 'image'/'audio'; the column must also appear as a {placeholder}. "
            "Omit for text-only."
        ),
    },
    "sys_params": ["model", "max_concurrency", "image_detail", "reasoning_effort"],
    "output": "All input columns unchanged, plus one new text column `out_col` with the per-row LLM response.",
    "example": 'result = sem_map(df, prompt="Summarize in one sentence: {text}", out_col="summary")',
    "cost": "O(N) LLM calls, no early-exit. If downstream limits discard most rows, process only the needed rows.",
    "use_when": "derive a new per-row column via LLM (summarize / classify / extract).",
    "not_when": "the value is computable with plain pandas/SQL.",
}


def sem_map(
    df: pd.DataFrame,
    prompt: str,
    out_col: str = "sem_map",
    modality_map: dict[str, str] | None = None,
    model: str = "gpt-4o-mini",
    max_concurrency: int = 20,
    image_detail: str = "low",
    reasoning_effort: str = "minimal",
) -> pd.DataFrame:
    """Add one output column by asking the model to return plain text."""
    out = df.copy()
    if df.empty:
        out[out_col] = []
        return out
    columns = parse_prompt_columns(prompt)
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' referenced in prompt not found in df")

    full_prompts: list[str] = []
    image_lists: list[list[str]] = []
    audio_lists: list[list[tuple[str, str]]] = []
    for _, row in df.iterrows():
        text, imgs, auds = build_row_prompt_and_media(
            prompt, row, columns, modality_map
        )
        full_prompts.append(text)
        image_lists.append(imgs)
        audio_lists.append(auds)

    fns = [
        (
            lambda p=p, imgs=imgs, auds=auds: call_model(
                p,
                model,
                imgs + auds,
                image_detail=image_detail,
                reasoning_effort=reasoning_effort,
            )
        )
        for p, imgs, auds in zip(full_prompts, image_lists, audio_lists)
    ]
    out[out_col] = run_batch(fns, max_concurrency)
    return out
